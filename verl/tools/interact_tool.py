from typing import Any, Optional, Tuple
import asyncio
import os

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .env_manager import TRAVEL_GYM_NAME, get_environment_manager
from .travel_tool_adapter import (
    TravelToolAdapterError,
    format_environment_action,
    normalize_tool_call,
    sanitize_public_feedback,
)
from .travel_reward_metrics import TRAVEL_REWARD_METRIC_NAMES

class InteractTool(BaseTool):
    """The persistent TravelGym tool used by SGLang multi-turn rollouts.

    ``execute`` forwards only public natural-language feedback to the Actor;
    terminal reward and diagnostics stay in the trainer-side ledger.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._conversation_data = {}  # request_id -> conversation state
        self._env_manager = get_environment_manager()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, env_name: Optional[str] = TRAVEL_GYM_NAME, max_turns: int = 25, **kwargs) -> str:
        """Create a TravelGym instance and initialize conversation state.
        
        Args:
            instance_id: Request ID for the conversation (serves as conversation identifier)
            env_name: Must be ``TravelGym`` (the default)
            max_turns: Maximum number of interaction turns
            **kwargs: Environment-specific configuration
            
        Returns:
            instance_id (request_id)
        """
        if instance_id in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} already exists !!!!!!!!")
            return instance_id
        
        env_name = env_name or TRAVEL_GYM_NAME
        if env_name != TRAVEL_GYM_NAME:
            raise ValueError(
                f"InteractTool supports only {TRAVEL_GYM_NAME}; received {env_name!r}"
            )
        kwargs["max_turns"] = max_turns
        self._env_manager.create_environment(instance_id, env_name, **kwargs)
        initial_context = sanitize_public_feedback(
            self._env_manager.get_initial_context(instance_id)
        )

        # Initialize conversation state (separate from environment)
        self._conversation_data[instance_id] = {
            "history": [],
            "initial_context": initial_context,
            "reward": 0.0,
            # TravelGym keeps correctness labels inside its private reward
            # ledger.  Do not copy rollout ground-truth IDs into this state.
            "ground_truth": None,
            "env_name": TRAVEL_GYM_NAME,
        }
        
        print(f"Created conversation {instance_id} with {env_name} environment")
        return instance_id

    def get_initial_context(self, instance_id: str) -> str:
        """Return reset-time public control state for the first Actor turn."""
        state = self._conversation_data.get(instance_id)
        return str(state.get("initial_context", "")) if state is not None else ""

    async def execute(self, instance_id: str, parameters: dict[str, Any], current_turns=0, **kwargs) -> Tuple[str, float, bool, str, str, dict]:
        """Execute action in the persistent environment.
        
        Args:
            instance_id: Request ID (conversation identifier)
            parameters: Action parameters (choice, content)
            
        Returns:
            (response_text, step_reward, is_terminated, choice, content, metrics)
        """
        
        if instance_id not in self._conversation_data:
            raise ValueError(f"Conversation {instance_id} not found. Call create() first.")
        
        # Get persistent environment
        env = self._env_manager.get_environment(instance_id)
        if env is None:
            raise ValueError(f"Environment for conversation {instance_id} not found")
        
        # Snapshot only aggregate counters.  The derived event is trainer-only
        # and never enters the model-visible observation.
        snapshot_getter = getattr(env, "get_turn_credit_snapshot", None)
        before_credit = snapshot_getter() if snapshot_getter is not None else {}

        # Normalize once at the API boundary.  The adapter does not inspect or
        # filter candidates; it only enforces the public choice/content shape.
        try:
            normalized = normalize_tool_call({"name": "interact_with_env", "arguments": parameters})
            choice, content = normalized["choice"], normalized["content"]
            formatted_action = format_environment_action(normalized)
        except TravelToolAdapterError:
            feedback = "Tool call rejected: invalid tool parameters."
            if hasattr(env, "register_external_invalid_call"):
                env.register_external_invalid_call()
            event_builder = getattr(env, "build_turn_credit_event", None)
            turn_event = (
                event_builder(before_credit, choice="")
                if event_builder is not None
                else {"choice": "", "invalid_call": True, "accepted": False}
            )
            conversation_state = self._conversation_data[instance_id]
            conversation_state["history"].append({"choice": "", "content": "", "observation": {"feedback": feedback}})
            is_done = bool(turn_event.get("terminated", False) or turn_event.get("truncated", False))
            return feedback, 0.0, is_done, "", "", {
                "turn_event": turn_event
            }
        
        try:
            step_timeout = float(os.environ.get("TRAVELGYM_STEP_TIMEOUT", "300"))
            if step_timeout <= 0:
                raise ValueError("TRAVELGYM_STEP_TIMEOUT must be positive")
            observation, reward, terminated, truncated, info = await asyncio.wait_for(
                env.step_async(formatted_action),
                timeout=step_timeout,
            )
        except asyncio.TimeoutError:
            print(f"Environment step timed out for {instance_id} after {step_timeout}s")
            if hasattr(env, "mark_user_simulator_failure"):
                env.mark_user_simulator_failure("outer_step_timeout")
            observation = {"feedback": "The environment operation failed."}
            reward, terminated, truncated, info = 0.0, False, True, {}
        except Exception as exc:
            print(f"Environment step failed for {instance_id}: {type(exc).__name__}")
            if hasattr(env, "mark_user_simulator_failure"):
                env.mark_user_simulator_failure(type(exc).__name__)
            # Return only neutral public text.  Raw exception strings are
            # internal diagnostics and must never enter the Actor transcript.
            observation = {"feedback": "The environment operation failed."}
            reward, terminated, truncated, info = 0.0, False, True, {}

        # Update conversation state
        conversation_state = self._conversation_data[instance_id]
        current_env_name = conversation_state["env_name"]
        conversation_state["reward"] = reward
        conversation_state["history"].append({
            "choice": choice,
            "content": content,
            "observation": observation,
            "reward": reward,
            "info": info
        })
        
        # Format response
        feedback = observation.get("feedback", "") if isinstance(observation, dict) else str(observation)
        feedback = sanitize_public_feedback(feedback)
        # Step rewards are intentionally not part of tool feedback.  TravelGym
        # returns terminal-only Reward through ``calc_reward`` after the
        # trajectory ends; exposing a per-step scalar would leak diagnostics.
        response_text = feedback
        
        is_done = terminated or truncated
        event_builder = getattr(env, "build_turn_credit_event", None)
        turn_event = (
            event_builder(before_credit, choice=choice)
            if event_builder is not None
            else {"choice": choice, "accepted": True}
        )
        print(f"Turn {current_turns}: Executed {choice} in conversation {instance_id} (Env: {current_env_name}), action: {formatted_action}, feedback: {feedback}, reward: {reward}, done: {is_done}")

        return response_text, reward, is_done, choice, content, {
            "turn_event": turn_event
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for the conversation.
        
        Args:
            instance_id: Request ID (conversation identifier)
            
        Returns:
            Final conversation reward
        """
        if instance_id not in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} not found for reward calculation !!!!!!!!")
            return 0.0
        
        conversation_state = self._conversation_data[instance_id]
        
        env = self._env_manager.get_environment(instance_id)
        if env is not None and hasattr(env, "get_terminal_reward"):
            # This is consumed by the trainer, never interpolated into
            # ``execute`` feedback.
            terminal_reward = float(env.get_terminal_reward())
            conversation_state["reward"] = terminal_reward
            return terminal_reward
        return float(conversation_state.get("reward", 0.0))

    async def get_reward_metadata(self, instance_id: str, **kwargs) -> dict:
        """Return trainer-only validity/version metadata before release."""
        env = self._env_manager.get_environment(instance_id)
        if env is None:
            return {
                "reward_valid": False,
                "reward_version": "unknown",
                "terminal_only": False,
            }
        if not hasattr(env, "get_reward_report"):
            return {
                "reward_valid": False,
                "reward_version": "unknown",
                "terminal_only": True,
            }
        report = env.get_reward_report(finalize=True)
        metadata = {
            "reward_valid": bool(report.get("reward_valid_for_training", report.get("reward_valid", False))),
            "reward_version": str(report.get("reward_version", "unknown")),
            "terminal_only": True,
            "termination_reason": report.get("termination_reason"),
            "actor_aspect_extraction_enabled": bool(
                report.get("actor_aspect_extraction_enabled", False)
            ),
            # The plan and validation details are trainer-side diagnostics;
            # they are never copied into tool feedback or Actor messages.
            "actor_aspects": report.get("actor_aspects"),
            "actor_aspects_raw": report.get("actor_aspects_raw"),
            "actor_aspect_extraction_format_error": report.get(
                "actor_aspect_extraction_format_error"
            ),
            "actor_aspect_extraction_invalid_aspects": report.get(
                "actor_aspect_extraction_invalid_aspects", []
            ),
            "actor_aspect_extraction_duplicate_aspects": report.get(
                "actor_aspect_extraction_duplicate_aspects", []
            ),
            "actor_aspect_extraction_direct_grpo_signal": False,
        }
        for key in TRAVEL_REWARD_METRIC_NAMES:
            try:
                metadata[key] = float(report.get(key, 0.0))
            except (TypeError, ValueError):
                metadata[key] = 0.0
        return metadata
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up conversation and environment.
        
        Args:
            instance_id: Request ID (conversation identifier)
        """
        # Clean up conversation state
        if instance_id in self._conversation_data:
            del self._conversation_data[instance_id]
        
        # Clean up environment through manager
        self._env_manager.release_environment(instance_id)
        
