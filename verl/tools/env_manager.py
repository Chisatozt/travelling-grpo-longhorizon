"""Lifecycle management for the project's single training environment.

The training and evaluation contract is TravelGym-only. Keeping the manager
focused on that environment prevents a stale dataset or tool configuration
from importing an unrelated Gym at rollout time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)
TRAVEL_GYM_NAME = "TravelGym"


class EnvironmentManager:
    """Manage persistent TravelGym instances for multi-turn conversations."""

    def __init__(self) -> None:
        self._environments: Dict[str, Any] = {}
        self._env_configs: Dict[str, Dict[str, Any]] = {}

    def create_environment(
        self,
        request_id: str,
        env_name: str = TRAVEL_GYM_NAME,
        **kwargs: Any,
    ) -> str:
        """Create and store one TravelGym instance for ``request_id``."""
        if env_name != TRAVEL_GYM_NAME:
            raise ValueError(
                f"This project supports only {TRAVEL_GYM_NAME}; received {env_name!r}"
            )
        if request_id in self._environments:
            logger.warning("Environment for request_id %s already exists", request_id)
            return request_id

        env = self._create_travelgym_environment(**kwargs)
        self._environments[request_id] = env
        self._env_configs[request_id] = {"env_name": TRAVEL_GYM_NAME, "kwargs": kwargs}
        logger.info("Created %s environment for request %s", TRAVEL_GYM_NAME, request_id)
        return request_id

    def get_environment(self, request_id: str) -> Optional[Any]:
        """Return the environment associated with a conversation."""
        return self._environments.get(request_id)

    def release_environment(self, request_id: str) -> None:
        """Close and forget a conversation's environment."""
        env = self._environments.pop(request_id, None)
        self._env_configs.pop(request_id, None)
        if env is not None and hasattr(env, "close"):
            env.close()
        if env is not None:
            logger.info("Released %s environment for request %s", TRAVEL_GYM_NAME, request_id)

    @staticmethod
    def _create_travelgym_environment(**kwargs: Any) -> Any:
        """Build a TravelGym configured for terminal-only public scoring."""
        import travelgym

        env_config = travelgym.get_default_config()
        env_config.max_steps = int(kwargs.get("max_turns", 20))
        env_config.data_mode = "single"
        scenario_id = kwargs.get("id")
        if scenario_id is not None:
            env_config.data_source = scenario_id
        env_config.model_name = (
            kwargs.get("model_name")
            or os.environ.get("USER_MODEL_NAME")
            or os.environ.get("ACTOR_MODEL_NAME")
            or env_config.model_name
        )
        env_config.one_choice_per_aspect = True
        env_config.require_action_before_answer = False
        env_config.reward_version = "travelgym-terminal-v1"

        env = travelgym.TravelEnv(config=env_config)
        env.reset()
        return env


_env_manager = EnvironmentManager()


def get_environment_manager() -> EnvironmentManager:
    """Return the process-wide TravelGym manager used by ``InteractTool``."""
    return _env_manager
