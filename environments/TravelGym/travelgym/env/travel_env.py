"""TravelGym public protocol with a selectable private User Simulator.

Search and Answer remain deterministic protocol operations. In formal GRPO,
Action uses the same question-aware DeepSeek judge/response policy as Teacher
trajectory collection. The local sequential simulator is retained only for
explicit offline diagnostics.
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping

from .user_simulator import (
    UserSimulatorError,
    evaluate_user_question,
    initialize_preferences,
    new_user_telemetry,
)

try:  # Gymnasium is optional for offline unit tests.
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised only in minimal installs
    gym = None


_ASPECT_BY_PREFIX = {
    "F": "flight",
    "H": "hotel",
    "A": "apartment",
    "C": "rental_car",
    "R": "restaurant",
}
_ASPECT_HINTS = {
    "flight": ("flight", "airline", "airport"),
    "hotel": ("hotel",),
    "apartment": ("apartment",),
    "rental_car": ("rental car", "car rental", "rental vehicle"),
    "restaurant": ("restaurant", "dining", "reservation"),
}
_ACTION_RE = re.compile(r"^\s*\[(search|action|answer|finish)\](?:\s*(.*))?$", re.IGNORECASE | re.DOTALL)
_OPTION_ID_RE = re.compile(r"^[A-Za-z]\d+$")
_PRIVATE_ID_RE = re.compile(r"\bP\d+\b", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _safe_public_text(value: Any) -> str:
    """Remove accidental private preference IDs from public text."""
    text = str(value or "")
    return _PRIVATE_ID_RE.sub("that preference", text).strip()


def _json_public(value: Any) -> Any:
    """Copy candidate attributes while dropping simulator-only explanations."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_public(child)
            for key, child in value.items()
            if str(key).casefold() not in {"type", "reason", "best_id_thought"}
        }
    if isinstance(value, list):
        return [_json_public(child) for child in value]
    if isinstance(value, tuple):
        return [_json_public(child) for child in value]
    return value


def _load_json(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            item = copy.deepcopy(dict(value))
            item.setdefault("id", str(key))
            result[str(key)] = item
    return result


BaseEnv = gym.Env if gym is not None else object


class TravelEnv(BaseEnv):
    """Gymnasium-compatible local simulator for the TravelGym JSON corpus."""

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        if config is None:
            from ..config import get_default_config

            config = get_default_config()
        elif isinstance(config, Mapping):
            from ..config import TravelGymConfig

            config = TravelGymConfig.from_dict(dict(config))
        self.config = config
        if hasattr(self.config, "validate"):
            self.config.validate()
        self._rng = random.Random(getattr(self.config, "seed", None))
        self._tasks = self._load_tasks()
        if not self._tasks:
            raise RuntimeError("TravelGym data files are missing or unreadable")
        self._model_client = None  # evaluator may inject a shared API client
        self._request_semaphore = None
        self._model_max_attempts = 3
        self._telemetry = None
        self._api_telemetry = new_user_telemetry()
        self._task_id: str | None = None
        self._task: dict[str, Any] | None = None
        self._steps = 0
        self._current_aspect: str | None = None
        self._searched: set[str] = set()
        self._answered: set[str] = set()
        self._visible: dict[str, list[str]] = {}
        self._action_counts: dict[str, int] = {}
        self._answers: dict[str, str] = {}
        self._evidence_seen: dict[str, int] = {}
        self._remaining_preferences: list[dict[str, str]] = []
        self._elicited_preference_ids: set[str] = set()
        self._agent_elicited_preference_ids: set[str] = set()
        self._simulator_proactive_preference_ids: set[str] = set()
        self._action_questions: dict[str, set[str]] = {}
        self._no_gain_action_counts: dict[str, int] = {}
        self._useful_action_count = 0
        self._redundant_action_count = 0
        self._duplicate_action_count = 0
        self._simulator_history: list[dict[str, str]] = []
        self._nonpreference_times = 0
        self._reward_valid = True
        self._simulator_error: str | None = None
        self._invalid_calls = 0
        self._wrong_answers = 0
        self._history: list[dict[str, str]] = []
        self._terminated = False
        self._truncated = False
        self._termination_reason: str | None = None
        self._terminal_report: dict[str, Any] | None = None

    def _load_tasks(self) -> dict[str, dict[str, Any]]:
        paths: list[Path] = []
        configured = getattr(self.config, "data_path", None)
        if configured:
            path = Path(str(configured))
            if path.is_file():
                paths.append(path)
        paths.extend(sorted(_data_dir().glob("travelgym_data_*.json")))
        tasks: dict[str, dict[str, Any]] = {}
        for path in paths:
            for key, task in _load_json(path).items():
                # The scenario IDs include their requested aspect counts and
                # are unique across the shipped variants.  Keep the first
                # copy if a custom data_path duplicates a built-in record.
                tasks.setdefault(key, task)
        return tasks

    @property
    def _aspects(self) -> list[str]:
        if not self._task:
            return []
        raw = self._task.get("dimensions") or []
        return [str(value) for value in raw if str(value) in self._task]

    def _select_task_id(self) -> str:
        mode = str(getattr(self.config, "data_mode", "random"))
        source = getattr(self.config, "data_source", "random")
        if mode == "single":
            candidates = [str(source)]
        elif mode == "list":
            candidates = [str(value) for value in source]
        else:
            candidates = list(self._tasks)
        available = [key for key in candidates if key in self._tasks]
        if not available:
            raise KeyError(f"TravelGym scenario not found: {source!r}")
        return available[0] if mode in {"single", "list"} else self._rng.choice(available)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(seed)
        self._task_id = self._select_task_id()
        self._task = copy.deepcopy(self._tasks[self._task_id])
        self._steps = 0
        self._current_aspect = self._aspects[0] if self._aspects else None
        self._searched = set()
        self._answered = set()
        self._visible = {}
        self._action_counts = {aspect: 0 for aspect in self._aspects}
        self._answers = {}
        self._evidence_seen = {aspect: 0 for aspect in self._aspects}
        self._remaining_preferences = initialize_preferences(self._task)
        self._elicited_preference_ids = set()
        self._agent_elicited_preference_ids = set()
        self._simulator_proactive_preference_ids = set()
        self._action_questions = {aspect: set() for aspect in self._aspects}
        self._no_gain_action_counts = {aspect: 0 for aspect in self._aspects}
        self._useful_action_count = 0
        self._redundant_action_count = 0
        self._duplicate_action_count = 0
        self._simulator_history = []
        self._nonpreference_times = 0
        self._reward_valid = True
        self._simulator_error = None
        self._api_telemetry = new_user_telemetry()
        self._invalid_calls = 0
        self._wrong_answers = 0
        self._terminated = False
        self._truncated = False
        self._termination_reason = None
        self._terminal_report = None
        initial = _safe_public_text(self._task.get("initial_description") or self._task.get("scenario"))
        self._history = [{"role": "user", "content": initial}]
        feedback = (
            f"{initial}\n"
            "Travel planning is ready. Start with Search for one aspect at a time.\n"
            + self._state_lines()
        )
        return self._observation(feedback), self._public_info()

    def _state_lines(self) -> str:
        current = self._current_aspect or "none"
        searched = ", ".join(aspect for aspect in self._aspects if aspect in self._searched) or "none"
        answered = ", ".join(aspect for aspect in self._aspects if aspect in self._answered) or "none"
        visible = ", ".join(self._visible.get(current, [])) if current else "none"
        counts = ", ".join(f"{aspect}={self._action_counts.get(aspect, 0)}" for aspect in self._aspects) or "none"
        return (
            f"Current aspect: {current}\n"
            f"Searched aspects: {searched}\n"
            f"Answered aspects: {answered}\n"
            f"Visible option IDs for current aspect: {visible}\n"
            f"Action count by aspect: {counts}"
        )

    def _public_info(self) -> dict[str, Any]:
        return {
            "current_aspect": self._current_aspect,
            "searched_aspects": [aspect for aspect in self._aspects if aspect in self._searched],
            "visible_option_ids": list(self._visible.get(self._current_aspect or "", [])),
            "action_count_by_aspect": dict(self._action_counts),
            "answered_aspects": [aspect for aspect in self._aspects if aspect in self._answered],
            "public_conversation_history": copy.deepcopy(self._history),
        }

    def get_turn_credit_snapshot(self) -> dict[str, Any]:
        """Return aggregate trainer-only state without private IDs or labels."""
        correct = 0
        best = 0
        legal = 0
        for aspect in self._aspects:
            answer = self._answers.get(aspect, "").upper()
            info = self._task.get(aspect, {}) if self._task else {}
            correct_ids = {str(value).upper() for value in info.get("correct_ids", [])}
            best_id = str(info.get("best_id", "")).upper()
            correct += int(bool(answer) and answer in correct_ids)
            best += int(bool(answer) and answer == best_id)
            legal += int(
                bool(answer)
                and aspect in self._searched
                and self._action_counts.get(aspect, 0) > 0
            )
        return {
            "step": int(self._steps),
            "current_aspect": str(self._current_aspect or ""),
            "searched_count": len(self._searched),
            "answered_count": len(self._answered),
            "action_count": sum(self._action_counts.values()),
            "elicited_preference_count": len(self._agent_elicited_preference_ids),
            "useful_action_count": int(self._useful_action_count),
            "redundant_action_count": int(self._redundant_action_count),
            "duplicate_action_count": int(self._duplicate_action_count),
            "invalid_call_count": int(self._invalid_calls),
            "wrong_answer_count": int(self._wrong_answers),
            "correct_answer_count": correct,
            "best_answer_count": best,
            "legal_answer_count": legal,
            "reward_valid": bool(self._reward_valid),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "termination_reason": str(self._termination_reason or ""),
        }

    def build_turn_credit_event(
        self,
        before: Mapping[str, Any],
        *,
        choice: str,
    ) -> dict[str, Any]:
        """Describe one transition for trainer-side causal credit routing."""
        after = self.get_turn_credit_snapshot()

        def delta(name: str) -> int:
            return max(0, int(after.get(name, 0)) - int(before.get(name, 0)))

        invalid = delta("invalid_call_count") > 0
        runtime_failure = bool(before.get("reward_valid", True)) and not bool(
            after.get("reward_valid", True)
        )
        choice = str(choice or "").casefold()
        accepted = not invalid and not runtime_failure
        if runtime_failure:
            outcome = "runtime_failure"
        elif invalid:
            outcome = "invalid_call"
        elif choice == "finish":
            outcome = "finish"
        else:
            outcome = f"accepted_{choice or 'unknown'}"
        action_delta = delta("action_count")
        useful_delta = delta("useful_action_count")
        return {
            "choice": choice,
            "aspect": str(before.get("current_aspect", "")),
            "outcome": outcome,
            "accepted": accepted,
            "invalid_call": invalid or runtime_failure,
            "new_search": delta("searched_count") > 0,
            "new_preference_count": delta("elicited_preference_count"),
            "useful_action": useful_delta > 0,
            "no_gain_action": choice == "action" and action_delta > 0 and useful_delta == 0,
            "duplicate_action": delta("duplicate_action_count") > 0,
            "redundant_action": delta("redundant_action_count") > 0,
            "completed_aspect": delta("answered_count") > 0,
            "correct_answer": delta("correct_answer_count") > 0,
            "best_answer": delta("best_answer_count") > 0,
            "legal_answer": delta("legal_answer_count") > 0,
            "wrong_answer": delta("wrong_answer_count") > 0,
            "terminated": bool(after.get("terminated", False)),
            "truncated": bool(after.get("truncated", False)),
            "termination_reason": str(after.get("termination_reason", "")),
        }

    def register_external_invalid_call(self) -> None:
        """Count a malformed tool call rejected before step() is entered."""
        if not self._terminated and not self._truncated:
            self._steps += 1
        self._invalid_calls += 1
        if self._steps >= int(getattr(self.config, "max_steps", 25)) and not self._terminated:
            self._truncated = True
            self._termination_reason = "max_steps"

    def _observation(self, feedback: str) -> dict[str, Any]:
        public_feedback = _safe_public_text(feedback)
        observation = {"feedback": public_feedback, **self._public_info()}
        try:
            from travel_grpo.evaluation.travel_contract import assert_public_observation

            assert_public_observation(observation)
        except ImportError:  # pragma: no cover - package use outside repository
            pass
        return observation

    def _append_history(self, feedback: str) -> None:
        # Keep the transcript public and compact.  The complete candidate list
        # is in the immediate feedback, while the history only records the
        # public event/result needed by downstream reducers.
        compact = _safe_public_text(feedback).split("\n", 1)[0]
        self._history.append({"role": "tool", "content": compact})

    def _infer_aspect(self, content: str, choice: str) -> tuple[str | None, bool]:
        text = str(content or "")
        if choice == "answer":
            match = re.fullmatch(r"\s*([A-Za-z]\d+)\s*", text)
            if match:
                return _ASPECT_BY_PREFIX.get(match.group(1)[0].upper()), False
        lowered = text.casefold()
        matches = [
            aspect
            for aspect, hints in _ASPECT_HINTS.items()
            if any(hint in lowered for hint in hints)
        ]
        if len(matches) > 1:
            return None, True
        if matches:
            return matches[0], False
        return self._current_aspect, False

    @staticmethod
    def _normalize_action_question(content: str) -> str:
        return re.sub(r"\W+", " ", str(content or "").casefold()).strip()

    def _register_revealed_preferences(
        self,
        agent_ids: tuple[str, ...] | set[str] = (),
        proactive_ids: tuple[str, ...] | set[str] = (),
    ) -> tuple[set[str], set[str]]:
        available_ids = {item["id"] for item in self._remaining_preferences}
        new_agent = set(agent_ids) & available_ids
        new_proactive = (set(proactive_ids) & available_ids) - new_agent
        revealed = new_agent | new_proactive
        if revealed:
            for item in self._remaining_preferences:
                if item["id"] in revealed:
                    pref_aspect = item["aspect"]
                    self._evidence_seen[pref_aspect] = self._evidence_seen.get(pref_aspect, 0) + 1
            self._remaining_preferences = [
                item for item in self._remaining_preferences if item["id"] not in revealed
            ]
        self._elicited_preference_ids.update(revealed)
        self._agent_elicited_preference_ids.update(new_agent)
        self._simulator_proactive_preference_ids.update(new_proactive)
        return new_agent, new_proactive

    def _record_action_outcome(self, aspect: str, content: str, new_agent_ids: set[str]) -> None:
        normalized = self._normalize_action_question(content)
        questions = self._action_questions.setdefault(aspect, set())
        duplicate = bool(normalized and normalized in questions)
        if normalized:
            questions.add(normalized)
        useful = bool(new_agent_ids)
        if useful:
            self._useful_action_count += 1
        else:
            self._no_gain_action_counts[aspect] = self._no_gain_action_counts.get(aspect, 0) + 1
        if duplicate:
            self._duplicate_action_count += 1
        if duplicate or (not useful and self._no_gain_action_counts.get(aspect, 0) > 1):
            self._redundant_action_count += 1

    def _minimum_completion_steps(self) -> int:
        return sum(
            1 + int(aspect not in self._searched)
            for aspect in self._aspects
            if aspect not in self._answered
        )

    def _deadline_blocks_action(self) -> bool:
        max_steps = max(1, int(getattr(self.config, "max_steps", 25)))
        return max_steps - self._steps <= self._minimum_completion_steps()

    def _advance_current(self) -> None:
        self._current_aspect = next(
            (aspect for aspect in self._aspects if aspect not in self._answered),
            None,
        )

    def _search_feedback(self, aspect: str) -> str:
        info = self._task.get(aspect, {}) if self._task else {}
        all_ids = [str(value) for value in info.get("all_ids", [])]
        option_by_id: dict[str, Mapping[str, Any]] = {}
        options = info.get("options", {})
        if isinstance(options, Mapping):
            for group in ("correct", "wrong", "noise"):
                values = options.get(group, [])
                if isinstance(values, list):
                    for option in values:
                        if isinstance(option, Mapping) and option.get("id") is not None:
                            option_by_id[str(option["id"])] = option
        lines = [f"Search results for {aspect}; all candidates are shown:"]
        for option_id in all_ids:
            candidate = _json_public(option_by_id.get(option_id, {"id": option_id}))
            if isinstance(candidate, Mapping):
                candidate = {key: value for key, value in candidate.items() if key != "id"}
            encoded = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"- {option_id}: {encoded}")
        lines.append(self._state_lines())
        return "\n".join(lines)

    def _action_feedback(self, aspect: str) -> tuple[str, tuple[str, ...]]:
        current = next(
            (item for item in self._remaining_preferences if item["aspect"] == aspect),
            None,
        )
        if current is not None:
            evidence = current.get("implicit_elicitation") or "Please use the stated travel priorities."
            return (
                f"The traveler says: {_safe_public_text(evidence)}\n{self._state_lines()}",
                (current["id"],),
            )
        return (
            f"The traveler has no additional preference details for {aspect}; compare the complete Search results.\n{self._state_lines()}",
            (),
        )

    def mark_user_simulator_failure(self, reason: str) -> None:
        """Invalidate an episode after an outer tool/runtime failure."""
        self._reward_valid = False
        self._simulator_error = _safe_public_text(reason)[:120] or "runtime_error"
        self._truncated = True
        self._termination_reason = "user_simulator_api_error"

    def _reject(self, message: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._invalid_calls += 1
        if self._steps >= int(getattr(self.config, "max_steps", 25)) and not self._terminated:
            self._truncated = True
            self._termination_reason = "max_steps"
        self._append_history(message)
        return self._observation(f"{message}\n{self._state_lines()}"), 0.0, self._terminated, self._truncated, self._public_info()

    def step(self, action: str):
        if self._task is None:
            self.reset()
        if self._terminated or self._truncated:
            message = "Tool call rejected: the episode is already finished."
            return self._reject(message)
        match = _ACTION_RE.match(str(action or ""))
        if not match:
            return self._reject("Tool call rejected: invalid tool parameters.")
        choice = match.group(1).casefold()
        content = (match.group(2) or "").strip()
        if choice == "action" and self.config.user_simulator_mode == "deepseek_api":
            raise UserSimulatorError(
                "DeepSeek User Simulator actions require step_async()"
            )
        self._steps += 1
        if choice == "finish":
            self._terminated = True
            self._termination_reason = "finish"
            feedback = f"Travel planning finished.\n{self._state_lines()}"
            self._append_history(feedback)
            return self._observation(feedback), 0.0, True, False, self._public_info()

        if not content:
            return self._reject("Tool call rejected: invalid tool parameters.")
        aspect, ambiguous = self._infer_aspect(content, choice)
        if ambiguous or aspect is None or aspect not in self._aspects:
            return self._reject("Tool call rejected: cross-aspect operation.")
        if self._current_aspect is not None and aspect != self._current_aspect:
            return self._reject("Tool call rejected: cross-aspect operation.")
        if aspect in self._answered:
            return self._reject("Tool call rejected: the aspect was already answered.")

        if choice == "search":
            if aspect in self._searched:
                return self._reject("Tool call rejected: repeated tool call.")
            info = self._task.get(aspect, {}) if self._task else {}
            self._searched.add(aspect)
            self._visible[aspect] = [str(value) for value in info.get("all_ids", [])]
            feedback = self._search_feedback(aspect)
            self._simulator_history.extend([
                {"role": "agent", "content": content},
                {
                    "role": "database",
                    "content": f"Search results for {aspect}; all candidates are shown. ... (skip detailed results here) ...",
                },
            ])
        elif choice == "action":
            if aspect not in self._searched:
                return self._reject("Tool call rejected: action-before-search.")
            if self._deadline_blocks_action():
                return self._reject(
                    "Tool call rejected: remaining turn budget must be used to complete unanswered aspects."
                )
            self._action_counts[aspect] = self._action_counts.get(aspect, 0) + 1
            feedback, agent_ids = self._action_feedback(aspect)
            new_agent, _ = self._register_revealed_preferences(agent_ids=agent_ids)
            self._record_action_outcome(aspect, content, new_agent)
            self._simulator_history.extend([
                {"role": "agent", "content": content, "note": "Offline local simulator diagnostic."},
                {"role": "user", "content": feedback.split("\n", 1)[0]},
            ])
        else:  # answer
            if aspect not in self._searched:
                return self._reject("Tool call rejected: answer-before-search.")
            if not re.fullmatch(r"[A-Za-z]\d+", content):
                return self._reject("Tool call rejected: invalid tool parameters.")
            answer_id = content.upper()
            visible = {value.upper() for value in self._visible.get(aspect, [])}
            if answer_id not in visible:
                return self._reject("Tool call rejected: answer ID was not visible in Search results.")
            self._answers[aspect] = answer_id
            self._answered.add(aspect)
            self._remaining_preferences = [
                item for item in self._remaining_preferences
                if item["aspect"] != aspect
            ]
            info = self._task.get(aspect, {}) if self._task else {}
            if answer_id not in {str(value).upper() for value in info.get("correct_ids", [])}:
                self._wrong_answers += 1
            self._advance_current()
            feedback = f"Your answer was recorded for {aspect}.\n{self._state_lines()}"
            if self._current_aspect is None:
                self._terminated = True
                self._termination_reason = "all_aspects_answered"
        if self._steps >= int(getattr(self.config, "max_steps", 25)) and not self._terminated:
            self._truncated = True
            self._termination_reason = "max_steps"
        self._append_history(feedback)
        return self._observation(feedback), 0.0, self._terminated, self._truncated, self._public_info()

    async def step_async(self, action: str):
        match = _ACTION_RE.match(str(action or ""))
        if (
            self.config.user_simulator_mode != "deepseek_api"
            or match is None
            or match.group(1).casefold() != "action"
        ):
            return self.step(action)
        if self._task is None:
            self.reset()
        if self._terminated or self._truncated:
            return self._reject("Tool call rejected: the episode is already finished.")

        content = (match.group(2) or "").strip()
        self._steps += 1
        if not content:
            return self._reject("Tool call rejected: invalid tool parameters.")
        aspect, ambiguous = self._infer_aspect(content, "action")
        if ambiguous or aspect is None or aspect not in self._aspects:
            return self._reject("Tool call rejected: cross-aspect operation.")
        if self._current_aspect is not None and aspect != self._current_aspect:
            return self._reject("Tool call rejected: cross-aspect operation.")
        if aspect in self._answered:
            return self._reject("Tool call rejected: the aspect was already answered.")
        if aspect not in self._searched:
            return self._reject("Tool call rejected: action-before-search.")
        if self._deadline_blocks_action():
            return self._reject(
                "Tool call rejected: remaining turn budget must be used to complete unanswered aspects."
            )

        self._action_counts[aspect] = self._action_counts.get(aspect, 0) + 1
        available = list(self._remaining_preferences)
        telemetry = self._telemetry if isinstance(self._telemetry, dict) else self._api_telemetry
        try:
            turn = await evaluate_user_question(
                content,
                task_id=str(self._task_id or "unknown"),
                aspect=aspect,
                scenario=str(self._task.get("scenario", "")) if self._task else "",
                history=self._simulator_history,
                available_preferences=available,
                nonpreference_times=self._nonpreference_times,
                elicitation_interval=int(getattr(self.config, "elicitation_interval", 3)),
                config=self.config,
                telemetry=telemetry,
                rng=self._rng,
                model_client=self._model_client,
                request_semaphore=self._request_semaphore,
                max_attempts=max(1, int(self._model_max_attempts)),
            )
        except UserSimulatorError as exc:
            self._reward_valid = False
            self._simulator_error = type(exc.__cause__ or exc).__name__
            self._truncated = True
            self._termination_reason = "user_simulator_api_error"
            feedback = f"The environment operation failed.\n{self._state_lines()}"
            self._append_history(feedback)
            return self._observation(feedback), 0.0, False, True, self._public_info()

        new_agent, _ = self._register_revealed_preferences(
            agent_ids=turn.elicited_preference_ids,
            proactive_ids=turn.proactive_preference_ids,
        )
        self._record_action_outcome(aspect, content, new_agent)
        self._nonpreference_times = turn.nonpreference_times
        self._simulator_history.extend([
            {"role": "agent", "content": content, "note": turn.note},
            {"role": "user", "content": turn.response},
        ])
        feedback = f"{_safe_public_text(turn.response)}\n{self._state_lines()}"
        if self._steps >= int(getattr(self.config, "max_steps", 25)):
            self._truncated = True
            self._termination_reason = "max_steps"
        self._append_history(feedback)
        return self._observation(
            feedback
        ), 0.0, self._terminated, self._truncated, self._public_info()

    def _build_terminal_report(self) -> dict[str, Any]:
        if self._terminal_report is not None:
            return copy.deepcopy(self._terminal_report)
        total = len(self._aspects)
        answered = len(self._answered)
        correct = 0
        best = 0
        legal = 0
        for aspect in self._aspects:
            answer = self._answers.get(aspect, "").upper()
            info = self._task.get(aspect, {}) if self._task else {}
            correct_ids = {str(value).upper() for value in info.get("correct_ids", [])}
            best_ids = {str(info.get("best_id", "")).upper()} if info.get("best_id") else set()
            if answer and answer in correct_ids:
                correct += 1
            if answer and answer in best_ids:
                best += 1
            if answer and aspect in self._searched and self._action_counts.get(aspect, 0) > 0:
                legal += 1
        preferences_total = sum(
            len(values)
            for aspect in self._aspects
            if isinstance(
                values := (self._task.get(aspect, {}).get("preferences", []) if self._task else []),
                list,
            )
        )
        agent_preferences_seen = len(self._agent_elicited_preference_ids)
        proactive_preferences_seen = len(self._simulator_proactive_preference_ids)
        correct_completion = correct / total if total else 0.0
        answer_coverage = answered / total if total else 0.0
        unanswered_count = max(0, total - answered)
        completion_success = float(total > 0 and answered == total and correct == total)
        answer_quality = best / answered if answered else 0.0
        best_answer_rate = best / answered if answered else 0.0
        legal_chain_rate = legal / answered if answered else 0.0
        coverage_adjusted_answer_quality = best / total if total else 0.0
        coverage_adjusted_legal_chain_rate = legal / total if total else 0.0
        hidden_hit_rate = agent_preferences_seen / preferences_total if preferences_total else 1.0
        max_steps = max(1, int(getattr(self.config, "max_steps", 25)))
        efficiency = max(0.0, 1.0 - min(1.0, self._steps / max_steps))
        policy_penalty = min(1.0, 0.05 * self._invalid_calls + 0.10 * self._wrong_answers)
        early_redundant = min(3, self._redundant_action_count)
        later_redundant = max(0, self._redundant_action_count - 3)
        redundant_action_penalty = min(0.60, 0.05 * early_redundant + 0.10 * later_redundant)
        incomplete_penalty = 1.00 * (1.0 - answer_coverage)
        zero_answer_penalty = 0.50 if total > 0 and answered == 0 else 0.0
        max_steps_reached = float(self._termination_reason == "max_steps")
        max_steps_penalty = 0.75 * max_steps_reached
        total_penalty = (
            policy_penalty + redundant_action_penalty + incomplete_penalty
            + zero_answer_penalty + max_steps_penalty
        )
        raw = (
            3.00 * correct_completion
            + 0.30 * coverage_adjusted_answer_quality
            + 0.20 * coverage_adjusted_legal_chain_rate
            + 0.15 * hidden_hit_rate
            + 0.05 * efficiency
            - total_penalty
        )
        raw_terminal_reward = raw / 3.70
        terminal = max(-1.0, min(1.0, raw_terminal_reward))
        if not self._reward_valid:
            terminal = 0.0
        telemetry = self._telemetry if isinstance(self._telemetry, dict) else self._api_telemetry
        user_metrics = {
            key: value for key, value in telemetry.items()
            if str(key).startswith("user_") and isinstance(value, (int, float))
        }
        self._terminal_report = {
            "reward_version": self.config.reward_version,
            "terminal_reward": terminal,
            "raw_terminal_reward": raw_terminal_reward,
            "reward_valid": self._reward_valid,
            "reward_valid_for_training": self._reward_valid,
            "terminal_only": True,
            "correct_completion": correct_completion,
            "completion_success": completion_success,
            "answer_coverage": answer_coverage,
            "unanswered_count": unanswered_count,
            "best_answer_rate": best_answer_rate,
            "answer_quality": answer_quality,
            "legal_chain_rate": legal_chain_rate,
            "coverage_adjusted_answer_quality": coverage_adjusted_answer_quality,
            "coverage_adjusted_legal_chain_rate": coverage_adjusted_legal_chain_rate,
            "hidden_preference_hit_rate": hidden_hit_rate,
            "agent_elicited_preference_count": agent_preferences_seen,
            "proactive_preference_count": proactive_preferences_seen,
            "useful_action_count": self._useful_action_count,
            "no_gain_action_count": sum(self._no_gain_action_counts.values()),
            "redundant_action_count": self._redundant_action_count,
            "duplicate_action_count": self._duplicate_action_count,
            "invalid_call_count": self._invalid_calls,
            "wrong_answer_count": self._wrong_answers,
            "efficiency": efficiency,
            "policy_penalty": policy_penalty,
            "redundant_action_penalty": redundant_action_penalty,
            "incomplete_penalty": incomplete_penalty,
            "zero_answer_penalty": zero_answer_penalty,
            "max_steps_reached": max_steps_reached,
            "max_steps_penalty": max_steps_penalty,
            "total_penalty": total_penalty,
            "termination_reason": self._termination_reason or "rollout_complete",
            "user_simulator_mode": self.config.user_simulator_mode,
            "user_simulator_error": self._simulator_error,
            **user_metrics,
        }
        return copy.deepcopy(self._terminal_report)

    def get_terminal_reward(self) -> float:
        return float(self._build_terminal_report()["terminal_reward"])

    def get_reward_report(self, finalize: bool = False) -> dict[str, Any]:
        del finalize
        return self._build_terminal_report()

    def close(self):
        return None

    def render(self):  # pragma: no cover - human mode is not used in GRPO
        return None


__all__ = ["TravelEnv"]
