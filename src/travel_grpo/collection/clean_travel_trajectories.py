"""Replay, clean and classify canonical TravelGym trajectories.

The cleaner is intentionally a small offline reducer.  It consumes only the
public transcript and an optional private task sidecar (used for recomputing
the terminal metrics).  It never creates a filtered candidate list and never
adds task labels to the returned messages.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

try:  # package import
    from .travel_canonical import (
        CanonicalError,
        canonical_hash,
        canonicalize_record,
        option_ids,
        tool_call_arguments,
        validate_canonical,
    )
except ImportError:  # direct script import
    from travel_canonical import (
        CanonicalError,
        canonical_hash,
        canonicalize_record,
        option_ids,
        tool_call_arguments,
        validate_canonical,
    )


ASPECT_BY_PREFIX = {"F": "flight", "H": "hotel", "A": "apartment", "C": "rental_car", "R": "restaurant"}
ASPECT_HINTS = {
    "flight": ("flight", "airline", "airport"),
    "hotel": ("hotel",),
    "apartment": ("apartment",),
    "rental_car": ("rental car", "car rental", "rental vehicle"),
    "restaurant": ("restaurant", "dining", "reservation"),
}
_CURRENT_RE = re.compile(r"Current aspect:\s*([A-Za-z_]+|none)", re.IGNORECASE)
_SEARCHED_RE = re.compile(r"Searched aspects:\s*([^\n]*)", re.IGNORECASE)
_ANSWERED_RE = re.compile(r"Answered aspects:\s*([^\n]*)", re.IGNORECASE)
_VISIBLE_RE = re.compile(r"Visible option IDs for current aspect:\s*([^\n]*)", re.IGNORECASE)

RECOVERABLE_KINDS = {
    "action-before-search",
    "answer-before-search",
    "cross-aspect",
    "repeated-search",
    "invalid-parameters",
    "invisible-id",
    "duplicate-answer",
    "vague-action",
}
FATAL_KINDS = {
    "tool_call_json_invalid",
    "assistant_tool_call_missing",
    "tool_call_missing",
    "invalid_choice",
    "invalid_content",
    "assistant_tool_call_truncated",
    "assistant_think_truncated",
    "assistant_think_missing",
    "assistant_content_truncated",
    "tool_call_id_mismatch",
    "tool_observation_misaligned",
    "observation-misaligned",
    "environment-failure",
    "unrepaired-recoverable-error",
    "wrong-terminal-answer",
}


@dataclass
class TaskSpec:
    """Private evaluator view of a TravelGym task."""

    task_id: str | None = None
    aspects: list[str] = field(default_factory=list)
    correct_ids: dict[str, set[str]] = field(default_factory=dict)
    best_ids: dict[str, set[str]] = field(default_factory=dict)
    all_ids: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, task: Mapping[str, Any] | None, task_id: str | None = None) -> "TaskSpec | None":
        if not isinstance(task, Mapping):
            return None
        raw_aspects = task.get("dimensions") or task.get("aspects")
        aspects = [str(value) for value in raw_aspects] if isinstance(raw_aspects, (list, tuple)) else []
        if not aspects:
            for key, value in task.items():
                if key in ASPECT_HINTS and isinstance(value, Mapping):
                    aspects.append(key)
        spec = cls(task_id=str(task_id or task.get("id")) if (task_id or task.get("id")) else None, aspects=aspects)
        containers: list[Mapping[str, Any]] = []
        if isinstance(task.get("preferences"), Mapping):
            containers.append(task["preferences"])
        containers.append(task)
        for aspect in aspects:
            info: Mapping[str, Any] | None = None
            for container in containers:
                candidate = container.get(aspect)
                if isinstance(candidate, Mapping):
                    info = candidate
                    break
            if info is None:
                continue
            correct = info.get("correct_ids") or info.get("correct_options") or []
            all_values = info.get("all_ids") or info.get("option_ids") or []
            best = info.get("best_ids") or info.get("best_id") or []
            if isinstance(best, str):
                best = [best]
            spec.correct_ids[aspect] = {str(value).upper() for value in correct if isinstance(value, (str, int))}
            spec.all_ids[aspect] = {str(value).upper() for value in all_values if isinstance(value, (str, int))}
            spec.best_ids[aspect] = {str(value).upper() for value in best if isinstance(value, (str, int))}
        return spec

    @property
    def has_labels(self) -> bool:
        return bool(self.aspects) and all(aspect in self.correct_ids for aspect in self.aspects)


@dataclass
class ReplayEvent:
    assistant_index: int
    tool_index: int | None
    choice: str | None
    content: str
    aspect: str | None
    feedback: str
    rejected: bool
    error_kind: str | None
    current_before: str | None = None
    current_after: str | None = None
    visible_after: set[str] = field(default_factory=set)
    searched_after: set[str] = field(default_factory=set)
    answered_after: set[str] = field(default_factory=set)
    answer_id: str | None = None
    accepted: bool = False
    repair_for: str | None = None


def _feedback(message: Mapping[str, Any] | None) -> str:
    if not isinstance(message, Mapping):
        return ""
    value = message.get("content", "")
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value or "")


def _aspect_from_text(text: str, choice: str | None = None) -> str | None:
    if choice == "answer":
        ids = option_ids(text)
        if len(ids) == 1:
            return ASPECT_BY_PREFIX.get(ids[0][0])
    lowered = text.casefold()
    matches = [aspect for aspect, hints in ASPECT_HINTS.items() if any(hint in lowered for hint in hints)]
    return matches[0] if len(matches) == 1 else None


def _mentioned_aspects(text: str) -> tuple[str, ...]:
    """Return public aspect mentions without consulting task labels."""
    lowered = str(text or "").casefold()
    found = [
        aspect
        for aspect, hints in ASPECT_HINTS.items()
        if any(hint in lowered for hint in hints)
    ]
    for option_id in option_ids(text):
        aspect = ASPECT_BY_PREFIX.get(option_id[:1])
        if aspect and aspect not in found:
            found.append(aspect)
    return tuple(found)


def _parse_public_state(feedback: str) -> tuple[str | None, set[str], set[str], set[str]]:
    current_match = _CURRENT_RE.search(feedback)
    current = current_match.group(1).casefold() if current_match else None
    if current == "none":
        current = None
    def parse_set(pattern: re.Pattern[str]) -> set[str]:
        match = pattern.search(feedback)
        if not match:
            return set()
        text = match.group(1).strip()
        if not text or text.casefold() == "none":
            return set()
        return {value.strip().casefold() for value in text.split(",") if value.strip()}
    searched = parse_set(_SEARCHED_RE)
    answered = parse_set(_ANSWERED_RE)
    visible_match = _VISIBLE_RE.search(feedback)
    visible = {value.upper() for value in option_ids(visible_match.group(1) if visible_match else "")}
    return current, searched, answered, visible


def _rejection_kind(feedback: str) -> str | None:
    lowered = feedback.casefold()
    # These neutral fallbacks are emitted by TravelGym/evaluator adapters when
    # the backend or User Simulator failed.  They are not public protocol
    # refusals and therefore make the remainder of a cached trajectory
    # untrustworthy; classify them as a fatal suffix boundary.
    environment_failure_markers = (
        "environment operation failed",
        "environment could not complete",
        "environment/api",
        "internal error",
        "backend is experiencing some issues",
        "responding system met some issues",
        "not sure how to respond to your latest utterance right now",
        "evaluation_timeout",
        "evaluation_error",
    )
    if any(marker in lowered for marker in environment_failure_markers):
        return "environment-failure"
    if "action-before-search" in lowered:
        return "action-before-search"
    if "answer-before-search" in lowered:
        return "answer-before-search"
    if "cross-aspect" in lowered or "one tool call may target only one aspect" in lowered:
        return "cross-aspect"
    if "repeated tool call" in lowered or "already searched" in lowered or "duplicate search" in lowered:
        return "repeated-search"
    if "invisible" in lowered or "not visible" in lowered or "visible option" in lowered and "invalid" in lowered:
        return "invisible-id"
    if (
        "already answered" in lowered
        or "duplicate answer" in lowered
        or "already recommended" in lowered
        or "same initial" in lowered and "option" in lowered
        or "same aspect" in lowered and "answer" in lowered
    ):
        return "duplicate-answer"
    if "vague" in lowered or "too general" in lowered or "too broad" in lowered:
        return "vague-action"
    if "invalid" in lowered or "missing required" in lowered or "parameter" in lowered and "reject" in lowered:
        return "invalid-parameters"
    if "tool call rejected" in lowered or "rejected" in lowered:
        return "invalid-parameters"
    return None


def _is_rejected(feedback: str) -> bool:
    lowered = feedback.casefold()
    return "tool call rejected" in lowered or "operation failed" in lowered or _rejection_kind(feedback) is not None


def _infer_protocol_error(
    choice: str | None,
    content: str,
    aspect: str | None,
    current: str | None,
    searched: set[str],
    visible: set[str],
    answered: set[str],
) -> str | None:
    if choice is None:
        return None
    if current is not None and aspect is not None and aspect != current:
        return "cross-aspect"
    if choice == "search":
        if current is not None and current in searched:
            return "repeated-search"
    elif choice == "action":
        if current is None or current not in searched:
            return "action-before-search"
    elif choice == "answer":
        if current is None or current not in searched:
            return "answer-before-search"
        ids = option_ids(content)
        answer = ids[0] if len(ids) == 1 and content.strip().upper() == ids[0] else None
        if answer is None:
            return "invalid-parameters"
        if not visible or answer not in visible:
            return "invisible-id"
        if current in answered:
            return "duplicate-answer"
    return None


def _make_events(record: Mapping[str, Any]) -> tuple[list[ReplayEvent], dict[int, list[str]]]:
    messages = record.get("messages", [])
    canonical_errors: dict[int, list[str]] = {}
    for error in record.get("_canonical_errors", []) or []:
        if isinstance(error, Mapping):
            try:
                canonical_errors[int(error.get("index"))] = [str(error.get("code"))]
            except (TypeError, ValueError):
                continue
    # Canonical rows produced by external converters sometimes keep the audit
    # code only on the offending message.  Fold those annotations into the
    # same reducer input; they are private metadata and never reach Actor
    # context or the public observation.
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        raw_errors = message.get("_canonical_errors", [])
        if isinstance(raw_errors, str):
            raw_errors = [raw_errors]
        if isinstance(raw_errors, Sequence):
            for code in raw_errors:
                canonical_errors.setdefault(message_index, []).append(str(code))
    events: list[ReplayEvent] = []
    current: str | None = None
    searched: set[str] = set()
    answered: set[str] = set()
    visible: set[str] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            index += 1
            continue
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        tool_index = index + 1 if isinstance(next_message, Mapping) and next_message.get("role") == "tool" else None
        args = tool_call_arguments(message)
        feedback = _feedback(messages[tool_index]) if tool_index is not None else ""
        raw_choice = args.get("choice") if args else None
        raw_content = args.get("content") if args else None
        choice = raw_choice.casefold() if isinstance(raw_choice, str) else None
        content = raw_content if isinstance(raw_content, str) else str(raw_content or "")
        aspect = _aspect_from_text(content, choice)
        if aspect is None:
            aspect = current
        current_before = current
        searched_before = set(searched)
        answered_before = set(answered)
        visible_before = set(visible)
        public_current, public_searched, public_answered, public_visible = _parse_public_state(feedback)
        # A rejection observation describes the state in which the rejected
        # call was attempted.  If no earlier accepted observation established
        # it, use that public value as the event's current-aspect context.
        event_current_before = current_before if current_before is not None else public_current
        after_current = public_current if (public_current is not None or "Current aspect:" in feedback) else current
        after_searched = public_searched if (public_searched or "Searched aspects:" in feedback) else set(searched)
        after_answered = public_answered if (public_answered or "Answered aspects:" in feedback) else set(answered)
        after_visible = public_visible if (public_visible or "Visible option IDs" in feedback) else set(visible)
        # A parseable call with malformed public parameters is recoverable
        # only when the environment supplied an explicit rejection followed
        # by a valid retry.  Keep this semantic error separate from malformed
        # JSON, which was already recorded in ``_canonical_errors`` and is
        # handled by the fatal suffix rule.
        argument_error = None
        if not isinstance(raw_choice, str) or raw_choice.casefold() not in {"search", "action", "answer"}:
            argument_error = "invalid-parameters"
        elif not isinstance(raw_content, str) or not raw_content.strip():
            argument_error = "invalid-parameters"
        elif choice in {"search", "action", "answer"} and len(_mentioned_aspects(content)) > 1:
            # A multi-aspect call violates the public state machine even when
            # an old transcript lacks the explicit rejection sentence.
            argument_error = "cross-aspect"
        kind = _rejection_kind(feedback) if feedback else None
        if kind is None:
            kind = argument_error
        if kind is None:
            kind = _infer_protocol_error(choice, content, aspect, event_current_before, searched_before, visible_before, answered_before)
        accepted = bool(args and tool_index is not None and not _is_rejected(feedback) and kind is None)
        answer_id = None
        if choice == "answer":
            ids = option_ids(content)
            if len(ids) == 1 and content.strip().upper() == ids[0]:
                answer_id = ids[0]
        # A Search is accepted only when its public observation contains the
        # complete result set.  Merely seeing an ID-looking token in an error
        # or truncated response must not establish visibility.  This is a
        # reducer validity check, never candidate filtering: all IDs in a
        # successful Search observation remain public.
        if accepted and choice == "search":
            search_ids = set(option_ids(feedback))
            if not search_ids:
                accepted = False
                kind = "environment-failure"
        event = ReplayEvent(
            assistant_index=index,
            tool_index=tool_index,
            choice=choice,
            content=content,
            aspect=aspect,
            feedback=feedback,
            rejected=not accepted,
            error_kind=kind,
            current_before=event_current_before,
            current_after=after_current,
            visible_after=set(after_visible),
            searched_after=set(after_searched),
            answered_after=set(after_answered),
            answer_id=answer_id,
            accepted=accepted,
        )
        if accepted:
            # A successful Search establishes visibility from public feedback;
            # Action does not alter it.  Accepted Answer closes the aspect and
            # the environment's feedback updates current/answered state above.
            if choice == "search" and aspect:
                after_current = aspect
                after_searched.add(aspect)
                # Legacy observations often predate the explicit
                # ``Visible option IDs`` line.  The Search response itself is
                # public and contains the complete option records; extracting
                # IDs here enforces answer visibility without filtering or
                # shrinking the candidate list.
                parsed_search_ids = set(option_ids(feedback))
                if parsed_search_ids:
                    after_visible = parsed_search_ids
            if choice == "answer" and aspect and answer_id:
                after_answered.add(aspect)
                after_current = None
        event.current_after = after_current
        event.searched_after = set(after_searched)
        event.answered_after = set(after_answered)
        event.visible_after = set(after_visible)
        current, searched, answered, visible = after_current, after_searched, after_answered, after_visible
        events.append(event)
        index = tool_index + 1 if tool_index is not None else index + 1
    return events, canonical_errors


def _future_repair(event: ReplayEvent, events: Sequence[ReplayEvent]) -> ReplayEvent | None:
    """Find a deterministic public repair before aspect switch/end."""
    kind = event.error_kind
    if kind not in RECOVERABLE_KINDS:
        return None
    # Older 244 traces do not always include the reducer's public state line.
    # Use the aspect inferred from the offending public call as a conservative
    # fallback, but never treat an unknown aspect as a wildcard: doing so could
    # incorrectly pair a repair from a different aspect.
    reference_aspect = event.current_before or event.aspect
    prior_accepted_aspects = {
        candidate.aspect
        for candidate in events
        if candidate.assistant_index < event.assistant_index and candidate.accepted and candidate.aspect
    }
    for candidate in events:
        if candidate.assistant_index <= event.assistant_index or not candidate.accepted:
            continue
        same = bool(reference_aspect and candidate.aspect in {reference_aspect, None})
        bootstrap = (
            not reference_aspect
            and not prior_accepted_aspects
            and candidate.choice == "search"
        )
        if kind == "action-before-search" and candidate.choice == "search" and (same or bootstrap):
            return candidate
        if kind == "answer-before-search":
            if candidate.choice == "search" and (same or bootstrap):
                return candidate
            if candidate.choice == "answer" and reference_aspect and candidate.aspect == reference_aspect:
                return candidate
        if kind == "cross-aspect" and reference_aspect and candidate.aspect == reference_aspect:
            return candidate
        if kind == "repeated-search" and candidate.choice in {"action", "answer"} and same:
            return candidate
        if kind == "invalid-parameters":
            # For a malformed ``choice`` there is no same operation to retry;
            # the first valid public call that resumes the current aspect is
            # the deterministic repair.  For malformed content/shape keep
            # the stricter same-choice requirement.
            choice_repaired = candidate.choice == event.choice or event.choice not in {"search", "action", "answer"}
            if choice_repaired and same:
                return candidate
        if kind == "invisible-id" and candidate.choice == "answer" and reference_aspect and candidate.aspect == reference_aspect:
            return candidate
        if kind == "duplicate-answer" and candidate.choice == "search" and reference_aspect and candidate.aspect != reference_aspect:
            return candidate
        if kind == "vague-action" and candidate.choice == "action" and len(candidate.content.strip()) >= 12 and same:
            return candidate
    return None


def _has_later_aspect(event: ReplayEvent, events: Sequence[ReplayEvent]) -> bool:
    for candidate in events:
        if candidate.assistant_index <= event.assistant_index or not candidate.accepted:
            continue
        if candidate.aspect and candidate.aspect != event.aspect:
            return True
    return False


def _answer_correct(event: ReplayEvent, spec: TaskSpec | None) -> bool | None:
    if not event.answer_id or not event.aspect or spec is None:
        return None
    if event.aspect not in spec.correct_ids:
        return None
    return event.answer_id in spec.correct_ids[event.aspect]


def recompute_metrics(messages: Sequence[Mapping[str, Any]], spec: TaskSpec | None) -> dict[str, Any]:
    """Recompute terminal correctness from the cleaned public transcript."""
    answers: dict[str, str] = {}
    for event in _make_events({"messages": list(messages)})[0]:
        if event.accepted and event.choice == "answer" and event.aspect and event.answer_id:
            answers[event.aspect] = event.answer_id
    if spec is None or not spec.has_labels:
        return {
            "correct_completion": 0.0,
            "completion_success": 0,
            "answered_aspects": sorted(answers),
            "correct_by_aspect": {},
            "metrics_available": False,
        }
    correct_by_aspect = {aspect: bool(answers.get(aspect) in spec.correct_ids.get(aspect, set())) for aspect in spec.aspects}
    correct_count = sum(correct_by_aspect.values())
    total = len(spec.aspects)
    completion = correct_count / total if total else 0.0
    return {
        "correct_completion": completion,
        "completion_success": int(total > 0 and len(answers) >= total and all(correct_by_aspect.values())),
        "answered_aspects": sorted(answers),
        "correct_by_aspect": correct_by_aspect,
        "answers": answers,
        "metrics_available": True,
    }


def classify_trajectory(
    metrics: Mapping[str, Any],
    *,
    retained_recoverable_error: bool = False,
    infrastructure_invalid: bool = False,
) -> tuple[str, float]:
    """Apply the specified post-clean classification priority."""
    if infrastructure_invalid:
        return "infrastructure_invalid", 0.0
    success = bool(metrics.get("completion_success"))
    correct = float(metrics.get("correct_completion", 0.0) or 0.0)
    if success and retained_recoverable_error:
        return "recoverable_correct", 1.0
    if success:
        # A fatal suffix removed before a successful endpoint does not change
        # this result: it is strict_gold after cleaning.
        return "strict_gold", 1.0
    if correct > 0:
        return "partial_correct", 0.5
    return "totally_wrong", 0.0


def clean_trajectory(
    record: Mapping[str, Any],
    *,
    task: Mapping[str, Any] | TaskSpec | None = None,
    task_id: str | None = None,
    reward_report: Mapping[str, Any] | None = None,
    source: str = "unknown",
    max_length: int = 32768,
    require_think: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize, replay and return ``(record, private_audit)``."""
    # Re-run canonicalisation even for already-canonical rows.  This hoists
    # converter audit annotations out of messages and re-applies the public
    # argument scrubber, ensuring private fields can never reach the Qwen
    # template on a second cleaning pass.
    canonical = canonicalize_record(record, source=source)
    if "assistant_train_mask" not in canonical:
        canonical["assistant_train_mask"] = [0] * len(canonical.get("messages", []))
    spec = task if isinstance(task, TaskSpec) else TaskSpec.from_mapping(task, task_id=task_id)
    audit: dict[str, Any] = {
        "schema_version": canonical.get("schema_version"),
        "canonical_hash_before": canonical_hash(canonical),
        "source": source,
        "task_id": task_id or (spec.task_id if spec else None),
        "fatal_truncation": False,
        "fatal_kind": None,
        "retained_recoverable_errors": [],
        "events": [],
    }
    report = reward_report or {}
    infrastructure_invalid = bool(
        isinstance(report, Mapping)
        and report.get("reward_valid_for_training", report.get("reward_valid", True)) is False
    )
    if infrastructure_invalid:
        audit["infrastructure_invalid"] = True
        canonical["assistant_train_mask"] = [0] * len(canonical.get("messages", []))
        metrics = recompute_metrics(canonical.get("messages", []), spec)
        category, weight = classify_trajectory(metrics, infrastructure_invalid=True)
        canonical["trainer_metadata"] = {
            **dict(canonical.get("trainer_metadata", {})),
            "trajectory_class": category,
            "sample_weight": weight,
        }
        audit["metrics"] = metrics
        audit["trajectory_class"] = category
        audit["sample_weight"] = weight
        audit["canonical_hash_after"] = canonical_hash(canonical)
        return canonical, audit

    events, canonical_errors = _make_events(canonical)
    missing_think_error = False
    if require_think:
        # Thinking is a required Teacher/SFT target, but the canonical schema
        # itself remains model-independent.  Mark missing reasoning as an
        # unrecoverable Assistant-turn error only for callers that opt into
        # the strict collection policy (the merge CLI does so by default).
        for index, message in enumerate(canonical.get("messages", [])):
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if message.get("tool_calls") and not str(message.get("reasoning_content") or "").strip():
                canonical_errors.setdefault(index, []).append("assistant_think_missing")
                missing_think_error = True
    audit["events"] = [
        {
            "assistant_index": event.assistant_index,
            "tool_index": event.tool_index,
            "choice": event.choice,
            "aspect": event.aspect,
            "accepted": event.accepted,
            "error_kind": event.error_kind,
            "answer_id": event.answer_id,
        }
        for event in events
    ]
    mask = [0] * len(canonical.get("messages", []))
    keep_indices: set[int] = set(range(len(canonical.get("messages", []))))
    fatal_index: int | None = None
    fatal_kind: str | None = None
    retained_recoverable = False

    # Canonical parse errors are unrecoverable at the Assistant turn and must
    # remove the complete suffix, including its observation.
    if canonical_errors:
        first = min(canonical_errors)
        if first < len(canonical.get("messages", [])) and canonical["messages"][first].get("role") == "tool":
            # An observation mismatch belongs to the Assistant/tool exchange;
            # truncate from the Assistant rather than leaving a dangling call.
            while first > 0 and canonical["messages"][first].get("role") != "assistant":
                first -= 1
        fatal_index = first
        source_error_index = min(canonical_errors)
        fatal_kind = canonical_errors[source_error_index][0]
    for event in events:
        if fatal_index is not None and event.assistant_index >= fatal_index:
            break
        if event.tool_index is None:
            fatal_index = event.assistant_index
            fatal_kind = "tool_observation_misaligned"
            break
        if event.error_kind in FATAL_KINDS:
            fatal_index = event.assistant_index
            fatal_kind = event.error_kind
            break
        if event.rejected:
            # A recoverable error is trainable context only when the
            # transcript contains the environment's explicit public refusal.
            # Inferring a protocol violation from turn order alone is useful
            # for diagnostics, but must not preserve a synthetic error that
            # could have hidden an untrusted observation.
            explicit_rejection = _is_rejected(event.feedback)
            repair = _future_repair(event, events) if explicit_rejection else None
            if event.error_kind in RECOVERABLE_KINDS and repair is not None:
                retained_recoverable = True
                event.repair_for = event.error_kind
                audit["retained_recoverable_errors"].append(
                    {"assistant_index": event.assistant_index, "kind": event.error_kind, "repair_index": repair.assistant_index}
                )
                continue
            fatal_index = event.assistant_index
            fatal_kind = (
                event.error_kind
                if explicit_rejection
                else "unrepaired-recoverable-error"
            ) or "unrepaired-recoverable-error"
            break
        if event.accepted:
            correctness = _answer_correct(event, spec)
            if event.choice == "answer" and correctness is False:
                if not _has_later_aspect(event, events):
                    fatal_index = event.assistant_index
                    fatal_kind = "wrong-terminal-answer"
                    break
                # Accepted but semantically wrong non-final Answer remains
                # context and is explicitly not supervised.
                continue
            mask[event.assistant_index] = 1

    if fatal_index is not None:
        keep_indices = {index for index in keep_indices if index < fatal_index}
        audit["fatal_truncation"] = True
        audit["fatal_kind"] = fatal_kind
    cleaned_messages = [message for index, message in enumerate(canonical.get("messages", [])) if index in keep_indices]
    cleaned_mask = [mask[index] if index < len(mask) and index in keep_indices else 0 for index in range(len(canonical.get("messages", []))) if index in keep_indices]
    canonical["messages"] = cleaned_messages
    canonical["assistant_train_mask"] = cleaned_mask
    # Ensure no stale masks from a malformed source survive.
    if len(cleaned_mask) != len(cleaned_messages):
        canonical["assistant_train_mask"] = [0] * len(cleaned_messages)
    metrics = recompute_metrics(cleaned_messages, spec)
    category, weight = classify_trajectory(
        metrics,
        retained_recoverable_error=retained_recoverable,
        infrastructure_invalid=False,
    )
    if not metrics.get("metrics_available", False):
        category, weight = "infrastructure_invalid", 0.0
        audit["metrics_unavailable"] = True
    if missing_think_error:
        category, weight = "infrastructure_invalid", 0.0
        audit["infrastructure_invalid"] = True
        audit["missing_think"] = True
    if not any(canonical["assistant_train_mask"]):
        # All-zero examples are useful for audit but not SFT targets.
        audit["no_supervised_tokens"] = True
    canonical["trainer_metadata"] = {
        **dict(canonical.get("trainer_metadata", {})),
        "trajectory_class": category,
        "sample_weight": weight,
        "enable_thinking": True,
    }
    audit["metrics"] = metrics
    audit["trajectory_class"] = category
    audit["sample_weight"] = weight
    audit["canonical_hash_after"] = canonical_hash(canonical)
    audit["assistant_train_mask"] = list(canonical["assistant_train_mask"])
    try:
        validate_canonical(canonical)
    except CanonicalError as exc:
        # A malformed input is kept as an auditable quarantine record rather
        # than silently dropped.
        audit["infrastructure_invalid"] = True
        audit["validation_error"] = str(exc)
        canonical["trainer_metadata"]["trajectory_class"] = "infrastructure_invalid"
        canonical["trainer_metadata"]["sample_weight"] = 0.0
    return canonical, audit


def is_sft_eligible(record: Mapping[str, Any], audit: Mapping[str, Any] | None = None) -> bool:
    metadata = record.get("trainer_metadata", {}) if isinstance(record, Mapping) else {}
    category = metadata.get("trajectory_class")
    if category not in {"strict_gold", "recoverable_correct", "partial_correct"}:
        return False
    if audit and audit.get("metrics_unavailable"):
        return False
    return bool(any(int(value) for value in record.get("assistant_train_mask", []) or []))


__all__ = [
    "TaskSpec",
    "clean_trajectory",
    "classify_trajectory",
    "is_sft_eligible",
    "recompute_metrics",
]
