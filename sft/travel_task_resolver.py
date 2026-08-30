"""Deterministic offline mapping from TravelGym transcripts to task labels.

The resolver is used only while building the private audit sidecar.  It never
adds the task, preference IDs, correct IDs, or option attributes to canonical
messages sent to the Actor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from .clean_travel_trajectories import ASPECT_BY_PREFIX, TaskSpec
    from .travel_canonical import canonical_hash, canonicalize_record, tool_call_arguments
except ImportError:
    from clean_travel_trajectories import ASPECT_BY_PREFIX, TaskSpec
    from travel_canonical import canonical_hash, canonicalize_record, tool_call_arguments


_WS = re.compile(r"\s+")


def _norm(text: Any) -> str:
    value = str(text or "").casefold()
    value = _WS.sub(" ", value).strip()
    return value


class TravelTaskResolver:
    """Load task JSON and resolve by exact key or unique initial description."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        task_paths: list[str | Path] | None = None,
        explicit_map: Mapping[str, str] | None = None,
    ):
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        self.project_root = root.resolve()
        if task_paths is None:
            task_paths = sorted((root / "gyms" / "TravelGym" / "travelgym" / "data").glob("travelgym_data_*.json"))
        self.tasks: dict[str, dict[str, Any]] = {}
        for path_value in task_paths:
            path = Path(path_value)
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, Mapping):
                continue
            for key, task in payload.items():
                if isinstance(task, Mapping):
                    item = dict(task)
                    item.setdefault("id", str(key))
                    item.setdefault("env_name", path.stem.replace("travelgym_data_", "travel"))
                    # Do not overwrite a duplicate key from another file; a
                    # task ID collision is not a safe basis for label recovery.
                    self.tasks.setdefault(str(key), item)
        self._by_initial: dict[str, list[str]] = {}
        for key, task in self.tasks.items():
            initial = task.get("initial_description") or task.get("scenario")
            normalized = _norm(initial)
            if normalized:
                self._by_initial.setdefault(normalized, []).append(key)
        self.explicit_map = {str(key): str(value) for key, value in (explicit_map or {}).items()}

    @staticmethod
    def load_alignment_map(path: str | Path | None) -> dict[str, str]:
        """Load an optional reviewed ``record-key -> task-id`` sidecar."""
        if path is None:
            return {}
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and isinstance(payload.get("records"), Mapping):
            payload = payload["records"]
        if not isinstance(payload, Mapping):
            raise ValueError("task alignment map must be a JSON object")
        return {str(key): str(value) for key, value in payload.items()}

    def resolve(self, record: Mapping[str, Any], private_meta: Mapping[str, Any] | None = None) -> tuple[str | None, TaskSpec | None]:
        meta = private_meta or {}
        def mapped_task(value: str) -> tuple[str | None, TaskSpec | None]:
            mapped_value = str(value)
            expected_env = None
            if "::" in mapped_value:
                expected_env, mapped_value = mapped_value.split("::", 1)
            if mapped_value in self.tasks:
                actual_env = str(self.tasks[mapped_value].get("env_name") or "")
                if expected_env and actual_env and expected_env != actual_env:
                    return None, None
                return mapped_value, TaskSpec.from_mapping(self.tasks[mapped_value], task_id=mapped_value)
            return None, None
        # An explicit sidecar map is the only permitted way to resolve a
        # transcript whose public text is genuinely ambiguous.  Keys may be a
        # source index (``"97"``) or ``<source_path>#<source_index>``.  The
        # mapped task must exist; otherwise we quarantine rather than guessing.
        source_index = meta.get("source_index")
        source_path = meta.get("source_path")
        # Candidate reports use repository-relative paths while merge helpers
        # often pass an absolute source path.  Accept both spellings (and the
        # normalized slash form) for the *explicit* reviewed map only.  This
        # does not infer an identity; it merely makes a human-approved map
        # portable across CLI working directories.
        path_candidates: list[str] = []
        if source_path is not None:
            raw_path = str(source_path)
            path_candidates.append(raw_path)
            path_candidates.append(raw_path.replace("\\", "/"))
            try:
                root_path = self.project_root
                resolved_path = Path(raw_path).resolve()
                path_candidates.append(str(resolved_path.relative_to(root_path)).replace("\\", "/"))
            except (OSError, ValueError):
                pass
        map_keys: list[str] = []
        if source_index is not None:
            for path_value in path_candidates:
                map_keys.append(f"{path_value}#{source_index}")
            map_keys.append(str(source_index))
        for map_key in map_keys:
            if map_key is None or map_key not in self.explicit_map:
                continue
            return mapped_task(self.explicit_map[map_key])
        # Candidate reports expose a stable opaque hash so a reviewed map does
        # not need to depend on a mutable source path.  This remains an
        # explicit human mapping; the resolver never selects a candidate.
        try:
            # Keep this derivation byte-identical to the candidate-audit and
            # quarantine paths.  A reviewed map copied from the private audit
            # must resolve the same opaque row on a later merge invocation.
            opaque_key = "opaque_sft::" + canonical_hash(canonicalize_record(record))
        except Exception:
            opaque_key = None
        if opaque_key and opaque_key in self.explicit_map:
            return mapped_task(self.explicit_map[opaque_key])
        candidates: list[str] = []
        explicit = record.get("task_id") or record.get("id") or meta.get("gold") or meta.get("task_key")
        if explicit is not None and str(explicit) in self.tasks:
            candidates = [str(explicit)]
        else:
            messages = record.get("messages", record.get("conversations", []))
            user_text = " ".join(
                str(message.get("content", message.get("value", "")))
                for message in messages
                if isinstance(message, Mapping) and str(message.get("role", message.get("from", ""))).casefold() in {"user", "human"}
            )
            normalized_user = _norm(user_text)
            for initial, keys in self._by_initial.items():
                if initial and initial in normalized_user:
                    candidates.extend(keys)
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return None, None
        if len(candidates) > 1:
            # Several generated tasks intentionally share an initial request
            # while differing in hidden preferences.  Resolve only when the
            # public answer IDs identify one task unambiguously; never choose
            # by insertion order and never use this for record deduplication.
            answers: list[tuple[str, str]] = []
            # Parse both legacy ShareGPT and OpenAI/DeepSeek records through
            # the canonical parser.  A regex over the legacy JSON is brittle
            # (the outer ``name``/``arguments`` keys may appear in either
            # order) and was leaving uniquely identifying public answers
            # unresolved.
            try:
                canonical = canonicalize_record(record)
            except Exception:
                canonical = {"messages": []}
            for message in canonical.get("messages", []):
                if not isinstance(message, Mapping) or message.get("role") != "assistant":
                    continue
                args = tool_call_arguments(message)
                if not isinstance(args, Mapping) or str(args.get("choice", "")).casefold() != "answer":
                    continue
                raw_id = str(args.get("content", "")).strip().upper()
                if raw_id:
                    answers.append((ASPECT_BY_PREFIX.get(raw_id[:1], ""), raw_id))
            scored: list[tuple[int, str]] = []
            for candidate in candidates:
                candidate_spec = TaskSpec.from_mapping(self.tasks[candidate], task_id=candidate)
                score = 0
                for aspect, answer_id in answers:
                    if answer_id in candidate_spec.correct_ids.get(aspect, set()):
                        score += 3
                    elif answer_id in candidate_spec.all_ids.get(aspect, set()):
                        score += 1
                # Do not score an ID merely because it appears in a Search
                # response: all candidates intentionally share the public
                # option universe.  It is retained above for audit/debugging,
                # but hidden correctness is required for deterministic label
                # recovery.
                scored.append((score, candidate))
            scored.sort(reverse=True)
            if not scored or (len(scored) > 1 and scored[0][0] == scored[1][0]):
                return None, None
            candidates = [scored[0][1]]
        task_id = candidates[0]
        return task_id, TaskSpec.from_mapping(self.tasks[task_id], task_id=task_id)


__all__ = ["TravelTaskResolver"]
