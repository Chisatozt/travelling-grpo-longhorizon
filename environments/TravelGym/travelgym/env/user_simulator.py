"""DeepSeek-backed TravelGym user simulator.

The prompts and four-way response policy mirror the upstream UserRL
TravelGym simulator used while collecting the project's Teacher trajectories.
Requests are cached by their complete private prompt hash; only aggregate
usage and sanitized request metadata are written to the append-only event log.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - validated by the runtime preflight
    AsyncOpenAI = None


PROMPT_VERSION = "userrl-travelgym-user-v1"
VAGUE_RESPONSE = (
    "Your question is too vague and general, and I am not sure how to respond to it. "
    "Please ask me about some specific aspects of my preferences, in a more detailed "
    "and concrete way, so that I can provide you with a more accurate response."
)
UNAVAILABLE_RESPONSE = (
    "This is a good question. However, I do not have specific preference in the aspect "
    "you ask about yet (or maybe I have already elicited that to you before). You may "
    "continue to ask me about other detailed and specific preferences."
)

JUDGE_SYSTEM = """## **Task**
You are an expert judge evaluating the type of an agent's conversation utterance in a travel planning scenario to determine the appropriate response strategy.

## **Instruction**
1. Analyze the agent's latest utterance in the context of the conversation
2. Determine if the agent is explicitly asking for preferences that you have, asking for preferences that you don't have, giving a too general query, or just making general conversations
3. If asking for preferences that you have, identify which specific preference from the available list matches
4. Classify the utterance type and provide the assessment in JSON format

## **Example Format**
```json
{
    "type": "1/2/3/4",
    "preference_id": "preference id if type is 2"
}
```

## **Important Notes**
- Type "1": Normal conversation, not preference-related
- Type "2": Agent explicitly and concretely asking for a preference that exists in the available preferences list. The way how agent asks must be concrete in order to be classified as Type "2".
- Type "3": Agent explicitly and concretely asking for preferences, but the specific preference is not available. Similarly, the way how agent asks must also be concrete and specific.
- Type "4": Agent making a very vague and general query about preference instead of focusing on a specific aspect (e.g. "Do you have any preferences for the car? (vague and general, type 4)" instead of "what exact model of the car do you like? (concrete and specific, type 2)")
- For Type "2", you must provide the exact one preference_id from the available preferences. If there's multiple preferences that match, choose the one that is most relevant to the conversation context.
- Be precise in identifying preference requests vs general conversation"""

JUDGE_USER = """**Travel Scenario:**
{scenario}

**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

**Available Preferences:**
{preferences_list}

Please analyze the agent's latest utterance and classify its type, then provide your assessment in JSON format wrapped in ```json and ```."""

RESPONSE_PREFERENCE_SYSTEM = """## **Task**
You are a helpful user in a travel planning conversation who needs to respond to an agent's explicit request for your preference, which you should elicit in an implicit and indirect manner.

## **Instruction**
1. The agent has explicitly asked about a specific preference that you have
2. Respond in a natural, conversational way that reveals your preference implicitly and indirectly
3. Use the provided implicit elicitation statement as guidance, but make it sound natural in context
4. Keep the conversation flowing while sharing your preference information
5. Provide your response in the specified JSON format

## **Example Format**
```json
{
    "thought": "Your thought process of how to respond naturally and implicitly reveal the preference under the guidance of the implicit elicitation statement",
    "response": "Your natural conversational response that implicitly reveals the preference"
}
```

## **Important Notes**
- Respond naturally as if you're a real person sharing preferences
- Don't directly state "My preference is..." - be more subtle, conversational, and indirect
- Use the implicit elicitation statement as inspiration but adapt it to the conversation context
- Keep responses appropriate length for natural conversation
- Maintain consistency with the conversation history"""

RESPONSE_PREFERENCE_USER = """**Your Preference:**
{preference}

**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

Please respond naturally to the agent's request while implicitly sharing your preference under the guidance of the implicit elicitation statement. Provide your response in JSON format wrapped in ```json and ```."""

RESPONSE_ELICIT_SYSTEM = """## **Task**
You are a helpful user in a travel planning conversation who needs to proactively, naturally, but indirectly introduce a preference into the conversation.

## **Instruction**
1. The conversation has gone several turns without preference discussion
2. Naturally steer the conversation to reveal one of your preferences
3. Use the provided implicit elicitation statement as guidance for how to reveal the preference
4. Make the preference revelation feel organic and contextually appropriate, but still in an implicit and indirect manner
5. Provide your response in the specified JSON format

## **Example Format**
```json
{
    "thought": "Your thought process of how to naturally and implicitly introduce the preference under the guidance of the implicit elicitation statement",
    "response": "Your natural conversational response that proactively introduces the preference"
}
```

## **Important Notes**
- Connect to the current conversation context when possible
- Make the preference introduction feel natural and not forced, but still in an implicit and indirect manner
- Use the implicit elicitation statement as inspiration but adapt to the conversation flow
- Don't abruptly change topics - find natural transitions and keep responses conversational and engaging
- If the implicit elicitation statement cannot clearly what high-level aspect (flight, restaurant, etc.) the preference is about, you should be clear about the high-level aspect in your elicitation to avoid confusion, but still elicit the concrete preference in an implicit way"""

RESPONSE_ELICIT_USER = """**Preference to Elicit:**
{preference}

**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

Please respond naturally while proactively introducing your preference into the conversation in an implicit and indirect manner under the guidance of the implicit elicitation statement. Provide your response in JSON format wrapped in ```json and ```."""

RESPONSE_NATURAL_SYSTEM = """## **Task**
You are a helpful user in a travel planning conversation who needs to respond naturally to the agent's utterance.

## **Instruction**
1. The agent's utterance is not related to any specific preferences you have
2. Respond naturally and in a succinct manner, like you are giving a half-hearted reply
3. Be neutral and do not reveal any new or arbitrary personal preferences
4. Provide your response in the specified JSON format

## **Example Format**
```json
{
    "thought": "Your thought process of how to respond naturally and keep the conversation flowing while being neutral",
    "response": "Your natural conversational response"
}
```

## **Important Notes**
- Keep responses natural, conversational and succinct
- Stay on topic with travel planning when appropriate, but do not actively ask any questions
- Don't introduce any personal preferences. If being asked, you should be neutral (e.g. "I don't have a preference on that", "Everything is fine") and do not arbitrarily reveal any new preferences."""

RESPONSE_NATURAL_USER = """**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

Please respond naturally to continue the conversation and keep neutral without revealing any new or arbitrary personal preferences. Provide your response in JSON format wrapped in ```json and ```."""


class UserSimulatorError(RuntimeError):
    """A fail-closed User Simulator request or response error."""


@dataclass(frozen=True)
class UserSimulatorTurn:
    response: str
    elicited_preference_ids: tuple[str, ...]
    judgment_type: str
    note: str
    nonpreference_times: int
    proactive_preference_ids: tuple[str, ...] = ()


def new_user_telemetry() -> dict[str, int | float]:
    return {
        "user_api_calls": 0,
        "user_api_errors": 0,
        "user_retries": 0,
        "user_cache_hits": 0,
        "user_judge_api_calls": 0,
        "user_response_api_calls": 0,
        "user_prompt_tokens": 0,
        "user_completion_tokens": 0,
        "user_total_tokens": 0,
        "user_reasoning_tokens": 0,
        "user_wall_time_seconds": 0.0,
    }


def build_history(history: Sequence[Mapping[str, Any]], with_note: bool = True) -> str:
    if not history:
        return "No previous conversation."
    parts: list[str] = []
    for entry in history:
        role = str(entry.get("role", "unknown"))
        content = str(entry.get("content", ""))
        if role == "agent":
            parts.append(f"Agent: {content}")
            note = str(entry.get("note", "")) if with_note else ""
            if note:
                parts.append(f"[Note: {note}]")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "database":
            parts.append(f"Database: {content}")
        else:
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("")
    return "\n".join(parts).strip()


def parse_json_response(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [match.group(1)] if match else []
    brace = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def initialize_preferences(task: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    counter = 1
    for aspect in task.get("dimensions", []):
        details = task.get(str(aspect), {})
        for raw in details.get("preferences", []) if isinstance(details, Mapping) else []:
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                result.append({
                    "id": f"P{counter}",
                    "aspect": str(raw[0]),
                    "subcategory": str(raw[1]),
                    "preference": str(raw[2]),
                    "implicit_elicitation": str(raw[3]),
                })
                counter += 1
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _usage_value(container: Any, key: str, default: Any = 0) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default) or default
    return getattr(container, key, default) or default


def _bump(telemetry: dict[str, Any] | None, key: str, value: int | float = 1) -> None:
    if isinstance(telemetry, dict):
        telemetry[key] = telemetry.get(key, 0) + value


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _SqliteResponseCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                request_kind TEXT NOT NULL,
                response_json TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL
            )"""
        )
        self._connection.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        value = json.loads(row[0])
        return value if isinstance(value, dict) else None

    def put(self, key: str, model: str, request_kind: str, response: dict[str, Any], usage: dict[str, int]) -> None:
        with self._lock:
            self._connection.execute(
                """INSERT OR IGNORE INTO responses
                   (cache_key, created_at, model, request_kind, response_json,
                    prompt_tokens, completion_tokens, total_tokens, reasoning_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    key, _utc_now(), model, request_kind,
                    json.dumps(response, ensure_ascii=False, sort_keys=True),
                    usage["prompt_tokens"], usage["completion_tokens"],
                    usage["total_tokens"], usage["reasoning_tokens"],
                ),
            )
            self._connection.commit()


_CACHE_LOCK = threading.Lock()
_CACHES: dict[str, _SqliteResponseCache] = {}
_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[tuple[str, str], Any] = {}
_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _cache(path: Path) -> _SqliteResponseCache:
    key = str(path.resolve())
    with _CACHE_LOCK:
        value = _CACHES.get(key)
        if value is None:
            value = _SqliteResponseCache(Path(key))
            _CACHES[key] = value
        return value


def _client(api_key: str, base_url: str) -> Any:
    if AsyncOpenAI is None:
        raise UserSimulatorError("OpenAI-compatible SDK is not installed")
    key = (base_url, hashlib.sha256(api_key.encode("utf-8")).hexdigest())
    with _CLIENT_LOCK:
        value = _CLIENTS.get(key)
        if value is None:
            value = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=None, max_retries=0)
            _CLIENTS[key] = value
        return value


def _semaphore(limit: int) -> asyncio.Semaphore:
    loop_key = id(asyncio.get_running_loop())
    value = _SEMAPHORES.get(loop_key)
    if value is None:
        value = asyncio.Semaphore(max(1, limit))
        _SEMAPHORES[loop_key] = value
    return value


@asynccontextmanager
async def _request_slot(semaphore: Any):
    if semaphore is None:
        yield
    else:
        async with semaphore:
            yield


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def _transient_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status) in {408, 409, 429} or int(status) >= 500
        except (TypeError, ValueError):
            pass
    name = type(exc).__name__.casefold()
    return any(token in name for token in ("timeout", "connection", "ratelimit", "internalserver"))


async def model_call(
    system_prompt: str,
    user_prompt: str,
    *,
    request_kind: str,
    task_id: str,
    aspect: str,
    config: Any,
    telemetry: dict[str, Any] | None,
    model_client: Any = None,
    request_semaphore: Any = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    model = str(config.model_name)
    timeout = float(config.timeout)
    is_judge = request_kind == "judge"
    token_env = (
        "TRAVELGYM_USER_JUDGE_MAX_TOKENS"
        if is_judge
        else "TRAVELGYM_USER_RESPONSE_MAX_TOKENS"
    )
    default_max_tokens = 128 if is_judge else 2048
    max_tokens = int(os.environ.get(token_env, str(default_max_tokens)))
    if max_tokens <= 0:
        raise UserSimulatorError(f"{token_env} must be positive")
    thinking = {"type": "disabled" if is_judge else "enabled"}
    response_format = {"type": "json_object"} if is_judge else None
    cache_path = Path(
        os.environ.get(
            "TRAVELGYM_USER_CACHE_PATH",
            str(Path(__file__).resolve().parents[4] / "outputs" / "travelgym_user_simulator" / "responses.sqlite3"),
        )
    )
    event_path = Path(
        os.environ.get("TRAVELGYM_USER_API_LOG_PATH", str(cache_path.with_suffix(".events.jsonl")))
    )
    request = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "temperature": float(config.temperature),
        "max_tokens": max_tokens,
        "thinking": thinking,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request_hash = _stable_hash(request)
    cached = _cache(cache_path).get(request_hash)
    if cached is not None:
        _bump(telemetry, "user_cache_hits")
        _append_event(event_path, {
            "at": _utc_now(), "event": "cache_hit", "request_hash": request_hash,
            "request_kind": request_kind, "model": model, "task_id": task_id, "aspect": aspect,
        })
        return cached

    client = model_client
    if client is None:
        api_key = str(config.api_key or "")
        base_url = str(config.base_url or "")
        if not api_key or not base_url:
            raise UserSimulatorError("DeepSeek API credentials are not configured")
        client = _client(api_key, base_url)
    semaphore = request_semaphore or _semaphore(
        int(os.environ.get("TRAVELGYM_USER_REQUEST_CONCURRENCY", "8"))
    )
    attempts = max(1, int(max_attempts))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        _bump(telemetry, "user_api_calls")
        _bump(telemetry, f"user_{request_kind}_api_calls")
        try:
            completion_kwargs = {
                "model": model,
                "messages": request["messages"],
                "temperature": float(config.temperature),
                "max_tokens": max_tokens,
                "n": 1,
                "timeout": timeout,
                "extra_body": {"thinking": thinking},
            }
            if response_format is not None:
                completion_kwargs["response_format"] = response_format
            async with _request_slot(semaphore):
                response = await asyncio.wait_for(
                    client.chat.completions.create(**completion_kwargs),
                    timeout=timeout + 5.0,
                )
            elapsed = time.perf_counter() - started
            message = response.choices[0].message
            parsed = parse_json_response(getattr(message, "content", ""))
            if parsed is None:
                raise UserSimulatorError("DeepSeek User Simulator returned invalid JSON")
            usage_obj = getattr(response, "usage", None)
            details = _usage_value(usage_obj, "completion_tokens_details", None)
            usage = {
                "prompt_tokens": int(_usage_value(usage_obj, "prompt_tokens")),
                "completion_tokens": int(_usage_value(usage_obj, "completion_tokens")),
                "total_tokens": int(_usage_value(usage_obj, "total_tokens")),
                "reasoning_tokens": int(_usage_value(details, "reasoning_tokens")),
            }
            for name, value in usage.items():
                _bump(telemetry, f"user_{name}", value)
            _bump(telemetry, "user_wall_time_seconds", elapsed)
            _cache(cache_path).put(request_hash, model, request_kind, parsed, usage)
            _append_event(event_path, {
                "at": _utc_now(), "event": "success", "request_hash": request_hash,
                "request_kind": request_kind, "model": model, "task_id": task_id,
                "aspect": aspect, "attempt": attempt, "elapsed_seconds": elapsed, **usage,
            })
            return parsed
        except Exception as exc:
            last_error = exc
            elapsed = time.perf_counter() - started
            _bump(telemetry, "user_api_errors")
            _bump(telemetry, "user_wall_time_seconds", elapsed)
            transient = isinstance(exc, UserSimulatorError) or _transient_error(exc)
            retry = transient and attempt < attempts
            if retry:
                _bump(telemetry, "user_retries")
            _append_event(event_path, {
                "at": _utc_now(), "event": "error", "request_hash": request_hash,
                "request_kind": request_kind, "model": model, "task_id": task_id,
                "aspect": aspect, "attempt": attempt, "elapsed_seconds": elapsed,
                "error_type": type(exc).__name__, "transient": transient, "retry": retry,
            })
            if not retry:
                break
            await asyncio.sleep(min(8.0, 2.0 ** (attempt - 1)) + random.random() * 0.25)
    raise UserSimulatorError(
        f"DeepSeek User Simulator {request_kind} failed after {attempts} bounded attempt(s)"
    ) from last_error


async def evaluate_user_question(
    question: str,
    *,
    task_id: str,
    aspect: str,
    scenario: str,
    history: list[dict[str, str]],
    available_preferences: list[dict[str, str]],
    nonpreference_times: int,
    elicitation_interval: int,
    config: Any,
    telemetry: dict[str, Any] | None,
    rng: random.Random,
    model_client: Any = None,
    request_semaphore: Any = None,
    max_attempts: int = 3,
) -> UserSimulatorTurn:
    history_text = build_history(history, with_note=True)
    preferences_text = "\n\n".join(
        f"Preference ID: {item['id']}\tAspect: {item['aspect']}\nPreference: {item['preference']}"
        for item in available_preferences
    )
    judgment = await model_call(
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            scenario=scenario,
            conversation_history=history_text,
            latest_utterance=question,
            preferences_list=preferences_text,
        ),
        request_kind="judge",
        task_id=task_id,
        aspect=aspect,
        config=config,
        telemetry=telemetry,
        model_client=model_client,
        request_semaphore=request_semaphore,
        max_attempts=max_attempts,
    )
    judgment_type = str(judgment.get("type", "")).strip()
    if judgment_type not in {"1", "2", "3", "4"}:
        raise UserSimulatorError("DeepSeek User Simulator returned an invalid judgment type")

    if judgment_type == "2":
        preference_id = str(judgment.get("preference_id", "")).strip()
        current = next((item for item in available_preferences if item["id"] == preference_id), None)
        if current is None:
            raise UserSimulatorError("DeepSeek User Simulator selected an unavailable preference")
        note = (
            "Agent's utterance is explicitly asking for a preference that I have. "
            f"Preference ID: {preference_id}. I will elicit this preference in an implicit and indirect manner."
        )
        preference_text = (
            f"Preference ID: {preference_id}\tAspect: {current['aspect']}\n"
            f"Preference: {current['preference']}\n"
            f"Implicit Elicitation Statement: {current['implicit_elicitation']}"
        )
        generated = await model_call(
            RESPONSE_PREFERENCE_SYSTEM,
            RESPONSE_PREFERENCE_USER.format(
                preference=preference_text,
                conversation_history=history_text,
                latest_utterance=question,
            ),
            request_kind="response",
            task_id=task_id,
            aspect=aspect,
            config=config,
            telemetry=telemetry,
            model_client=model_client,
            request_semaphore=request_semaphore,
            max_attempts=max_attempts,
        )
        response = str(generated.get("response", "")).strip()
        if not response:
            raise UserSimulatorError("DeepSeek User Simulator returned an empty preference response")
        return UserSimulatorTurn(response, (preference_id,), judgment_type, note, 0)

    if nonpreference_times >= int(elicitation_interval) and available_preferences:
        current = rng.choice(available_preferences)
        preference_id = current["id"]
        note = (
            "The agent's latest utterance is not related to any preference I have, and the topic is off "
            "the target for several turns. I will respond naturally and coherently, but also proactively "
            "elicit a preference in an implicit and indirect manner."
        )
        preference_text = (
            f"Preference ID: {preference_id}\tAspect: {current['aspect']}\n"
            f"Preference: {current['preference']}\n"
            f"Implicit Elicitation Statement: {current['implicit_elicitation']}"
        )
        generated = await model_call(
            RESPONSE_ELICIT_SYSTEM,
            RESPONSE_ELICIT_USER.format(
                preference=preference_text,
                conversation_history=history_text,
                latest_utterance=question,
            ),
            request_kind="response",
            task_id=task_id,
            aspect=aspect,
            config=config,
            telemetry=telemetry,
            model_client=model_client,
            request_semaphore=request_semaphore,
            max_attempts=max_attempts,
        )
        response = str(generated.get("response", "")).strip()
        if not response:
            raise UserSimulatorError("DeepSeek User Simulator returned an empty proactive response")
        return UserSimulatorTurn(response, (), judgment_type, note, 0, (preference_id,))

    if judgment_type == "3":
        note = (
            "Agent's utterance is explicitly asking for a preference, but I do not have this preference. "
            "I will respond in a neutral way, but still relevant to the conversation and coherent."
        )
        return UserSimulatorTurn(UNAVAILABLE_RESPONSE, (), judgment_type, note, nonpreference_times + 1)
    if judgment_type == "4":
        note = (
            "Agent's utterance is very vague and general, not explicitly asking for a detailed aspect "
            "of preference. I will point out that it is too general and vague."
        )
        return UserSimulatorTurn(VAGUE_RESPONSE, (), judgment_type, note, nonpreference_times + 1)

    note = "Agent's utterance is not related to preference or not explicitly asking for a preference."
    generated = await model_call(
        RESPONSE_NATURAL_SYSTEM,
        RESPONSE_NATURAL_USER.format(
            conversation_history=history_text,
            latest_utterance=question,
        ),
        request_kind="response",
        task_id=task_id,
        aspect=aspect,
        config=config,
        telemetry=telemetry,
        model_client=model_client,
        request_semaphore=request_semaphore,
        max_attempts=max_attempts,
    )
    response = str(generated.get("response", "")).strip()
    if not response:
        raise UserSimulatorError("DeepSeek User Simulator returned an empty natural response")
    return UserSimulatorTurn(response, (), judgment_type, note, nonpreference_times + 1)


__all__ = [
    "PROMPT_VERSION",
    "UNAVAILABLE_RESPONSE",
    "VAGUE_RESPONSE",
    "UserSimulatorError",
    "UserSimulatorTurn",
    "build_history",
    "evaluate_user_question",
    "initialize_preferences",
    "model_call",
    "new_user_telemetry",
]
