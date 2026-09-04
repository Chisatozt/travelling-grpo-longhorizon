import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gyms" / "TravelGym"))

import travelgym  # noqa: E402
from travelgym.env.user_simulator import VAGUE_RESPONSE  # noqa: E402


TASK_ID = "flight:4-119|restaurant:4-291"


class FakeCompletions:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
        message = SimpleNamespace(content=json.dumps(value))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    def __init__(self, values):
        self.completions = FakeCompletions(values)
        self.chat = SimpleNamespace(completions=self.completions)


def make_env(tmp_path, monkeypatch, client):
    monkeypatch.setenv("TRAVELGYM_USER_CACHE_PATH", str(tmp_path / "responses.sqlite3"))
    monkeypatch.setenv("TRAVELGYM_USER_API_LOG_PATH", str(tmp_path / "events.jsonl"))
    config = travelgym.get_default_config()
    config.user_simulator_mode = "deepseek_api"
    config.api_key = "test-key"
    config.base_url = "https://api.deepseek.com"
    config.model_name = "deepseek-v4-flash"
    config.data_mode = "single"
    config.data_source = TASK_ID
    config.timeout = 2.0
    env = travelgym.TravelEnv(config)
    env._model_client = client
    env._model_max_attempts = 1
    env.reset()
    env.step("[search] Search for the requested restaurant in Paris.")
    return env


def test_specific_question_uses_judge_and_response_api(tmp_path, monkeypatch):
    client = FakeClient([
        {"type": "2", "preference_id": "P1"},
        {"thought": "Reveal it indirectly.", "response": "I would really enjoy a pastry or dessert-focused place."},
    ])
    env = make_env(tmp_path, monkeypatch, client)

    observation, reward, terminated, truncated, _ = asyncio.run(
        env.step_async(
            "[action] For the restaurant, which cuisine or food style would you most enjoy?"
        )
    )

    assert reward == 0.0
    assert not terminated and not truncated
    assert "pastry or dessert-focused" in observation["feedback"]
    assert "P1" not in observation["feedback"]
    assert len(client.completions.calls) == 2
    judge_call, response_call = client.completions.calls
    assert "Agent's Latest Utterance" in judge_call["messages"][1]["content"]
    assert judge_call["max_tokens"] == 128
    assert judge_call["response_format"] == {"type": "json_object"}
    assert judge_call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert response_call["max_tokens"] == 2048
    assert "response_format" not in response_call
    assert response_call["extra_body"] == {"thinking": {"type": "enabled"}}
    report = env.get_reward_report()
    assert report["reward_valid_for_training"] is True
    assert report["user_api_calls"] == 2
    assert report["user_judge_api_calls"] == 1
    assert report["user_response_api_calls"] == 1
    assert report["user_total_tokens"] == 30
    assert (tmp_path / "events.jsonl").is_file()


def test_vague_question_matches_teacher_policy(tmp_path, monkeypatch):
    client = FakeClient([{"type": "4"}])
    env = make_env(tmp_path, monkeypatch, client)

    observation, _, _, truncated, _ = asyncio.run(
        env.step_async("[action] Do you have any restaurant preferences?")
    )

    assert not truncated
    assert observation["feedback"].startswith(VAGUE_RESPONSE)
    assert len(client.completions.calls) == 1
    report = env.get_reward_report()
    assert report["hidden_preference_hit_rate"] == 0.0
    assert report["useful_action_count"] == 0
    assert report["no_gain_action_count"] == 1
    assert report["user_api_calls"] == 1


def test_api_failure_invalidates_trajectory_without_local_fallback(tmp_path, monkeypatch):
    client = FakeClient([RuntimeError("provider unavailable")])
    env = make_env(tmp_path, monkeypatch, client)

    observation, reward, terminated, truncated, _ = asyncio.run(
        env.step_async(
            "[action] For the restaurant, which cuisine or food style would you most enjoy?"
        )
    )

    assert reward == 0.0
    assert not terminated and truncated
    assert observation["feedback"].startswith("The environment operation failed.")
    report = env.get_reward_report()
    assert report["terminal_reward"] == 0.0
    assert report["reward_valid_for_training"] is False
    assert report["termination_reason"] == "user_simulator_api_error"
    assert report["user_api_calls"] == 1
    assert report["user_api_errors"] == 1


def test_response_cache_reuses_both_teacher_calls(tmp_path, monkeypatch):
    first_client = FakeClient([
        {"type": "2", "preference_id": "P1"},
        {"thought": "Reveal it indirectly.", "response": "A dessert-focused place sounds ideal."},
    ])
    first_env = make_env(tmp_path, monkeypatch, first_client)
    first_observation, *_ = asyncio.run(
        first_env.step_async(
            "[action] For the restaurant, which cuisine or food style would you most enjoy?"
        )
    )

    second_client = FakeClient([])
    second_env = make_env(tmp_path, monkeypatch, second_client)
    second_observation, *_ = asyncio.run(
        second_env.step_async(
            "[action] For the restaurant, which cuisine or food style would you most enjoy?"
        )
    )

    assert first_observation["feedback"] == second_observation["feedback"]
    assert len(first_client.completions.calls) == 2
    assert second_client.completions.calls == []
    report = second_env.get_reward_report()
    assert report["user_api_calls"] == 0
    assert report["user_cache_hits"] == 2


def test_transient_failure_is_retried_and_counted(tmp_path, monkeypatch):
    client = FakeClient([TimeoutError("temporary timeout"), {"type": "4"}])
    env = make_env(tmp_path, monkeypatch, client)
    env._model_max_attempts = 2

    observation, _, _, truncated, _ = asyncio.run(
        env.step_async("[action] Do you have any restaurant preferences?")
    )

    assert not truncated
    assert observation["feedback"].startswith(VAGUE_RESPONSE)
    assert len(client.completions.calls) == 2
    report = env.get_reward_report()
    assert report["reward_valid_for_training"] is True


def test_proactive_reveal_is_not_credited_as_agent_elicitation(tmp_path, monkeypatch):
    client = FakeClient([
        {"type": "4"},
        {"type": "4"},
        {"type": "4"},
        {"type": "4"},
        {"thought": "Offer context.", "response": "A dessert-focused place would be welcome."},
    ])
    env = make_env(tmp_path, monkeypatch, client)

    questions = [
        "Could you say something about restaurant preferences?",
        "Any broad thoughts about restaurant choices?",
        "What do you generally want from a restaurant?",
        "Could you share any restaurant preference at all?",
    ]
    for question in questions:
        _, _, _, truncated, _ = asyncio.run(
            env.step_async(f"[action] {question}")
        )
        assert not truncated

    report = env.get_reward_report()
    assert report["proactive_preference_count"] == 1
    assert report["agent_elicited_preference_count"] == 0
    assert report["hidden_preference_hit_rate"] == 0.0
    assert report["useful_action_count"] == 0
    assert report["no_gain_action_count"] == 4
    assert report["redundant_action_count"] == 3
    assert report["user_api_calls"] == 5
