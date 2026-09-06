from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import travelgym
from travelgym.env.actor_aspects import (
    build_actor_aspect_messages,
    parse_actor_aspect_response,
)


def _write_task(tmp_path):
    task_id = "actor-aspect-test"
    payload = {
        task_id: {
            "id": task_id,
            "dimensions": ["flight", "restaurant"],
            "scenario": "Book the trip.",
            "initial_description": "Book the trip.",
            "flight": {
                "all_ids": ["F1", "F2"],
                "correct_ids": ["F1"],
                "best_id": "F1",
                "preferences": [],
                "options": {
                    "correct": [{"id": "F1", "departure": "08:00"}],
                    "wrong": [{"id": "F2", "departure": "22:00"}],
                    "noise": [],
                },
            },
            "restaurant": {
                "all_ids": ["R1", "R2"],
                "correct_ids": ["R1"],
                "best_id": "R1",
                "preferences": [],
                "options": {
                    "correct": [{"id": "R1", "cuisine": "local"}],
                    "wrong": [{"id": "R2", "cuisine": "fast food"}],
                    "noise": [],
                },
            },
        }
    }
    data_path = tmp_path / "actor_aspect_task.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    return task_id, data_path


def _make_env(tmp_path, actor_result, *, enabled=True, max_steps=12):
    task_id, data_path = _write_task(tmp_path)
    config = travelgym.get_default_config()
    config.data_path = str(data_path)
    config.data_mode = "single"
    config.data_source = task_id
    config.user_simulator_mode = "local"
    config.enable_actor_aspect_extraction = enabled
    config.max_steps = max_steps
    env = travelgym.TravelEnv(config)
    env.set_actor_aspect_extraction(actor_result)
    return env


def test_extraction_prompt_contains_only_public_requirement_contract():
    requirement = "Please arrange a flight and a restaurant reservation."
    messages = build_actor_aspect_messages(requirement)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == requirement
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "dimensions" not in serialized
    assert "task_id" not in serialized
    assert "secret-task" not in serialized
    assert "candidate" not in serialized
    assert "preference" not in serialized
    assert "correct_ids" not in serialized
    assert "best_id" not in serialized


def test_parser_retains_order_and_invalid_duplicate_items_without_repairing():
    result = parse_actor_aspect_response(
        '{"aspects":["restaurant","cruise","flight","flight"]}'
    )

    assert result.aspects == ("restaurant", "cruise", "flight", "flight")
    assert result.invalid_aspects == ("cruise",)
    assert result.duplicate_aspects == ("flight",)
    assert result.error_count == 2


def test_eval_actor_call_reuses_public_prompt_only():
    from travel_grpo.evaluation.eval import _extract_actor_aspects_for_eval

    class Completions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=3,
                    total_tokens=14,
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"aspects":["restaurant","flight"]}'
                        )
                    )
                ],
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    telemetry = {}
    result = asyncio.run(
        _extract_actor_aspects_for_eval(
            "Book a restaurant and then a flight.",
            client=client,
            model_name="test-actor",
            telemetry=telemetry,
        )
    )

    assert result.aspects == ("restaurant", "flight")
    serialized = json.dumps(completions.kwargs["messages"], ensure_ascii=False)
    assert "Book a restaurant and then a flight." in serialized
    assert "secret-task" not in serialized
    assert "dimensions" not in serialized
    assert "correct_ids" not in serialized
    assert "tools" not in completions.kwargs
    assert telemetry["actor_aspect_extraction_calls"] == 1
    assert telemetry["actor_aspect_extraction_total_tokens"] == 14


def test_actor_plan_controls_execution_order_and_public_state(tmp_path):
    env = _make_env(tmp_path, {"aspects": ["restaurant", "flight"]})
    observation, _ = env.reset()
    assert observation["current_aspect"] == "restaurant"

    env.step("[search] Search for restaurant options.")
    observation, _, _, _, _ = env.step("[answer] R1")
    assert observation["current_aspect"] == "flight"
    assert "actor_aspects" not in observation
    assert "required_aspects" not in observation

    env.step("[search] Search for flight options.")
    _, _, terminated, truncated, _ = env.step("[answer] F1")
    assert terminated and not truncated
    report = env.get_reward_report()
    assert report["actor_aspects"] == ["restaurant", "flight"]
    assert report["answer_coverage"] == 1.0
    assert report["completion_success"] == 1.0


def test_omitted_required_aspect_blocks_complete_success(tmp_path):
    env = _make_env(tmp_path, {"aspects": ["flight"]})
    env.step("[search] Search for flight options.")
    env.step("[answer] F1")

    report = env.get_reward_report()
    assert report["actor_aspects"] == ["flight"]
    assert report["answer_coverage"] == 0.5
    assert report["unanswered_count"] == 1
    assert report["completion_success"] == 0.0
    assert report["reward_valid"] is True


def test_invalid_actor_aspect_is_retained_and_rejected_explicitly(tmp_path):
    env = _make_env(tmp_path, {"aspects": ["cruise", "flight"]})
    observation, _, _, _, _ = env.step("[search] Search for the current aspect.")

    assert "unsupported category" in observation["feedback"]
    env.step("[finish]")
    report = env.get_reward_report()
    assert report["actor_aspects"] == ["cruise", "flight"]
    assert report["actor_aspect_extraction_invalid_aspects"] == ["cruise"]
    assert report["actor_aspect_extraction_error_count"] == 1
    assert report["reward_valid"] is True


def test_known_but_unavailable_actor_aspect_is_not_silently_removed(tmp_path):
    env = _make_env(tmp_path, {"aspects": ["hotel"]})
    observation, _, _, _, _ = env.step("[search] Search for the hotel.")

    assert "no options are available" in observation["feedback"]
    env.step("[finish]")
    report = env.get_reward_report()
    assert report["actor_aspects"] == ["hotel"]
    assert report["actor_aspect_extraction_invalid_aspects"] == []
    assert report["reward_valid"] is True


def test_format_error_does_not_fallback_to_hidden_dimensions(tmp_path):
    env = _make_env(tmp_path, parse_actor_aspect_response("not json"))
    observation, _ = env.reset()

    assert observation["current_aspect"] is None
    assert "flight" not in observation["feedback"]
    env.step("[finish]")
    report = env.get_reward_report()
    assert report["actor_aspects"] == []
    assert report["actor_aspect_extraction_format_error"] == "invalid_json_object"
    assert report["answer_coverage"] == 0.0
    assert report["completion_success"] == 0.0
    assert report["reward_valid"] is True


def test_disabled_flag_preserves_legacy_hidden_dimension_order(tmp_path):
    env = _make_env(tmp_path, {"aspects": ["restaurant"]}, enabled=False)
    observation, _ = env.reset()

    assert observation["current_aspect"] == "flight"
    report = env.get_reward_report()
    assert report["actor_aspect_extraction_enabled"] is False
    assert report["actor_aspects"] is None
    assert report["actor_aspect_extraction_direct_grpo_signal"] is False
