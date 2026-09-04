from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gyms" / "TravelGym"))

import travelgym  # noqa: E402


def make_env(tmp_path: Path, *, max_steps: int = 10):
    tmp_path.mkdir(parents=True, exist_ok=True)
    task_id = "tiny-flight"
    payload = {
        task_id: {
            "id": task_id,
            "dimensions": ["flight"],
            "scenario": "Choose one flight.",
            "initial_description": "I need a flight.",
            "flight": {
                "all_ids": ["F1", "F2"],
                "correct_ids": ["F1"],
                "best_id": "F1",
                "preferences": [
                    [
                        "flight",
                        "schedule",
                        "prefer a morning departure",
                        "An early departure would suit me best.",
                        "The correct option departs in the morning.",
                    ]
                ],
                "options": {
                    "correct": [{"id": "F1", "departure": "08:00"}],
                    "wrong": [{"id": "F2", "departure": "22:00"}],
                    "noise": [],
                },
            },
        }
    }
    data_path = tmp_path / "tiny.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    config = travelgym.get_default_config()
    config.data_path = str(data_path)
    config.data_mode = "single"
    config.data_source = task_id
    config.user_simulator_mode = "local"
    config.max_steps = max_steps
    env = travelgym.TravelEnv(config)
    env.reset()
    return env


def test_complete_useful_chain_gets_full_quality_credit(tmp_path):
    env = make_env(tmp_path)
    env.step("[search] Search for the requested flight.")
    env.step("[action] What departure time do you prefer for the flight?")
    env.step("[answer] F1")

    report = env.get_reward_report()
    assert report["reward_version"] == "travelgym-terminal-v2"
    assert report["answer_coverage"] == 1.0
    assert report["coverage_adjusted_answer_quality"] == 1.0
    assert report["coverage_adjusted_legal_chain_rate"] == 1.0
    assert report["hidden_preference_hit_rate"] == 1.0
    assert report["agent_elicited_preference_count"] == 1
    assert report["proactive_preference_count"] == 0
    assert report["useful_action_count"] == 1
    assert report["redundant_action_count"] == 0
    assert report["total_penalty"] == 0.0


def test_one_no_gain_grace_then_duplicate_is_penalized(tmp_path):
    env = make_env(tmp_path)
    env.step("[search] Search for the requested flight.")
    env.step("[action] What departure time do you prefer for the flight?")
    env.step("[action] Do you care about the flight meal?")
    env.step("[action] Do you care about the flight meal?")
    env.step("[answer] F1")

    report = env.get_reward_report()
    assert report["useful_action_count"] == 1
    assert report["no_gain_action_count"] == 2
    assert report["duplicate_action_count"] == 1
    assert report["redundant_action_count"] == 1
    assert report["redundant_action_penalty"] == pytest.approx(0.05)


def test_max_steps_and_zero_answer_penalties_stack(tmp_path):
    env = make_env(tmp_path, max_steps=2)
    env.step("[search] Search for the requested flight.")
    observation, _, terminated, truncated, _ = env.step(
        "[action] What departure time do you prefer for the flight?"
    )

    assert not terminated and truncated
    assert "remaining turn budget" in observation["feedback"]
    report = env.get_reward_report()
    assert report["termination_reason"] == "max_steps"
    assert report["answer_coverage"] == 0.0
    assert report["unanswered_count"] == 1
    assert report["incomplete_penalty"] == 1.0
    assert report["zero_answer_penalty"] == 0.5
    assert report["max_steps_penalty"] == 0.75
    assert report["policy_penalty"] == pytest.approx(0.05)
    assert report["terminal_reward"] < 0.0


def test_deadline_guard_reserves_last_step_for_answer(tmp_path):
    env = make_env(tmp_path, max_steps=3)
    env.step("[search] Search for the requested flight.")
    observation, _, _, truncated, _ = env.step(
        "[action] What departure time do you prefer for the flight?"
    )
    assert not truncated
    assert "remaining turn budget" in observation["feedback"]

    _, _, terminated, truncated, _ = env.step("[answer] F1")
    assert terminated and not truncated
    report = env.get_reward_report()
    assert report["answer_coverage"] == 1.0
    assert report["max_steps_reached"] == 0.0
    assert report["max_steps_penalty"] == 0.0


def test_valid_wrong_answer_scores_above_no_answer(tmp_path):
    wrong = make_env(tmp_path / "wrong")
    wrong.step("[search] Search for the requested flight.")
    wrong.step("[action] What departure time do you prefer for the flight?")
    wrong.step("[answer] F2")

    missing = make_env(tmp_path / "missing")
    missing.step("[search] Search for the requested flight.")
    missing.step("[action] What departure time do you prefer for the flight?")
    missing.step("[finish]")

    wrong_report = wrong.get_reward_report()
    missing_report = missing.get_reward_report()
    assert wrong_report["answer_coverage"] == 1.0
    assert wrong_report["wrong_answer_count"] == 1
    assert missing_report["answer_coverage"] == 0.0
    assert wrong_report["terminal_reward"] > missing_report["terminal_reward"]


def test_proactive_reveal_has_no_hidden_or_useful_action_credit(tmp_path):
    env = make_env(tmp_path)
    _, proactive = env._register_revealed_preferences(proactive_ids=("P1",))
    env._record_action_outcome("flight", "Tell me anything about the flight.", set())
    env.step("[finish]")

    report = env.get_reward_report()
    assert proactive == {"P1"}
    assert report["proactive_preference_count"] == 1
    assert report["agent_elicited_preference_count"] == 0
    assert report["hidden_preference_hit_rate"] == 0.0
    assert report["useful_action_count"] == 0
    assert report["no_gain_action_count"] == 1
    assert report["redundant_action_count"] == 0

def test_turn_credit_events_capture_behavior_without_private_ids(tmp_path):
    env = make_env(tmp_path)

    before = env.get_turn_credit_snapshot()
    env.step("[search] Search for the requested flight.")
    search_event = env.build_turn_credit_event(before, choice="search")

    before = env.get_turn_credit_snapshot()
    env.step("[action] What departure time do you prefer for the flight?")
    useful_event = env.build_turn_credit_event(before, choice="action")

    before = env.get_turn_credit_snapshot()
    env.step("[answer] F1")
    answer_event = env.build_turn_credit_event(before, choice="answer")

    assert search_event["new_search"]
    assert useful_event["useful_action"]
    assert useful_event["new_preference_count"] == 1
    assert answer_event["completed_aspect"]
    assert answer_event["correct_answer"]
    assert answer_event["best_answer"]
    assert answer_event["legal_answer"]
    serialized = json.dumps([search_event, useful_event, answer_event])
    assert "correct_ids" not in serialized
    assert "best_id" not in serialized
    assert "preference_ids" not in serialized


def test_external_invalid_call_produces_penalty_event_and_consumes_budget(tmp_path):
    env = make_env(tmp_path, max_steps=1)
    before = env.get_turn_credit_snapshot()
    env.register_external_invalid_call()
    event = env.build_turn_credit_event(before, choice="")

    assert event["invalid_call"]
    assert not event["accepted"]
    assert event["outcome"] == "invalid_call"
    assert event["truncated"]
    assert event["termination_reason"] == "max_steps"
    report = env.get_reward_report()
    assert report["invalid_call_count"] == 1
    assert report["max_steps_reached"] == 1.0

