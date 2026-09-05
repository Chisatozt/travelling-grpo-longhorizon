"""CPU-level acceptance tests for canonical TravelGym SFT preparation."""

from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from travel_grpo.collection.clean_travel_trajectories import clean_trajectory, is_sft_eligible
from travel_grpo.training.sft.qwen35_mask import (
    assert_template_equivalence,
    causal_target_mask,
    exact_assistant_token_mask,
    native_template_ids,
    template_messages,
)
from travel_grpo.collection.travel_canonical import canonical_hash, canonical_tools_schema, canonicalize_record, validate_canonical
from verl.tools.travel_tool_adapter import format_environment_action, normalize_tool_call, sanitize_public_feedback
from verl.trainer.ppo.hard_case_pool import HardCasePool, compose_task_key


def assistant(choice: str, content: str, *, think: str = "decide") -> dict:
    payload = {"name": "interact_with_env", "arguments": {"choice": choice, "content": content}}
    return {"role": "assistant", "content": f"<think>{think}</think>\n<tool_call>{json.dumps(payload)}</tool_call>"}


def task_one() -> dict:
    return {"id": "task-1", "dimensions": ["flight"], "flight": {"all_ids": ["F1", "F2"], "correct_ids": ["F1"], "best_id": "F1"}}


def complete_messages(answer: str = "F1") -> list[dict]:
    return [
        {"role": "user", "content": "book a flight"},
        assistant("search", "Search flight"),
        {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"},
        assistant("action", "Which amenities matter most?"),
        {"role": "tool", "content": "I prefer a direct flight."},
        assistant("answer", answer),
        {"role": "tool", "content": "Your answer was recorded."},
    ]


class CanonicalPipelineTests(unittest.TestCase):
    def test_hard_case_identity_includes_environment_variant(self):
        self.assertEqual(compose_task_key("t1", "travel22"), "travel22::t1")
        self.assertEqual(compose_task_key("travel33::t1", "travel22"), "travel33::t1")
        self.assertEqual(compose_task_key("t1"), "t1")

    def test_legacy_and_deepseek_shapes_are_canonical(self):
        legacy = {"system": "TravelGym", "conversations": [{"from": "human", "value": "book"}] + complete_messages()[1:]}
        # complete_messages uses role fields; convert the assistant/tool tail
        # to a second input shape rather than relying on a model-specific text.
        canonical = canonicalize_record(legacy)
        validate_canonical(canonical)
        self.assertEqual(canonical["schema_version"], "travelgym-canonical-v1")
        self.assertEqual(canonical["messages"][0]["role"], "system")
        self.assertEqual(canonical["messages"][2]["tool_calls"][0]["function"]["arguments"]["choice"], "search")
        deepseek = {
            "messages": [
                {"role": "user", "content": "book"},
                {"role": "assistant", "reasoning_content": "decide", "content": {"name": "interact_with_env", "arguments": {"choice": "search", "content": "Search flight"}}},
                {"role": "tool", "content": "F1 F2"},
            ]
        }
        converted = canonicalize_record(deepseek)
        self.assertEqual(converted["messages"][1]["reasoning_content"], "decide")
        self.assertEqual(converted["messages"][2]["tool_call_id"], converted["messages"][1]["tool_calls"][0]["id"])

    def test_recoverable_error_is_context_only_and_repair_is_supervised(self):
        messages = [
            {"role": "user", "content": "book"},
            assistant("action", "ask before searching"),
            {"role": "tool", "content": "Tool call rejected: action-before-search."},
        ] + complete_messages()[1:]
        cleaned, audit = clean_trajectory({"messages": messages}, task=task_one())
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct")
        self.assertEqual(cleaned["assistant_train_mask"][1], 0)
        # Search repair and later correct turns remain targets.
        self.assertGreaterEqual(sum(cleaned["assistant_train_mask"]), 3)
        self.assertEqual(audit["retained_recoverable_errors"][0]["kind"], "action-before-search")

    def test_unrepaired_and_wrong_terminal_answers_truncate_suffix(self):
        # No later valid protocol action repairs this public rejection, so the
        # offending assistant turn and its observation are removed together
        # with the untrusted suffix.
        bad = {"messages": [{"role": "user", "content": "book"}, assistant("action", "before search"), {"role": "tool", "content": "Tool call rejected: action-before-search."}]}
        cleaned, audit = clean_trajectory(bad, task=task_one())
        self.assertTrue(audit["fatal_truncation"])
        self.assertEqual(audit["fatal_kind"], "action-before-search")
        self.assertEqual(len(cleaned["messages"]), 1)
        wrong, wrong_audit = clean_trajectory({"messages": complete_messages("F2")}, task=task_one())
        self.assertEqual(wrong_audit["fatal_kind"], "wrong-terminal-answer")
        self.assertEqual(wrong["trainer_metadata"]["trajectory_class"], "totally_wrong")

    def test_nonfinal_wrong_answer_stays_context_and_fatal_suffix_can_be_strict(self):
        task = {"id": "task-2", "dimensions": ["flight", "hotel"], "flight": {"all_ids": ["F1", "F2"], "correct_ids": ["F2"]}, "hotel": {"all_ids": ["H1"], "correct_ids": ["H1"]}}
        messages = complete_messages("F1") + [
            assistant("search", "Search hotel"), {"role": "tool", "content": "Here are all the options for <hotel>: H1"},
            assistant("answer", "H1"), {"role": "tool", "content": "recorded"},
        ]
        cleaned, audit = clean_trajectory({"messages": messages}, task=task)
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "partial_correct")
        # The wrong first answer is not supervised, while the later answer is.
        answer_indices = [i for i, m in enumerate(cleaned["messages"]) if m.get("role") == "assistant" and m.get("tool_calls") and m["tool_calls"][0]["function"]["arguments"]["choice"] == "answer"]
        self.assertEqual(cleaned["assistant_train_mask"][answer_indices[0]], 0)
        self.assertEqual(cleaned["assistant_train_mask"][answer_indices[-1]], 1)

    def test_mask_alignment_and_adapter(self):
        messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "tool", "content": "o"}]
        def render(items, generation):
            result = [101]
            for item in items:
                result += [len(item["role"]), len(item.get("content", ""))]
            if generation:
                result += [999]
            return result
        full = render(messages, False)
        mask = exact_assistant_token_mask(messages, full, render, [0, 1, 0])
        self.assertEqual(full, render(messages, False))
        self.assertEqual(mask[3:5], [1, 1])
        call = {"id": "x", "type": "function", "function": {"name": "interact_with_env", "arguments": '{"choice":"answer","content":"F1"}'}}
        self.assertEqual(normalize_tool_call(call), {"choice": "answer", "content": "F1"})
        self.assertEqual(format_environment_action({"choice": "search", "content": "flight"}), "[search] flight")
        self.assertNotIn("Reward:", sanitize_public_feedback("Reward: 1.0\npublic feedback"))

    def test_causal_target_mask_supervises_the_target_token(self):
        # token_mask is aligned with input_ids. Causal losses predict
        # input_ids[1:], so the first token-position mask must be discarded.
        token_mask = [0, 0, 1, 1, 0]
        self.assertEqual(causal_target_mask(token_mask), [0, 1, 1, 0])
        predicted_tokens = [11, 12, 13, 14]
        supervised = [token for token, keep in zip(predicted_tokens, causal_target_mask(token_mask)) if keep]
        self.assertEqual(supervised, [12, 13])

    def test_native_template_accepts_transformers_five_batch_encoding(self):
        class MappingTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                del messages, kwargs
                return {"input_ids": [[101, 102, 103]]}

        self.assertEqual(
            native_template_ids(MappingTokenizer(), [{"role": "user", "content": "hello"}]),
            [101, 102, 103],
        )

    def test_hard_case_pool_requires_three_valid_zero_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = HardCasePool(Path(directory) / "pool.json", threshold=3)
            for step in (1, 2):
                result = pool.observe_group(task_id="t", reward_valid=[True, True], correct_completion=[0, 0], group_size=2, step=step)
                self.assertFalse(result["qualified"])
            result = pool.observe_group(task_id="t", reward_valid=[True, True], correct_completion=[0, 0], group_size=2, step=3)
            self.assertTrue(result["qualified"])
            self.assertIn("t", pool.admitted)
            pool.observe_group(task_id="invalid", reward_valid=[False, False], correct_completion=[0, 0], group_size=2, step=4)
            self.assertNotIn("invalid", pool.admitted)
            # The state is checkpoint-friendly and restoring it does not alter
            # the rollout/sampler API.
            state = pool.state_dict()
            restored = HardCasePool(Path(directory) / "restored.json", threshold=3)
            restored.load_state_dict(state)
            self.assertEqual(restored.admitted, pool.admitted)
            self.assertEqual(restored.streaks, pool.streaks)

    def test_hard_case_pool_ignores_nonterminal_metadata(self):
        class Output:
            non_tensor_batch = {
                "reward_scores": [
                    {
                        "interact_with_env": 0.0,
                        "interact_with_env_reward_valid": 1.0,
                        "interact_with_env_terminal_only": 0.0,
                        "interact_with_env_correct_completion": 0.0,
                    },
                    {
                        "interact_with_env": 0.0,
                        "interact_with_env_reward_valid": 1.0,
                        "interact_with_env_terminal_only": 0.0,
                        "interact_with_env_correct_completion": 0.0,
                    },
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            pool = HardCasePool(Path(directory) / "pool.json", threshold=1)
            self.assertEqual(
                pool.observe_output(Output(), task_ids=["t", "t"], group_size=2, step=1),
                [],
            )
            self.assertFalse(pool.admitted)

    def test_hard_case_pool_rejects_partial_group_metadata(self):
        class Output:
            non_tensor_batch = {
                "reward_scores": [
                    {"interact_with_env": 0.0, "interact_with_env_reward_valid": 1.0,
                     "interact_with_env_terminal_only": 1.0, "interact_with_env_correct_completion": 0.0},
                    {"interact_with_env": 0.0, "interact_with_env_reward_valid": 1.0,
                     "interact_with_env_terminal_only": 1.0, "interact_with_env_correct_completion": 0.0},
                    # A third row cannot form a second complete group of two.
                    {"interact_with_env": 0.0, "interact_with_env_reward_valid": 1.0,
                     "interact_with_env_terminal_only": 1.0, "interact_with_env_correct_completion": 0.0},
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            pool = HardCasePool(Path(directory) / "pool.json", threshold=1)
            self.assertEqual(
                pool.observe_output(
                    Output(), task_ids=["t", "t", "t"],
                    sources=["s", "s", "s"],
                    group_size=2, step=1,
                ),
                [],
            )
            self.assertFalse(pool.admitted)

    def test_sft_eligibility_excludes_wrong_and_infrastructure_rows(self):
        base = {"messages": [{"role": "user", "content": "u"}, assistant("search", "Search flight")], "assistant_train_mask": [0, 1]}
        for category in ("strict_gold", "recoverable_correct", "partial_correct"):
            row = {**base, "trainer_metadata": {"trajectory_class": category, "sample_weight": 0.5 if category == "partial_correct" else 1.0}}
            self.assertTrue(is_sft_eligible(row))
        for category in ("totally_wrong", "infrastructure_invalid", "overlength_quarantine"):
            row = {**base, "trainer_metadata": {"trajectory_class": category, "sample_weight": 0.0}}
            self.assertFalse(is_sft_eligible(row))

    def _run_recoverable(self, error_call, error_feedback, repair_tail=None, *, task=None):
        task = task or task_one()
        messages = [{"role": "user", "content": "book"}]
        messages.extend(error_call)
        messages.extend(repair_tail or complete_messages()[1:])
        cleaned, audit = clean_trajectory({"messages": messages}, task=task)
        self.assertFalse(audit.get("fatal_truncation"), audit)
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct", audit)
        self.assertTrue(audit["retained_recoverable_errors"])
        error_index = next(
            index for index, value in enumerate(cleaned["assistant_train_mask"])
            if value == 0 and cleaned["messages"][index].get("role") == "assistant"
        )
        self.assertEqual(cleaned["assistant_train_mask"][error_index], 0)
        return cleaned, audit

    def test_each_public_recoverable_error_is_kept_with_repair(self):
        # The rejected pair is context-only; the later legal call is the
        # supervised repair.  Feedback strings intentionally contain only the
        # public reason emitted by PublicControl/TravelEnv.
        cases = [
            (assistant("action", "ask before search"), "Tool call rejected: action-before-search."),
            (assistant("answer", "F1"), "Tool call rejected: answer-before-search."),
        ]
        for bad, feedback in cases:
            with self.subTest(feedback=feedback):
                self._run_recoverable([bad, {"role": "tool", "content": feedback}], feedback)

        # Cross-aspect and repeated Search occur after a valid flight Search;
        # the repair returns to the current aspect.
        prefix = [
            assistant("search", "Search flight"),
            {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"},
        ]
        for bad, feedback, tail in [
            (
                assistant("action", "Ask about hotel amenities"),
                "Tool call rejected: cross-aspect operation.",
                [assistant("action", "Which flight amenities matter?"), {"role": "tool", "content": "direct"}, assistant("answer", "F1"), {"role": "tool", "content": "recorded"}],
            ),
            (
                assistant("search", "Search flight again"),
                "Tool call rejected: repeated tool call.",
                [assistant("action", "Which flight amenities matter?"), {"role": "tool", "content": "direct"}, assistant("answer", "F1"), {"role": "tool", "content": "recorded"}],
            ),
        ]:
            with self.subTest(feedback=feedback):
                cleaned, audit = clean_trajectory(
                    {"messages": [{"role": "user", "content": "book"}] + prefix + [bad, {"role": "tool", "content": feedback}] + tail},
                    task=task_one(),
                )
                self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct", audit)
                self.assertEqual(audit["retained_recoverable_errors"][0]["kind"], "cross-aspect" if "cross" in feedback else "repeated-search")

        # Parseable but semantically invalid arguments (empty content) can be
        # repaired with the same operation.
        invalid = assistant("action", "")
        cleaned, audit = clean_trajectory(
            {"messages": [{"role": "user", "content": "book"}, assistant("search", "Search flight"), {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"}, invalid, {"role": "tool", "content": "Tool call rejected: invalid tool parameters."}, assistant("action", "Which flight amenities matter?"), {"role": "tool", "content": "direct"}, assistant("answer", "F1"), {"role": "tool", "content": "recorded"}]},
            task=task_one(),
        )
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct", audit)
        self.assertEqual(audit["retained_recoverable_errors"][0]["kind"], "invalid-parameters")

        # Invisible ID and duplicate Answer both have a public repair.
        cleaned, audit = clean_trajectory(
            {"messages": [{"role": "user", "content": "book"}, assistant("search", "Search flight"), {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"}, assistant("answer", "F9"), {"role": "tool", "content": "Tool call rejected: answer ID was not visible in Search results."}, assistant("answer", "F1"), {"role": "tool", "content": "recorded"}]},
            task=task_one(),
        )
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct", audit)
        self.assertEqual(audit["retained_recoverable_errors"][0]["kind"], "invisible-id")

        two = {"id": "two", "dimensions": ["flight", "hotel"], "flight": {"all_ids": ["F1", "F2"], "correct_ids": ["F1"]}, "hotel": {"all_ids": ["H1"], "correct_ids": ["H1"]}}
        duplicate_messages = [
            {"role": "user", "content": "book"},
            assistant("search", "Search flight"), {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"},
            assistant("answer", "F1"), {"role": "tool", "content": "recorded"},
            assistant("answer", "F1"), {"role": "tool", "content": "Tool call rejected: the aspect was already answered."},
            assistant("search", "Search hotel"), {"role": "tool", "content": "Here are all the options for <hotel>: H1"},
            assistant("answer", "H1"), {"role": "tool", "content": "recorded"},
        ]
        cleaned, audit = clean_trajectory({"messages": duplicate_messages}, task=two)
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct", audit)
        self.assertEqual(audit["retained_recoverable_errors"][0]["kind"], "duplicate-answer")

        vague_messages = [
            {"role": "user", "content": "book"},
            assistant("search", "Search flight"), {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"},
            assistant("action", "Tell me about travel"), {"role": "tool", "content": "Your question is too vague and general."},
            assistant("action", "Which flight amenities matter?"), {"role": "tool", "content": "direct"},
            assistant("answer", "F1"), {"role": "tool", "content": "recorded"},
        ]
        cleaned, audit = clean_trajectory({"messages": vague_messages}, task=task_one())
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "recoverable_correct", audit)
        self.assertEqual(audit["retained_recoverable_errors"][0]["kind"], "vague-action")

    def test_inferred_error_without_public_rejection_truncates_suffix(self):
        # Turn order alone is not enough to certify a recoverable error.  If
        # the environment did not expose its refusal, discard the untrusted
        # suffix even when a later legal-looking repair exists.
        messages = [
            {"role": "user", "content": "book"},
            assistant("action", "ask before search"),
            {"role": "tool", "content": "The request was not processed."},
            *complete_messages()[1:],
        ]
        cleaned, audit = clean_trajectory({"messages": messages}, task=task_one())
        self.assertTrue(audit["fatal_truncation"])
        self.assertEqual(audit["fatal_kind"], "unrepaired-recoverable-error")
        self.assertEqual(len(cleaned["messages"]), 1)

    def test_fatal_suffix_removed_then_success_is_strict_gold(self):
        messages = complete_messages("F1") + [
            {"role": "assistant", "content": "<think>unfinished"},
            {"role": "tool", "content": "ignored"},
        ]
        cleaned, audit = clean_trajectory({"messages": messages}, task=task_one(), require_think=True)
        self.assertTrue(audit["fatal_truncation"])
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "strict_gold")
        self.assertEqual(cleaned["trainer_metadata"]["sample_weight"], 1.0)
        self.assertEqual(cleaned["messages"][-1]["role"], "tool")

    def test_parse_errors_and_observation_mismatch_truncate_from_assistant(self):
        malformed = {"messages": [{"role": "user", "content": "book"}, {"role": "assistant", "content": "<think>x</think><tool_call>{bad}</tool_call>"}, {"role": "tool", "content": "stale"}, assistant("search", "Search flight"), {"role": "tool", "content": "F1 F2"}]}
        cleaned, audit = clean_trajectory(malformed, task=task_one())
        self.assertEqual(len(cleaned["messages"]), 1)
        self.assertEqual(audit["fatal_kind"], "tool_call_json_invalid")

        mismatch = {"messages": [{"role": "user", "content": "book"}, {"role": "assistant", "reasoning_content": "x", "content": "", "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "interact_with_env", "arguments": {"choice": "search", "content": "Search flight"}}}]}, {"role": "tool", "tool_call_id": "call_b", "content": "F1 F2"}, assistant("answer", "F1"), {"role": "tool", "content": "stale"}]}
        cleaned, audit = clean_trajectory(mismatch, task=task_one())
        self.assertEqual(len(cleaned["messages"]), 1)
        self.assertIn(audit["fatal_kind"], {"tool_call_id_mismatch", "tool_observation_misaligned"})

    def test_missing_think_is_infrastructure_invalid_but_hidden_text_is_not_a_drop_rule(self):
        no_think = {"messages": [{"role": "user", "content": "book"}, {"role": "assistant", "content": json.dumps({"name": "interact_with_env", "arguments": {"choice": "search", "content": "Search flight"}})}, {"role": "tool", "content": "Here are all the options for <flight>: F1 F2"}, assistant("answer", "F1"), {"role": "tool", "content": "recorded"}]}
        cleaned, audit = clean_trajectory(no_think, task=task_one(), require_think=True)
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "infrastructure_invalid")
        self.assertTrue(audit.get("missing_think"))
        hidden = complete_messages("F1")
        hidden[2]["content"] = "Reward: 1.0\nHere are all the options for <flight>: F1 F2"
        cleaned, audit = clean_trajectory({"messages": hidden}, task=task_one())
        self.assertEqual(cleaned["trainer_metadata"]["trajectory_class"], "strict_gold")

    def test_template_equivalence_is_strict(self):
        official = [1, 2, 3, 4]
        assert_template_equivalence(official, [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            assert_template_equivalence(official, [1, 2, 9, 4])

        messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "tool", "content": "o"}]
        def render(items, generation):
            ids = [101]
            for item in items:
                ids += [len(item["role"]), len(item.get("content", ""))]
            if generation:
                ids += [999]
            return ids
        full = render(messages, False)
        assert_template_equivalence(full, render(messages, False))
        mask = exact_assistant_token_mask(messages, full, render, [0, 1, 0])
        self.assertEqual(mask[3:5], [1, 1])

    def test_qwen_template_boundary_preserves_structured_arguments(self):
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "interact_with_env",
                    "arguments": {"choice": "search", "content": "flight"},
                },
            }],
        }]
        prepared = template_messages(messages)
        self.assertIsInstance(prepared[0]["tool_calls"][0]["function"]["arguments"], dict)
        self.assertIsInstance(messages[0]["tool_calls"][0]["function"]["arguments"], dict)

        legacy = copy.deepcopy(messages)
        legacy[0]["tool_calls"][0]["function"]["arguments"] = '{"choice":"search","content":"flight"}'
        parsed = template_messages(legacy)
        self.assertEqual(parsed[0]["tool_calls"][0]["function"]["arguments"], messages[0]["tool_calls"][0]["function"]["arguments"])

    def test_tool_schema_is_shared_with_eval_and_sglang_config(self):
        expected = canonical_tools_schema()[0]
        eval_schema = yaml.safe_load(Path("configs/tools/interact_tool_schema.yaml").read_text(encoding="utf-8"))["tool_schema"]
        sglang_schema = yaml.safe_load(Path("configs/tools/interact_tool_config.yaml").read_text(encoding="utf-8"))["tools"][0]["tool_schema"]
        self.assertEqual(eval_schema, expected)
        self.assertEqual(sglang_schema, expected)


if __name__ == "__main__":
    unittest.main()
