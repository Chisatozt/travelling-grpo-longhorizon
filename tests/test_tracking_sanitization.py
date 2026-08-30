from __future__ import annotations

import unittest
from collections import UserDict

from verl.utils.tracking import sanitize_tracking_payload, sanitize_validation_samples


class TrackingSanitizationTests(unittest.TestCase):
    def test_sensitive_values_are_not_exported(self):
        value = sanitize_tracking_payload({"api_key": "sk-secret", "preference_id": "P1", "raw_reward_ledger": {"x": 1}, "collection/api_errors": 2})
        self.assertNotIn("api_key", value)
        self.assertNotIn("preference_id", value)
        self.assertNotIn("raw_reward_ledger", value)
        self.assertEqual(value["collection/api_errors"], 2)

    def test_mapping_configs_are_sanitized_recursively(self):
        value = sanitize_tracking_payload(
            UserDict(
                {
                    "trainer": UserDict({"messages": [{"content": "private"}], "total_training_steps": 200}),
                    "run_label": "public-label",
                }
            )
        )
        self.assertNotIn("messages", value["trainer"])
        self.assertEqual(value["trainer"]["total_training_steps"], 200)

    def test_validation_samples_redact_reasoning_blocks(self):
        samples = sanitize_validation_samples([
            ("prompt", "<think>private chain</think>\n<tool_call>{}</tool_call>", 0.5),
        ])
        self.assertNotIn("private chain", samples[0][1])
        self.assertIn("[REASONING_REDACTED]", samples[0][1])

    def test_generic_text_values_are_scrubbed_too(self):
        value = sanitize_tracking_payload({
            "text": "<think>secret chain</think>\npreference_ids: [P1]\npublic scalar",
        })
        self.assertNotIn("secret chain", value["text"])
        self.assertNotIn("P1", value["text"])
        self.assertIn("public scalar", value["text"])


if __name__ == "__main__":
    unittest.main()
