import unittest

from ifpruning_data import (
    DEFAULT_CHAT_TEMPLATE,
    build_sft_sequence,
    extract_sft_examples,
    ensure_bos,
    truncate_prompt,
)


class DataNormalizationTests(unittest.TestCase):
    def test_default_template_contains_required_no_thinking_channel(self):
        self.assertIn(r"<|channel>thought\n<channel|>", DEFAULT_CHAT_TEMPLATE)

    def test_system_and_every_multiturn_target_are_preserved(self):
        examples = {
            "conversations": [[
                {"from": "system", "value": "Answer as a scientist."},
                {"from": "human", "value": "Initial question"},
                {"from": "gpt", "value": "Initial answer"},
                {"from": "human", "value": "Follow-up question"},
                {"from": "gpt", "value": "Final answer"},
            ]]
        }
        router_prompt, context, response = extract_sft_examples(examples, 0)[-1]
        self.assertEqual(router_prompt, "Initial question")
        self.assertEqual(response, "Final answer")
        self.assertEqual(
            [message["role"] for message in context],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(context[-1]["content"], "Follow-up question")

        targets = extract_sft_examples(examples, 0)
        self.assertEqual(len(targets), 2)
        self.assertEqual([target[2] for target in targets], ["Initial answer", "Final answer"])
        self.assertTrue(all(target[0] == "Initial question" for target in targets))
        self.assertEqual(
            [message["role"] for message in targets[0][1]],
            ["system", "user"],
        )

    def test_alpaca_input_is_part_of_the_prompt(self):
        examples = {
            "instruction": ["Summarize"],
            "input": ["A long article"],
            "output": ["A summary"],
        }
        router_prompt, context, response = extract_sft_examples(examples, 0)[-1]
        self.assertEqual(router_prompt, "Summarize\nA long article")
        self.assertEqual(context, [{"role": "user", "content": router_prompt}])
        self.assertEqual(response, "A summary")

    def test_bos_is_added_once_and_preserved_when_truncating(self):
        self.assertEqual(ensure_bos([10, 11], 2), [2, 10, 11])
        self.assertEqual(ensure_bos([2, 10, 11], 2), [2, 10, 11])
        self.assertEqual(truncate_prompt([2, 10, 11, 12, 13], 3, 2), [2, 12, 13])


class FakeTokenizer:
    bos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def __call__(self, text, add_special_tokens=False):
        if text == "PROMPT":
            return {"input_ids": [10, 11]}
        return {"input_ids": list(range(20, 20 + len(text)))}


class SequenceBoundaryTests(unittest.TestCase):
    def test_complete_response_has_one_eot_and_one_bos(self):
        sequence = build_sft_sequence(
            FakeTokenizer(),
            [{"role": "user", "content": "question"}],
            "ok",
            max_seq_length=32,
            max_response_length=8,
            stop_id=106,
            chat_template=DEFAULT_CHAT_TEMPLATE,
        )
        self.assertEqual(sequence["input_ids"][0], 2)
        self.assertEqual(sequence["labels"][-1], 106)
        self.assertEqual(sequence["labels"].count(106), 1)
        self.assertFalse(sequence["response_truncated"])

    def test_truncated_response_does_not_get_fake_eot(self):
        sequence = build_sft_sequence(
            FakeTokenizer(),
            [{"role": "user", "content": "question"}],
            "a very long response",
            max_seq_length=32,
            max_response_length=4,
            stop_id=106,
            chat_template=DEFAULT_CHAT_TEMPLATE,
        )
        self.assertTrue(sequence["response_truncated"])
        self.assertNotIn(106, sequence["labels"])


if __name__ == "__main__":
    unittest.main()
