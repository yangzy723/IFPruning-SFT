"""Dataset normalization and response-only SFT sequence construction."""


_BASE_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{'<|turn>' + ('model' if m['role'] == 'assistant' else m['role']) + '\\n' "
    "+ m['content'] + '<turn|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|turn>model\\n'}}{% endif %}"
)

CHAT_TEMPLATE_FORMAT = "gemma4_text_no_thinking"
DEFAULT_CHAT_TEMPLATE = _BASE_CHAT_TEMPLATE.replace(
    r"<|turn>model\n",
    r"<|turn>model\n<|channel>thought\n<channel|>",
)


def _normalize_hermes(raw_messages: list[dict]) -> list[dict[str, str]]:
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    return [
        {"role": role_map[message["from"]], "content": message["value"].strip()}
        for message in raw_messages
        if message.get("from") in role_map and str(message.get("value", "")).strip()
    ]


def extract_sft_examples(
    examples: dict[str, list], index: int
) -> list[tuple[str, list[dict[str, str]], str]]:
    """Return every valid assistant turn with its matching conversational context.

    Following the IFP paper, multi-turn conversations use the first human message
    for sub-network selection. Each assistant turn is nevertheless retained as an
    independent response target so earlier turns are not silently discarded.
    """
    if "conversations" in examples:
        messages = _normalize_hermes(examples["conversations"][index])
        first_user = next(
            (message["content"] for message in messages if message["role"] == "user"),
            "",
        )
        if not first_user:
            return []
        targets = []
        for target_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            context = messages[:target_index]
            if any(item["role"] == "user" for item in context):
                targets.append((first_user, context, message["content"]))
        return targets

    if "instruction" in examples and "output" in examples:
        instruction = str(examples["instruction"][index]).strip()
        extra_input = str(examples["input"][index]).strip()
        prompt = f"{instruction}\n{extra_input}".strip() if extra_input else instruction
        response = str(examples["output"][index]).strip()
        return [(prompt, [{"role": "user", "content": prompt}], response)]

    raise ValueError(f"Unsupported dataset columns: {sorted(examples)}")


def ensure_bos(input_ids: list[int], bos_token_id: int | None) -> list[int]:
    if bos_token_id is not None and (not input_ids or input_ids[0] != bos_token_id):
        return [int(bos_token_id)] + input_ids
    return input_ids


def truncate_prompt(input_ids: list[int], max_length: int, bos_token_id: int | None) -> list[int]:
    if len(input_ids) <= max_length:
        return input_ids
    if max_length <= 0:
        return []
    if bos_token_id is not None and input_ids and input_ids[0] == bos_token_id:
        if max_length == 1:
            return [int(bos_token_id)]
        return [int(bos_token_id)] + input_ids[-(max_length - 1) :]
    return input_ids[-max_length:]


def build_sft_sequence(
    tokenizer,
    context_messages: list[dict[str, str]],
    response: str,
    *,
    max_seq_length: int,
    max_response_length: int,
    stop_id: int,
    chat_template: str,
) -> dict:
    """Build one causal sequence with response-only labels and a valid boundary."""
    if max_response_length < 1 or max_seq_length < 2:
        raise ValueError("max_response_length must be >= 1 and max_seq_length must be >= 2")

    prompt_text = tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_ids = ensure_bos(prompt_ids, tokenizer.bos_token_id)

    # A truncated answer is a valid prefix target, but adding EOT at its cut
    # point teaches the model to stop halfway through code or reasoning.
    full_response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    response_truncated = len(full_response_ids) >= max_response_length
    if response_truncated:
        response_ids = full_response_ids[:max_response_length]
    else:
        response_ids = full_response_ids + [int(stop_id)]
    if len(response_ids) >= max_seq_length:
        response_ids = response_ids[: max_seq_length - 1]
        response_truncated = True

    prompt_ids = truncate_prompt(
        prompt_ids,
        max_seq_length - len(response_ids),
        tokenizer.bos_token_id,
    )
    if not prompt_ids or not response_ids:
        raise ValueError("Tokenized sample has an empty prompt or response")

    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
        "num_target_tokens": len(response_ids),
        "response_truncated": response_truncated,
    }
