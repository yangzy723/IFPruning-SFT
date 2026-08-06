#!/usr/bin/env python3
"""Validated IFPruning inference with shared train/inference components."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ifpruning_core import DynamicMaskedFFN, SparsityPredictor, find_transformer_layers
from ifpruning_data import CHAT_TEMPLATE_FORMAT, ensure_bos


LOGGER = logging.getLogger("ifpruning_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./gemma-12B-ifpruning-output")
    parser.add_argument(
        "--predictor-model",
        default=None,
        help="Override the predictor path stored in the checkpoint manifest",
    )
    parser.add_argument(
        "--prompt", default=None, help="Run one prompt instead of the interactive shell"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-predictor-length", type=int, default=1024)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--score-dir", default="./routing_scores")
    parser.add_argument(
        "--dense", action="store_true", help="Disable masks for a dense checkpoint baseline"
    )
    return parser.parse_args()


def read_manifest(checkpoint: Path) -> dict:
    path = checkpoint / "ifpruning_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing IFPruning manifest: {path}")
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    required_fields = (
        "chat_template_format",
        "num_layers",
        "full_intermediate_dim",
        "target_intermediate_dim",
        "predictor_model",
        "predictor_hidden_dim",
        "mask_temperature",
        "softtopk_iters",
    )
    missing = [field for field in required_fields if field not in manifest]
    if missing:
        raise ValueError(f"Checkpoint manifest is incomplete: missing {missing} in {path}")
    if manifest["chat_template_format"] != CHAT_TEMPLATE_FORMAT:
        raise ValueError("Checkpoint uses an unsupported chat-template format")
    return manifest


def predictor_payload(checkpoint: Path) -> Path:
    filename = "predictor.safetensors"
    path = checkpoint / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing predictor state: {path}")
    return path


def resolve_predictor_config(manifest: dict) -> dict:
    """Load predictor settings from the current checkpoint manifest."""
    return {
        "hidden_dim": int(manifest["predictor_hidden_dim"]),
        "mask_temperature": float(manifest["mask_temperature"]),
        "softtopk_iters": int(manifest["softtopk_iters"]),
    }


class IFPruningInferenceModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        predictor_path: str,
        target_dim: int,
        predictor_config: dict,
    ):
        super().__init__()
        self.base_model = base_model
        text_config = base_model.config.text_config
        self.num_layers = int(text_config.num_hidden_layers)
        self.ffn_dim = int(text_config.intermediate_size)
        if not 0 < target_dim <= self.ffn_dim:
            raise ValueError(f"Invalid target dimension {target_dim} for FFN size {self.ffn_dim}")

        self.predictor = SparsityPredictor(
            num_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            extractor_path=predictor_path,
            local_files_only=True,
            hidden_dim=predictor_config["hidden_dim"],
            extractor_dtype=next(base_model.parameters()).dtype,
        )

        self.layers = find_transformer_layers(base_model)
        if len(self.layers) != self.num_layers:
            raise ValueError(
                f"Config declares {self.num_layers} layers, found {len(self.layers)} MLP blocks"
            )
        for layer in self.layers:
            layer.mlp = DynamicMaskedFFN(
                layer.mlp,
                target_dim=target_dim,
                mask_temperature=predictor_config["mask_temperature"],
                softtopk_iters=predictor_config["softtopk_iters"],
            )
            layer.mlp.mask_alpha.fill_(1.0)

        predictor_device = next(base_model.parameters()).device
        self.predictor.to(device=predictor_device)

    @torch.inference_mode()
    def compute_and_lock_mask(
        self,
        predictor_input_ids: torch.Tensor,
        predictor_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        predictor_device = next(self.predictor.parameters()).device
        scores = self.predictor(
            predictor_input_ids.to(predictor_device),
            predictor_attention_mask.to(predictor_device),
        )
        for index, layer in enumerate(self.layers):
            layer_device = layer.mlp.gate_proj.weight.device
            layer.mlp.layer_scores = scores[:, index, :].to(layer_device)
        return scores.float().cpu()


def load_model(args: argparse.Namespace):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    manifest = read_manifest(checkpoint)
    payload_path = predictor_payload(checkpoint)
    predictor_state = load_file(str(payload_path))
    predictor_config = resolve_predictor_config(manifest)

    predictor_source = args.predictor_model or manifest["predictor_model"]

    base_tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if not base_tokenizer.chat_template:
        raise ValueError("Checkpoint tokenizer is missing its chat template")
    if base_tokenizer.pad_token_id is None or base_tokenizer.eos_token_id is None:
        raise ValueError("Checkpoint tokenizer must define PAD and EOS tokens")

    predictor_tokenizer_path = checkpoint / "predictor_tokenizer"
    if not predictor_tokenizer_path.is_dir():
        raise FileNotFoundError(f"Missing predictor tokenizer: {predictor_tokenizer_path}")
    predictor_tokenizer = AutoTokenizer.from_pretrained(
        str(predictor_tokenizer_path), local_files_only=True
    )
    dtype = (
        "auto"
        if args.dtype == "auto"
        else {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )

    model = IFPruningInferenceModel(
        base_model=base_model,
        predictor_path=str(predictor_source),
        target_dim=int(manifest["target_intermediate_dim"]),
        predictor_config=predictor_config,
    )
    if int(manifest["num_layers"]) != model.num_layers:
        raise ValueError("Checkpoint manifest layer count does not match the base model")
    if int(manifest["full_intermediate_dim"]) != model.ffn_dim:
        raise ValueError("Checkpoint manifest FFN dimension does not match the base model")

    model.predictor.load_state_dict(predictor_state, strict=True)
    model.eval()
    return model, base_tokenizer, predictor_tokenizer, manifest


def tokenize_base_prompt(
    tokenizer, instruction: str, device: torch.device
) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": instruction}]
    chat_template = tokenizer.chat_template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    ids = ensure_bos(encoded["input_ids"][0].tolist(), tokenizer.bos_token_id)
    encoded["input_ids"] = torch.tensor([ids], dtype=torch.long)
    encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return {key: value.to(device) for key, value in encoded.items()}


def stopping_ids(tokenizer) -> list[int]:
    if "<turn|>" not in tokenizer.get_vocab():
        raise ValueError("Checkpoint tokenizer is missing the <turn|> token")
    return [
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<turn|>")),
    ]


def safe_score_name(index: int, prompt: str) -> str:
    stem = "".join(
        character
        for character in prompt[:30]
        if character.isalnum() or character.isspace()
    )
    stem = stem.strip().replace(" ", "_") or "prompt"
    return f"score_{index:02d}_{stem}.pt"


def run_prompt(
    model: IFPruningInferenceModel,
    base_tokenizer,
    predictor_tokenizer,
    instruction: str,
    args: argparse.Namespace,
    prompt_index: int,
    previous_scores: torch.Tensor | None,
) -> torch.Tensor:
    input_device = next(model.base_model.parameters()).device
    base_inputs = tokenize_base_prompt(base_tokenizer, instruction, input_device)
    predictor_inputs = predictor_tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_predictor_length,
    )

    if args.dense:
        scores = torch.empty(0)
    else:
        scores = model.compute_and_lock_mask(
            predictor_inputs["input_ids"],
            predictor_inputs["attention_mask"],
        )
        score_path = Path(args.score_dir) / safe_score_name(prompt_index, instruction)
        torch.save({"prompt": instruction, "scores": scores.squeeze(0)}, score_path)
        LOGGER.info("Routing scores saved to %s", score_path)

        if previous_scores is not None and previous_scores.shape == scores.shape:
            target_dim = model.layers[0].mlp.target_ffn_dim
            previous_top = torch.topk(previous_scores, target_dim, dim=-1).indices
            current_top = torch.topk(scores, target_dim, dim=-1).indices
            previous_mask = torch.zeros_like(previous_scores, dtype=torch.bool).scatter_(
                -1, previous_top, True
            )
            current_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                -1, current_top, True
            )
            overlap = (previous_mask & current_mask).sum(dim=-1).float() / target_dim
            LOGGER.info(
                "Cross-prompt TopK overlap: mean=%.4f max=%.4f",
                overlap.mean(),
                overlap.max(),
            )
            if torch.all(overlap > 0.999):
                LOGGER.warning(
                    "All layer masks are identical across different prompts; routing is static"
                )

    stop_tokens = stopping_ids(base_tokenizer)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": base_tokenizer.pad_token_id,
        "eos_token_id": stop_tokens,
    }
    if args.do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        output = model.base_model.generate(**base_inputs, **generation_kwargs)
    input_length = base_inputs["input_ids"].shape[1]
    response_ids = output[0, input_length:]
    final_token_id = int(response_ids[-1]) if response_ids.numel() else None
    LOGGER.info(
        "Generated %d tokens; final_token_id=%s; stopped_on_boundary=%s",
        response_ids.numel(),
        final_token_id,
        final_token_id in stop_tokens if final_token_id is not None else False,
    )
    response = base_tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    print(f"[IFP Model] {response}")
    return scores


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    Path(args.score_dir).mkdir(parents=True, exist_ok=True)

    model, base_tokenizer, predictor_tokenizer, manifest = load_model(args)
    LOGGER.info(
        "Loaded target=%s/%s channels (%.2f%% retained)",
        manifest["target_intermediate_dim"],
        model.ffn_dim,
        100.0 * int(manifest["target_intermediate_dim"]) / model.ffn_dim,
    )

    if args.prompt is not None:
        run_prompt(model, base_tokenizer, predictor_tokenizer, args.prompt, args, 1, None)
        return

    print("IFPruning interactive shell. Type 'quit' or 'exit' to stop.")
    previous_scores = None
    prompt_index = 1
    while True:
        try:
            instruction = input("\n[User] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not instruction:
            continue
        if instruction.lower() in {"quit", "exit"}:
            break
        previous_scores = run_prompt(
            model,
            base_tokenizer,
            predictor_tokenizer,
            instruction,
            args,
            prompt_index,
            previous_scores,
        )
        prompt_index += 1


if __name__ == "__main__":
    main()
