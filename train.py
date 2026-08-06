#!/usr/bin/env python3
"""IFPruning supervised fine-tuning entry point."""

import os
import argparse
import json
import logging
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset, concatenate_datasets
from safetensors.torch import save_file, load_file
from deepspeed.ops.adam import FusedAdam
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from ifpruning_core import DynamicMaskedFFN, SparsityPredictor, find_transformer_layers
from ifpruning_data import (
    CHAT_TEMPLATE_FORMAT,
    DEFAULT_CHAT_TEMPLATE,
    build_sft_sequence,
    extract_sft_examples,
)

LOGGER = logging.getLogger("ifpruning_sft")

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", LOCAL_RANK))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_RANK0 = RANK == 0


class RankFilter(logging.Filter):
    """Filters log records to inject distributed rank information."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = RANK
        record.local_rank = LOCAL_RANK
        record.world_size = WORLD_SIZE
        return True


def setup_logging(output_dir: str, log_level: str = "INFO") -> tuple[logging.Logger, Path]:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("ifpruning_sft")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [rank=%(rank)s/%(world_size)s pid=%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_dir / f"rank_{RANK}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    if IS_RANK0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
    return logger, log_dir


def rank0_json_dump(path: Path, obj: object) -> None:
    if not IS_RANK0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(path)

@dataclass
class RunConfig:
    base_model: str = "./gemma-4-12B"
    predictor_model: str = "./Qwen3.5-0.8B"
    output_dir: str = "./gemma-12B-ifpruning-output"
    overwrite_output_dir: bool = False

    dataset_alpaca: str = "./alpaca-cleaned/alpaca_data_cleaned.json"
    dataset_hermes: str = "./OpenHermes-2.5/openhermes2_5.json"
    hermes_sample_size: int = 100000
    cache_dir: str = "./hf_cache"

    local_files_only: bool = True

    target_intermediate_dim: int = 4096
    max_seq_length: int = 2048
    max_response_length: int = 512
    max_predictor_length: int = 1024

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1
    max_steps: int = -1
    base_lr: float = 2e-6
    predictor_lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: float = 0.03
    max_grad_norm: float = 1.0

    mask_warmup_steps: int = 2000
    mask_temperature: float = 1.0
    softtopk_iters: int = 32
    abort_on_zero_loss_steps: int = 5
    abort_on_routing_collapse_logs: int = 10
    min_routing_input_std: float = 1e-7

    predictor_hidden_dim: int = 128
    predictor_output_init_std: float = 1e-4

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    zero_stage: int = 3

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 1
    eval_steps: int = 500
    validation_samples: int = 512
    dataloader_num_workers: int = 2
    preprocessing_num_proc: int = 8
    preprocessing_batch_size: int = 1000
    seed: int = 42
    report_to: str = "none"
    resume: str = "none"


def parse_args() -> RunConfig:
    """Parses command-line arguments mapped to the RunConfig dataclass."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name, default in asdict(RunConfig()).items():
        if isinstance(default, bool):
            parser.add_argument(
                f"--{name}", action=argparse.BooleanOptionalAction, default=default
            )
        else:
            parser.add_argument(f"--{name}", type=type(default), default=default)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="_launcher_local_rank",
        type=int,
        default=LOCAL_RANK,
        help=argparse.SUPPRESS,
    )
    parsed = vars(parser.parse_args())
    parsed.pop("_launcher_local_rank", None)
    return RunConfig(**parsed)


def make_deepspeed_config(cfg: RunConfig, log_dir: Path) -> str:
    """Write the ZeRO-3 configuration used by the current training pipeline."""

    ds = {
        "bf16": {"enabled": bool(cfg.bf16)},
        "fp16": {"enabled": bool(cfg.fp16)},
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 200_000_000,
            "reduce_scatter": True,
            "reduce_bucket_size": 200_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
            "stage3_prefetch_bucket_size": 200_000_000,
            "stage3_param_persistence_threshold": 100_000,
            "stage3_max_live_parameters": 500_000_000,
            "stage3_max_reuse_distance": 500_000_000,
            "sub_group_size": 500_000_000,
        },
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
    }

    path = log_dir / f"ds_config.rank{RANK}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(ds, file, ensure_ascii=False, indent=2, sort_keys=True)
    return str(path)


def patch_model_for_ifpruning(
    base_model: nn.Module, cfg: RunConfig
) -> tuple[nn.Module, nn.ModuleList]:
    """Modifies the base model architecture in-place to support IFPruning."""
    llm_cfg = base_model.config.text_config
    num_layers = int(llm_cfg.num_hidden_layers)
    ffn_dim = int(llm_cfg.intermediate_size)
    layers = find_transformer_layers(base_model)
    if len(layers) != num_layers:
        raise ValueError(f"Config declares {num_layers} layers but found {len(layers)} MLP blocks")

    for layer in layers:
        layer.mlp = DynamicMaskedFFN(
            layer.mlp,
            target_dim=cfg.target_intermediate_dim,
            mask_temperature=cfg.mask_temperature,
            softtopk_iters=cfg.softtopk_iters,
        )

    base_model.predictor = SparsityPredictor(
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        extractor_path=cfg.predictor_model,
        local_files_only=cfg.local_files_only,
        cache_dir=cfg.cache_dir,
        hidden_dim=cfg.predictor_hidden_dim,
        output_init_std=cfg.predictor_output_init_std,
        extractor_dtype=(
            torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
        ),
    )
    if cfg.gradient_checkpointing:
        base_model.predictor.extractor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    orig_forward = base_model.forward

    def ifp_forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        predictor_input_ids=None,
        predictor_attention_mask=None,
        **kwargs
    ):
        kwargs.pop("use_cache", None)

        p_ids = predictor_input_ids if predictor_input_ids is not None else input_ids
        p_mask = (
            predictor_attention_mask
            if predictor_attention_mask is not None
            else attention_mask
        )

        scores = self.predictor(p_ids, p_mask)
        for i, layer in enumerate(layers):
            layer.mlp.layer_scores = scores[:, i, :]

        return orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            **kwargs
        )

    def set_mask_alpha(self, alpha: float):
        for layer in layers:
            layer.mlp.mask_alpha.fill_(float(max(0.0, min(1.0, alpha))))

    orig_save = base_model.save_pretrained

    def ifp_save_pretrained(self, save_directory: str, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()

        if not state_dict:
            if IS_RANK0:
                raise RuntimeError(
                    "Model state is empty; predictor export cannot be made recoverable"
                )
            return

        base_state = {k: v for k, v in state_dict.items() if not k.startswith("predictor.")}
        orig_save(save_directory, state_dict=base_state, **kwargs)

        if IS_RANK0:
            prefix = "predictor."
            pred_state = {}
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    clean_key = key[len(prefix):]
                    pred_state[clean_key] = value.detach().cpu().contiguous()
            if not pred_state:
                raise RuntimeError("Predictor state was empty during checkpoint export")
            filename = "predictor.safetensors"
            save_file(pred_state, str(Path(save_directory) / filename))
            rank0_json_dump(
                Path(save_directory) / "ifpruning_config.json",
                {
                    "chat_template_format": CHAT_TEMPLATE_FORMAT,
                    "zero_stage": cfg.zero_stage,
                    "num_layers": num_layers,
                    "full_intermediate_dim": ffn_dim,
                    "target_intermediate_dim": cfg.target_intermediate_dim,
                    "predictor_model": cfg.predictor_model,
                    "predictor_hidden_dim": cfg.predictor_hidden_dim,
                    "mask_temperature": cfg.mask_temperature,
                    "softtopk_iters": cfg.softtopk_iters,
                },
            )

    base_model.forward = types.MethodType(ifp_forward, base_model)
    base_model.set_mask_alpha = types.MethodType(set_mask_alpha, base_model)
    base_model.save_pretrained = types.MethodType(ifp_save_pretrained, base_model)

    return base_model, layers


def tokenize_sft_dataset(
    raw: Dataset,
    base_tokenizer: PreTrainedTokenizerBase,
    predictor_tokenizer: PreTrainedTokenizerBase,
    cfg: RunConfig,
    training_args: TrainingArguments,
) -> Dataset:
    """Tokenize full conversational context and supervise only its matching response."""
    if cfg.max_response_length < 1 or cfg.max_seq_length < 2:
        raise ValueError("max_response_length must be >= 1 and max_seq_length must be >= 2")
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    if predictor_tokenizer.pad_token_id is None:
        predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    base_tokenizer.padding_side = "right"

    chat_template = base_tokenizer.chat_template or DEFAULT_CHAT_TEMPLATE
    base_tokenizer.chat_template = chat_template
    if "<turn|>" not in base_tokenizer.get_vocab():
        raise ValueError("The Gemma tokenizer is missing the <turn|> token")
    stop_id = base_tokenizer.convert_tokens_to_ids("<turn|>")

    def process_batch(examples):
        output = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "predictor_input_ids": [],
            "predictor_attention_mask": [],
            "num_target_tokens": [],
            "response_truncated": [],
        }
        router_prompts = []
        batch_size = len(next(iter(examples.values())))
        for index in range(batch_size):
            for router_prompt, context_messages, response in extract_sft_examples(
                examples, index
            ):
                if not router_prompt or not context_messages or not response:
                    continue
                sequence = build_sft_sequence(
                    base_tokenizer,
                    context_messages,
                    response,
                    max_seq_length=cfg.max_seq_length,
                    max_response_length=cfg.max_response_length,
                    stop_id=stop_id,
                    chat_template=chat_template,
                )
                for key in (
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "num_target_tokens",
                    "response_truncated",
                ):
                    output[key].append(sequence[key])
                router_prompts.append(router_prompt)

        if router_prompts:
            predictor_encoding = predictor_tokenizer(
                router_prompts,
                add_special_tokens=True,
                truncation=True,
                max_length=cfg.max_predictor_length,
            )
            output["predictor_input_ids"] = predictor_encoding["input_ids"]
            output["predictor_attention_mask"] = predictor_encoding["attention_mask"]
        return output

    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized = raw.map(
            process_batch,
            batched=True,
            batch_size=cfg.preprocessing_batch_size,
            num_proc=max(1, cfg.preprocessing_num_proc),
            remove_columns=raw.column_names,
        )
        return tokenized.filter(lambda example: example["num_target_tokens"] > 0)


class DualCollator:
    """Pad base-model and predictor inputs independently."""

    def __init__(self, base_pad_id: int, predictor_pad_id: int):
        self.base_pad_id = base_pad_id
        self.predictor_pad_id = predictor_pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        base_length = max(len(item["input_ids"]) for item in features)
        predictor_length = max(len(item["predictor_input_ids"]) for item in features)

        def pad(values, length, pad_id):
            return values + [pad_id] * (length - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(item["input_ids"], base_length, self.base_pad_id) for item in features],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(item["attention_mask"], base_length, 0) for item in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(item["labels"], base_length, -100) for item in features],
                dtype=torch.long,
            ),
            "predictor_input_ids": torch.tensor(
                [
                    pad(item["predictor_input_ids"], predictor_length, self.predictor_pad_id)
                    for item in features
                ],
                dtype=torch.long,
            ),
            "predictor_attention_mask": torch.tensor(
                [pad(item["predictor_attention_mask"], predictor_length, 0) for item in features],
                dtype=torch.long,
            ),
        }


class IFPruningTrainer(Trainer):
    """Custom Trainer implementing distinct learning rates for base and predictor networks."""
    def __init__(self, *args, predictor_lr: float, base_lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_lr = predictor_lr
        self.base_lr = base_lr

    def _is_no_decay(self, name: str, param: nn.Parameter) -> bool:
        shape = getattr(param, "ds_shape", param.shape)
        return len(shape) < 2 or any(k in name.lower() for k in ["bias", "norm", "ln"])

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        trainable_dtypes = {
            param.dtype for param in self.model.parameters() if param.requires_grad
        }
        if len(trainable_dtypes) != 1:
            details = ", ".join(
                f"{dtype}: {sum(
                    1
                    for param in self.model.parameters()
                    if param.requires_grad and param.dtype == dtype
                )}"
                for dtype in sorted(trainable_dtypes, key=str)
            )
            raise RuntimeError(
                "ZeRO-3 requires all trainable parameters to use one dtype; "
                f"found {details}"
            )

        groups = {}
        for name, param in sorted(self.model.named_parameters(), key=lambda item: item[0]):
            if not param.requires_grad:
                continue
            scope = "pred" if name.startswith("predictor.") else "base"
            use_decay = not self._is_no_decay(name, param)
            group_key = (scope, use_decay)
            groups.setdefault(group_key, []).append(param)

        optim_groups = []
        for (scope, use_decay), params in sorted(groups.items()):
            optim_groups.append(
                {
                    "params": params,
                    "lr": self.predictor_lr if scope == "pred" else self.base_lr,
                    "weight_decay": self.args.weight_decay if use_decay else 0.0,
                }
            )
            LOGGER.info(
                "Optimizer group: scope=%s decay=%s dtype=%s tensors=%d",
                scope,
                use_decay,
                params[0].dtype,
                len(params),
            )

        self.optimizer = FusedAdam(
            optim_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )

        return self.optimizer

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Persist routing-health metrics before Trainer writes log_history."""
        enriched = dict(logs)
        if "loss" in enriched:
            wrapped = getattr(self, "model_wrapped", None)
            if wrapped is None:
                wrapped = self.model
            root = wrapped.module if hasattr(wrapped, "module") else wrapped
            first_masked_ffn = next(
                module for module in root.modules() if isinstance(module, DynamicMaskedFFN)
            )
            enriched["mask_alpha"] = float(first_masked_ffn.mask_alpha.item())
            enriched["routing_head_active_fraction"] = float(
                root.predictor.last_head_active_fraction.item()
            )
            enriched["routing_input_std"] = float(
                root.predictor.last_score_input_std.item()
            )
        super().log(enriched, start_time=start_time)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Evaluate every batch with the deploy-time fully sparse mask."""
        root = model.module if hasattr(model, "module") else model
        first_masked_ffn = next(
            module for module in root.modules() if isinstance(module, DynamicMaskedFFN)
        )
        previous_alpha = float(first_masked_ffn.mask_alpha.item())
        root.set_mask_alpha(1.0)
        try:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
        finally:
            root.set_mask_alpha(previous_alpha)


class IFPruningCallback(TrainerCallback):
    """Callback evaluating and managing the dynamic mask alpha schedule."""
    def __init__(
        self,
        warmup: int,
        abort_zero: int,
        abort_collapse_logs: int = 10,
        min_routing_input_std: float = 1e-7,
    ):
        self.warmup = warmup
        self.abort = abort_zero
        self.abort_collapse_logs = abort_collapse_logs
        self.min_routing_input_std = min_routing_input_std
        self.zero_streak = 0
        self.collapse_streak = 0

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        alpha = 1.0 if self.warmup <= 0 else min(1.0, state.global_step / self.warmup)
        root = model.module if hasattr(model, "module") else model
        root.set_mask_alpha(alpha)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        loss = float(logs["loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {state.global_step}: {loss}"
            )
        if loss <= 1e-8 and state.global_step > 1:
            self.zero_streak += 1
            if self.abort > 0 and self.zero_streak >= self.abort:
                raise RuntimeError(
                    f"Training loss stayed near zero for {self.zero_streak} log events"
                )
        else:
            self.zero_streak = 0

        input_std = float(logs.get("routing_input_std", float("nan")))
        active_fraction = float(
            logs.get("routing_head_active_fraction", float("nan"))
        )
        if math.isfinite(input_std) and input_std < self.min_routing_input_std:
            self.collapse_streak += 1
            LOGGER.warning(
                "Routing may be collapsing: input-conditioned score std=%.3e",
                input_std,
            )
            if (
                self.abort_collapse_logs > 0
                and self.collapse_streak >= self.abort_collapse_logs
            ):
                raise RuntimeError(
                    "Routing remained input-independent for "
                    f"{self.collapse_streak} consecutive log events"
                )
        else:
            self.collapse_streak = 0

        LOGGER.info(
            "Step %d | Loss=%.4f | LR=%.3e | Alpha=%.4f | "
            "RoutingStd=%.3e | HeadActive=%.4f",
            state.global_step,
            loss,
            float(logs.get("learning_rate", 0.0)),
            float(logs.get("mask_alpha", 0.0)),
            input_std,
            active_fraction,
        )


def parameter_numel(parameter: nn.Parameter) -> int:
    return int(getattr(parameter, "ds_numel", None) or parameter.numel())


def resolve_resume_checkpoint(cfg: RunConfig) -> str | None:
    if cfg.resume == "none":
        return None
    checkpoint = (
        get_last_checkpoint(cfg.output_dir) if cfg.resume == "auto" else cfg.resume
    )
    if not checkpoint or not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    return str(checkpoint)


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    cfg = parse_args()
    resume_checkpoint = resolve_resume_checkpoint(cfg)
    output_path = Path(cfg.output_dir)
    if (
        resume_checkpoint is None
        and output_path.exists()
        and any(entry.name != "logs" for entry in output_path.iterdir())
        and not cfg.overwrite_output_dir
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Choose a new --output_dir, resume explicitly, or pass --overwrite_output_dir."
        )

    global LOGGER
    LOGGER, log_dir = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)

    rank0_json_dump(log_dir / "run_config.json", asdict(cfg))
    if cfg.bf16 and cfg.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    if cfg.zero_stage != 3:
        raise ValueError("The current training pipeline requires --zero_stage 3")
    if cfg.base_lr <= 0 or cfg.predictor_lr <= 0:
        raise ValueError("base_lr and predictor_lr must be positive")
    if cfg.mask_temperature <= 0 or cfg.softtopk_iters < 1:
        raise ValueError("mask_temperature and softtopk_iters must be positive")
    if cfg.max_predictor_length < 1:
        raise ValueError("max_predictor_length must be positive")
    LOGGER.warning(
        "This entry point implements the SFT stage only. A fresh predictor has not received "
        "the continued-pretraining stage used by the paper; do not compare it directly with "
        "the paper result."
    )

    ta_kwargs = {
        "output_dir": cfg.output_dir,
        "do_train": True,
        "do_eval": cfg.validation_samples > 0,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.base_lr,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "warmup_steps": cfg.warmup_steps,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "eval_steps": cfg.eval_steps,
        "eval_strategy": "steps" if cfg.validation_samples > 0 else "no",
        "prediction_loss_only": True,
        "report_to": [] if cfg.report_to == "none" else cfg.report_to.split(","),
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "deepspeed": make_deepspeed_config(cfg, log_dir),
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if cfg.gradient_checkpointing else None
        ),
    }

    training_args = TrainingArguments(**ta_kwargs)

    LOGGER.info("Initializing tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        local_files_only=cfg.local_files_only,
        use_fast=True,
    )
    predictor_tokenizer = AutoTokenizer.from_pretrained(
        cfg.predictor_model,
        local_files_only=cfg.local_files_only,
        use_fast=True,
    )

    LOGGER.info("Loading local datasets...")
    alpaca_raw = load_dataset(
        "json",
        data_files=cfg.dataset_alpaca,
        split="train",
        cache_dir=cfg.cache_dir,
    )
    hermes_raw = load_dataset(
        "json",
        data_files=cfg.dataset_hermes,
        split="train",
        cache_dir=cfg.cache_dir,
    )

    hermes_sample_size = min(cfg.hermes_sample_size, len(hermes_raw))
    hermes_raw = hermes_raw.shuffle(seed=cfg.seed).select(range(hermes_sample_size))

    LOGGER.info("Tokenizing datasets...")
    alpaca_tok = tokenize_sft_dataset(
        alpaca_raw, base_tokenizer, predictor_tokenizer, cfg, training_args
    )
    hermes_tok = tokenize_sft_dataset(
        hermes_raw, base_tokenizer, predictor_tokenizer, cfg, training_args
    )

    tokenized_dataset = concatenate_datasets([alpaca_tok, hermes_tok]).shuffle(
        seed=cfg.seed
    )
    truncated_count = sum(bool(value) for value in tokenized_dataset["response_truncated"])
    validation_count = min(
        max(0, cfg.validation_samples), max(0, len(tokenized_dataset) - 1)
    )
    if validation_count:
        eval_dataset = tokenized_dataset.select(range(validation_count))
        train_dataset = tokenized_dataset.select(
            range(validation_count, len(tokenized_dataset))
        )
    else:
        eval_dataset = None
        train_dataset = tokenized_dataset
    LOGGER.info(
        "Dataset preparation complete. train=%d validation=%d truncated_responses=%d (%.2f%%)",
        len(train_dataset),
        validation_count,
        truncated_count,
        100.0 * truncated_count / max(1, len(tokenized_dataset)),
    )

    LOGGER.info("Loading and patching the base model...")
    model_kwargs = {
        "dtype": (
            torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
        ),
        "local_files_only": cfg.local_files_only,
        "attn_implementation": cfg.attn_implementation,
    }
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    model, layers = patch_model_for_ifpruning(base_model, cfg)
    base_parameter_count = sum(
        parameter_numel(parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("predictor.")
    )
    ffn_parameter_count = sum(
        parameter_numel(parameter)
        for layer in layers
        for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
        for parameter in projection.parameters()
    )
    retention = cfg.target_intermediate_dim / layers[0].mlp.full_ffn_dim
    theoretical_active = (
        base_parameter_count - ffn_parameter_count + ffn_parameter_count * retention
    )
    predictor_parameter_count = sum(
        parameter_numel(parameter) for parameter in model.predictor.parameters()
    )
    LOGGER.info(
        "Parameters: base=%.3fB, theoretical active base=%.3fB, "
        "predictor=%.3fB, FFN retention=%.2f%%",
        base_parameter_count / 1e9,
        theoretical_active / 1e9,
        predictor_parameter_count / 1e9,
        retention * 100.0,
    )

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer = IFPruningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DualCollator(
            base_tokenizer.pad_token_id, predictor_tokenizer.pad_token_id
        ),
        processing_class=base_tokenizer,
        callbacks=[
            IFPruningCallback(
                cfg.mask_warmup_steps,
                cfg.abort_on_zero_loss_steps,
                cfg.abort_on_routing_collapse_logs,
                cfg.min_routing_input_std,
            )
        ],
        predictor_lr=cfg.predictor_lr,
        base_lr=cfg.base_lr,
    )

    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        manifest_path = checkpoint_path / "ifpruning_config.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing IFPruning manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

        expected_manifest = {
            "chat_template_format": CHAT_TEMPLATE_FORMAT,
            "zero_stage": cfg.zero_stage,
            "num_layers": len(layers),
            "full_intermediate_dim": layers[0].mlp.full_ffn_dim,
            "target_intermediate_dim": cfg.target_intermediate_dim,
            "predictor_model": cfg.predictor_model,
            "predictor_hidden_dim": cfg.predictor_hidden_dim,
            "mask_temperature": cfg.mask_temperature,
            "softtopk_iters": cfg.softtopk_iters,
        }
        mismatches = {
            key: (manifest[key], expected)
            for key, expected in expected_manifest.items()
            if manifest[key] != expected
        }
        if mismatches:
            raise ValueError(
                "Checkpoint architecture/config does not match this run: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )

        predictor_filename = "predictor.safetensors"
        predictor_path = checkpoint_path / predictor_filename
        if not predictor_path.exists():
            raise FileNotFoundError(f"Missing predictor state: {predictor_path}")
        LOGGER.info("Restoring predictor from %s", predictor_path)
        model.predictor.load_state_dict(load_file(str(predictor_path)), strict=True)

    if resume_checkpoint:
        LOGGER.info("Resuming from %s", resume_checkpoint)
    else:
        LOGGER.info("Starting a fresh training run")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model(cfg.output_dir)
    trainer.save_state()
    if IS_RANK0:
        base_tokenizer.save_pretrained(cfg.output_dir)
        predictor_tokenizer.save_pretrained(
            str(Path(cfg.output_dir) / "predictor_tokenizer")
        )

    LOGGER.info("Training cycle complete.")

if __name__ == "__main__":
    main()