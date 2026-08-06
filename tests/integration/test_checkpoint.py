#!/usr/bin/env python3
"""Two-phase DeepSpeed checkpoint save and restore test.

Usage:
  TEST_PHASE=1 torchrun --nproc_per_node=2 tests/integration/test_checkpoint.py
  TEST_PHASE=2 torchrun --nproc_per_node=2 tests/integration/test_checkpoint.py
"""

import logging
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import (  # noqa: E402
    DualCollator,
    IFPruningCallback,
    IFPruningTrainer,
    RunConfig,
    make_deepspeed_config,
    patch_model_for_ifpruning,
)


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
LOGGER = logging.getLogger("checkpoint_test")


def create_dummy_dataset(num_samples: int = 16):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]] * num_samples,
            "attention_mask": [[1, 1, 1]] * num_samples,
            "labels": [[-100, 2, 3]] * num_samples,
            "predictor_input_ids": [[1, 2, 3]] * num_samples,
            "predictor_attention_mask": [[1, 1, 1]] * num_samples,
            "num_target_tokens": [2] * num_samples,
        }
    )


class AuditLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if RANK == 0 and logs and "loss" in logs:
            LOGGER.info(
                "step=%d loss=%.6f lr=%.2e alpha=%.4f",
                state.global_step,
                logs["loss"],
                logs.get("learning_rate", 0.0),
                logs.get("mask_alpha", 0.0),
            )


def build_trainer(cfg: RunConfig) -> IFPruningTrainer:
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=cfg.max_steps,
        learning_rate=cfg.base_lr,
        bf16=cfg.bf16,
        save_steps=cfg.save_steps,
        report_to=[],
        logging_steps=1,
        deepspeed=make_deepspeed_config(cfg, log_dir),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        local_files_only=cfg.local_files_only,
        attn_implementation=cfg.attn_implementation,
    )
    model, layers = patch_model_for_ifpruning(base_model, cfg)
    return IFPruningTrainer(
        model=model,
        args=training_args,
        train_dataset=create_dummy_dataset(),
        data_collator=DualCollator(base_pad_id=0, predictor_pad_id=0),
        predictor_lr=cfg.predictor_lr,
        base_lr=cfg.base_lr,
        callbacks=[
            IFPruningCallback(
                cfg.mask_warmup_steps,
                cfg.abort_on_zero_loss_steps,
                cfg.abort_on_routing_collapse_logs,
                cfg.min_routing_input_std,
            ),
            AuditLogCallback(),
        ],
    )


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    phase = int(os.environ.get("TEST_PHASE", "0"))
    if phase not in {1, 2}:
        raise SystemExit("Set TEST_PHASE=1 to save or TEST_PHASE=2 to restore")

    cfg = RunConfig(
        max_steps=5,
        save_steps=5,
        logging_steps=1,
        mask_warmup_steps=10,
        output_dir="./gemma-12B-ifpruning-test-ckpt",
    )
    trainer = build_trainer(cfg)

    if phase == 1:
        LOGGER.info("Saving checkpoint after %d steps", cfg.max_steps)
        trainer.train()
        return

    checkpoint = get_last_checkpoint(cfg.output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {cfg.output_dir}")
    predictor_path = Path(checkpoint) / "predictor.safetensors"
    if not predictor_path.exists():
        raise FileNotFoundError(f"Missing predictor payload: {predictor_path}")
    trainer.model.predictor.load_state_dict(load_file(str(predictor_path)), strict=True)
    trainer.args.max_steps += 5
    LOGGER.info("Restoring checkpoint %s", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
