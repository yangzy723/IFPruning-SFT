#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Checkpoint Integrity Validator
-------------------------------------
Validates that DeepSpeed ZeRO states and decoupled IFPruning predictor weights 
can be successfully restored without partition mismatch or metadata corruption.

Usage:
  Phase 1 (Checkpoint): TEST_PHASE=1 torchrun --nproc_per_node=2 test_ckpt.py
  Phase 2 (Restore):  TEST_PHASE=2 torchrun --nproc_per_node=2 test_ckpt.py
"""

import os
import sys
import logging
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, TrainingArguments, TrainerCallback
from deepspeed.ops.adam import FusedAdam

# Import core architecture components directly from train.py to ensure parity
from train import (
    RunConfig,
    SparsityPredictor,
    patch_model_for_ifpruning,
    IFPruningTrainer,
    IFPruningCallback,
    DualCollator,
    make_deepspeed_config
)

# ==============================================================================
# Configuration & Environment
# ==============================================================================
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
IS_RANK0 = (RANK == 0)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [rank=%(process)d] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ckpt_validator")

# ==============================================================================
# Validation Logic
# ==============================================================================
def create_dummy_dataset(num_samples: int = 16):
    """Creates a minimal viable dataset to trigger the Trainer's initialization sequence."""
    from datasets import Dataset
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3]] * num_samples,
        "attention_mask": [[1, 1, 1]] * num_samples,
        "labels": [[-100, 2, 3]] * num_samples,
        "predictor_input_ids": [[1, 2, 3]] * num_samples,
        "predictor_attention_mask": [[1, 1, 1]] * num_samples,
        "num_target_tokens": [2] * num_samples
    })

class AuditLogCallback(TrainerCallback):
    """自定义回调：高亮打印每个 Step 的状态，用于直观确认断点续训的连续性。"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if IS_RANK0 and logs is not None and "loss" in logs:
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            alpha = logs.get("mask_alpha", 0.0) # 依赖 IFPruningCallback 注入
            print(f"\n[AUDIT LOG] Step: {state.global_step:02d} | Loss: {loss:.6f} | LR: {lr:.2e} | Alpha: {alpha:.4f}")

def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    test_phase = int(os.environ.get("TEST_PHASE", "0"))
    if test_phase not in [1, 2]:
        logger.error("Please set TEST_PHASE=1 (Checkpoint) or TEST_PHASE=2 (Restore).")
        sys.exit(1)

    # Use default configuration aligned with train.py
    cfg = RunConfig()
    
    # Override steps for rapid testing
    cfg.max_steps = 5 
    cfg.save_steps = 5
    cfg.logging_steps = 1
    cfg.mask_warmup_steps = 10 # 缩短测试下的预热周期以观察 Alpha 变化
    cfg.output_dir = "./gemma-12B-ifpruning-test-ckpt"
    
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ta_kwargs = {
        "output_dir": cfg.output_dir,
        "do_train": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.base_lr,
        "bf16": cfg.bf16,
        "save_steps": cfg.save_steps,
        "save_safetensors": True,
        "safe_serialization": True,
        "report_to": ["none"],
        "logging_steps": 1, # 强制每步打印
        "deepspeed": make_deepspeed_config(cfg, log_dir)
    }
    
    # 像 train.py 一样自动过滤不支持的参数
    valid_args = inspect.signature(TrainingArguments.__init__).parameters
    t_args = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if k in valid_args})

    dummy_dataset = create_dummy_dataset()

    logger.info("Initializing architecture for validation...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16,
        "local_files_only": cfg.local_files_only,
        "attn_implementation": cfg.attn_implementation
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    model, layers = patch_model_for_ifpruning(base_model, cfg)

    trainer = IFPruningTrainer(
        model=model,
        args=t_args,
        train_dataset=dummy_dataset,
        data_collator=DualCollator(b_pad=0, p_pad=0),
        p_lr=cfg.predictor_lr,
        b_lr=cfg.base_lr,
        callbacks=[
            IFPruningCallback(layers, cfg.mask_warmup_steps, cfg.abort_on_zero_loss_steps),
            AuditLogCallback()
        ]
    )

    if test_phase == 1:
        logger.info("=== PHASE 1: Executing Checkpoint Routine ===")
        try:
            trainer.train()
            logger.info("Phase 1 Complete: Checkpoint generated successfully.")
        except Exception as e:
            logger.error(f"Phase 1 Failed: {e}", exc_info=True)
            sys.exit(1)

    elif test_phase == 2:
        logger.info("=== PHASE 2: Executing Restoration Audit ===")
        
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(cfg.output_dir)
        
        if not resume_ckpt:
            logger.error(f"Cannot perform Phase 2. No checkpoint found in {cfg.output_dir}.")
            sys.exit(1)
            
        logger.info(f"Targeting checkpoint: {resume_ckpt}")
        
        # Validate Custom State Decoupling
        pred_ckpt_path = Path(resume_ckpt) / "predictor_mlp.safetensors"
        if not pred_ckpt_path.exists():
            logger.error(f"Integrity Error: Decoupled predictor weights missing at {pred_ckpt_path}")
            sys.exit(1)
            
        try:
            logger.info("Loading decoupled predictor weights...")
            ckpt_state = load_file(str(pred_ckpt_path))
            load_result = model.predictor.mlp.load_state_dict(ckpt_state, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys during restore: {load_result.missing_keys}")
            logger.info("Decoupled weights restored successfully.")
        except Exception as e:
            logger.error(f"Predictor restoration failed: {e}", exc_info=True)
            sys.exit(1)

        # Extend steps to allow trainer to resume and execute
        trainer.args.max_steps += 5
        
        try:
            logger.info("Resuming distributed training from checkpoint...")
            trainer.train(resume_from_checkpoint=resume_ckpt)
            logger.info("Phase 2 Complete: DeepSpeed ZeRO partitions and custom layers restored flawlessly.")
        except Exception as e:
            logger.error(f"Phase 2 Failed: Distributed state mismatch or corruption detected. {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()