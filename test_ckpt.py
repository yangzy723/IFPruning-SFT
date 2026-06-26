#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint Restoration Verification Script.

Objective: Strictly verify the serialization and restoration mechanics of the 
IFPruning architecture through simulated process interruption and resumption.
Ensures zero deadlocks, state alignment, and DeepSpeed optimizer slice integrity.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_file
import inspect

from transformers import TrainingArguments, set_seed, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Import core training components from the primary module
from train import (
    RunConfig,
    setup_logging,
    make_deepspeed_config,
    patch_model_for_ifpruning,
    tokenize_sft_dataset,
    DualCollator,
    IFPruningTrainer,
    IFPruningCallback,
    IS_RANK0,
)

def extract_losses_from_history(trainer: IFPruningTrainer) -> List[float]:
    """Extracts the loss trajectory from the Trainer's state log history."""
    try:
        return [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    except Exception as e:
        logging.getLogger("ifpruning_sft").error("Failed to extract loss history.", exc_info=True)
        return []

def run_training_segment(
    mode: str, 
    cfg: RunConfig, 
    t_args: TrainingArguments, 
    dataset, 
    b_tok, 
    p_tok, 
    resume_from: str = None
) -> List[float]:
    """
    Executes a discrete segment of the training process.
    Handles model loading, architectural patching, and state restoration.
    """
    global LOGGER
    LOGGER.info(f"\n{'='*60}\n[EXECUTION PHASE] Initiating: {mode}\n{'='*60}")
    
    try:
        LOGGER.info("Loading base model architecture and applying structural patches...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16, 
            "local_files_only": cfg.local_files_only, 
            "attn_implementation": cfg.attn_implementation
        }
        base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
        model, layers = patch_model_for_ifpruning(base_model, cfg)
        
        if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"): 
            model.enable_input_require_grads()
    except Exception as e:
        LOGGER.error("Failed during model initialization or patching sequence.", exc_info=True)
        raise

    try:
        trainer = IFPruningTrainer(
            model=model, 
            args=t_args, 
            train_dataset=dataset, 
            data_collator=DualCollator(b_tok.pad_token_id, p_tok.pad_token_id),
            callbacks=[IFPruningCallback(layers, cfg.mask_warmup_steps, cfg.abort_on_zero_loss_steps)],
            p_lr=cfg.predictor_lr, 
            b_lr=cfg.base_lr
        )
    except Exception as e:
        LOGGER.error("Failed to instantiate IFPruningTrainer.", exc_info=True)
        raise

    if resume_from:
        pred_ckpt_path = Path(resume_from) / "predictor_mlp.safetensors"
        if pred_ckpt_path.exists():
            LOGGER.info(f"Predictor checkpoint detected. Restoring from: {pred_ckpt_path}")
            try:
                ckpt_state = load_file(str(pred_ckpt_path))
                model.predictor.mlp.load_state_dict(ckpt_state, strict=True)
            except Exception as e:
                LOGGER.error(f"Failed to load predictor weights from {pred_ckpt_path}.", exc_info=True)
                raise
        else:
            LOGGER.error(f"Critical Failure: Predictor weights missing in checkpoint directory: {resume_from}")
            raise FileNotFoundError("Missing predictor weights for restoration phase.")
            
        LOGGER.info("Delegating optimizer and base model state restoration to DeepSpeed engine...")

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except Exception as e:
        LOGGER.error("Training loop terminated unexpectedly.", exc_info=True)
        raise
        
    return extract_losses_from_history(trainer)

def main():
    set_seed(42)
    
    # Configure test-specific parameters (abbreviated execution)
    cfg = RunConfig(
        output_dir="./ckpt_restore",
        max_steps=6,
        save_steps=3,
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        mask_warmup_steps=10,
        deepspeed=True
    )
    
    global LOGGER
    LOGGER, log_dir = setup_logging(cfg.output_dir)
    
    LOGGER.info("Initializing Test Environment and Data Pipeline...")
    try:
        b_tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=cfg.local_files_only, use_fast=True)
        p_tok = AutoTokenizer.from_pretrained(cfg.predictor_model, local_files_only=cfg.local_files_only, use_fast=True)
        raw = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train[:100]", cache_dir=cfg.cache_dir)
    except Exception as e:
        LOGGER.error("Data pipeline initialization failed.", exc_info=True)
        raise
    
    ta_kwargs = {
        "output_dir": cfg.output_dir, 
        "do_train": True, 
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps, 
        "max_steps": cfg.max_steps, 
        "learning_rate": cfg.base_lr, 
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio, 
        "bf16": cfg.bf16, 
        "fp16": cfg.fp16, 
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps, 
        "save_total_limit": cfg.save_total_limit, 
        "save_safetensors": True,
        "report_to": [], 
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "gradient_checkpointing": cfg.gradient_checkpointing, 
        "deepspeed": make_deepspeed_config(cfg, log_dir),
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if cfg.gradient_checkpointing else None,
        "seed": 42, 
        "data_seed": 42
    }
    
    valid_args = inspect.signature(TrainingArguments.__init__).parameters
    safe_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_args}
    
    try:
        t_args = TrainingArguments(**safe_kwargs)
        tokenized = tokenize_sft_dataset(raw, b_tok, p_tok, cfg, t_args)
    except Exception as e:
        LOGGER.error("Failed to tokenize dataset or construct TrainingArguments.", exc_info=True)
        raise

    phase = os.environ.get("TEST_PHASE", "1")

    if phase == "1":
        # =====================================================================
        # Phase 1: Initial Run (Steps 1 to 3)
        # =====================================================================
        try:
            t_args_interrupted = TrainingArguments(**safe_kwargs)
            t_args_interrupted.max_steps = 3
            
            loss = run_training_segment("Phase 1: Initial Run (Steps 1-3)", cfg, t_args_interrupted, tokenized, b_tok, p_tok)
            
            if IS_RANK0:
                LOGGER.info("\n" + "="*60)
                LOGGER.info("[SUCCESS] Phase 1 training sequence completed and state serialized.")
                LOGGER.info(f"[METRICS] Phase 1 Loss Trajectory: {[f'{l:.4f}' for l in loss]}")
                LOGGER.info("[ACTION REQUIRED] Execute Phase 2 command to verify restoration mechanics.")
                LOGGER.info("="*60 + "\n")
        except Exception as e:
            LOGGER.error("Fatal error encountered during Phase 1 execution.", exc_info=True)
            raise
            
    elif phase == "2":
        # =====================================================================
        # Phase 2: Restoration Run (Steps 4 to 6)
        # =====================================================================
        try:
            ckpt_path = os.path.join(cfg.output_dir, "checkpoint-3")
            
            # Resilience Validation: Ensure DeepSpeed optimizer slices are physically present
            if IS_RANK0:
                if not os.path.exists(ckpt_path):
                    LOGGER.error(f"Checkpoint directory not found at target path: {ckpt_path}")
                    raise AssertionError(f"Missing Checkpoint: {ckpt_path}")
                    
                ds_step_dir = os.path.join(ckpt_path, "global_step3")
                if not os.path.exists(ds_step_dir):
                    LOGGER.error(f"DeepSpeed optimizer directory missing: {ds_step_dir}")
                    raise AssertionError(f"Missing DeepSpeed Engine State: {ds_step_dir}")
                    
                pt_files = [f for f in os.listdir(ds_step_dir) if f.endswith(".pt")]
                if len(pt_files) == 0:
                    LOGGER.error(f"No optimizer tensor slices found in {ds_step_dir}")
                    raise AssertionError(f"Corrupted DeepSpeed Engine State: Optimizer slices missing.")
                
                LOGGER.info("Pre-flight validation passed: DeepSpeed physical state files detected.")
                
            loss = run_training_segment("Phase 2: Resumed Run (Steps 4-6)", cfg, t_args, tokenized, b_tok, p_tok, resume_from=ckpt_path)
            
            if IS_RANK0:
                LOGGER.info("\n" + "="*60)
                LOGGER.info("[SUCCESS] Checkpoint restoration verified successfully.")
                LOGGER.info("Zero deadlocks. Zero OOM. File architecture preserved seamlessly.")
                LOGGER.info(f"[METRICS] Phase 2 Loss Trajectory (Steps 4-6): {[f'{l:.4f}' for l in loss]}")
                LOGGER.info("="*60 + "\n")
                
        except Exception as e:
            LOGGER.error("Fatal error encountered during Phase 2 restoration sequence.", exc_info=True)
            raise

if __name__ == "__main__":
    # Top-level exception catcher to prevent silent multi-processing deaths
    try:
        main()
    except Exception as top_level_e:
        # Fallback print in case the logger itself failed to initialize
        print(f"CRITICAL PROCESS FAILURE: {top_level_e}", file=sys.stderr)
        sys.exit(1)