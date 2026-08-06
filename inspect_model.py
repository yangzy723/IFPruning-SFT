#!/usr/bin/env python3
"""Inspect the local base-model configuration and transformer layout."""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from ifpruning_core import find_transformer_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./gemma-4-12B")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--print-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text_config = config.text_config
    print(f"architecture: {config.architectures}")
    print(f"model_type: {config.model_type}")
    print(f"hidden_size: {text_config.hidden_size}")
    print(f"num_hidden_layers: {text_config.num_hidden_layers}")
    print(f"intermediate_size: {text_config.intermediate_size}")
    if args.config_only:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    layers = find_transformer_layers(model)
    print(f"transformer_layers: {len(layers)}")
    print(f"layer_type: {type(layers[0]).__name__}")
    print(f"mlp_type: {type(layers[0].mlp).__name__}")
    if args.print_model:
        print(model)


if __name__ == "__main__":
    main()
