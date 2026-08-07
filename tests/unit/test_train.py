import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from train import IFPruningTrainer, PredictorTokenizerCheckpointCallback


class MixedDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.predictor = nn.ModuleDict(
            {
                "backbone": nn.Linear(4, 4, bias=False, dtype=torch.bfloat16),
                "head": nn.Linear(4, 4, bias=False, dtype=torch.float32),
            }
        )


class TrainingConfigTests(unittest.TestCase):
    def test_validation_batch_and_checkpoint_retention(self):
        from train import RunConfig

        config = RunConfig()
        self.assertLess(config.per_device_eval_batch_size, config.per_device_train_batch_size)
        self.assertEqual(config.save_steps, 500)
        self.assertEqual(config.save_total_limit, 1)

    def test_predictor_tokenizer_is_saved_in_checkpoint(self):
        tokenizer = SimpleNamespace(
            save_pretrained=lambda path: Path(path).mkdir(parents=True)
        )
        callback = PredictorTokenizerCheckpointCallback(tokenizer)
        with tempfile.TemporaryDirectory() as output_dir:
            callback.on_save(
                SimpleNamespace(output_dir=output_dir),
                SimpleNamespace(global_step=500, is_world_process_zero=True),
                SimpleNamespace(),
            )
            self.assertTrue(
                (Path(output_dir) / "checkpoint-500" / "predictor_tokenizer").is_dir()
            )


class ZeRO3OptimizerGroupingTests(unittest.TestCase):
    def test_mixed_trainable_dtypes_fail_before_deepspeed(self):
        trainer = object.__new__(IFPruningTrainer)
        trainer.optimizer = None
        trainer.model = MixedDtypeModel()
        trainer.predictor_lr = 1e-5
        trainer.base_lr = 2e-6
        trainer.args = SimpleNamespace(
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.999,
        )

        with patch("train.FusedAdam", side_effect=lambda groups, betas: groups):
            with self.assertRaisesRegex(RuntimeError, "all trainable parameters"):
                trainer.create_optimizer()

    def test_decay_uses_original_zero_shape(self):
        trainer = object.__new__(IFPruningTrainer)
        parameter = nn.Parameter(torch.ones(1))
        parameter.ds_shape = torch.Size((4, 4))
        self.assertFalse(trainer._is_no_decay("projection.weight", parameter))


if __name__ == "__main__":
    unittest.main()
