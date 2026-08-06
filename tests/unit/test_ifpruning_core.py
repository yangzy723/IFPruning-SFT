import unittest
from unittest import mock

import torch
import torch.nn as nn

from ifpruning_core import (
    BoundedSoftTopK,
    DynamicMaskedFFN,
    SparsityPredictor,
)


class DummyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = mock.Mock()
        self.config.text_config.hidden_size = 8
        self.config.text_config.use_cache = True
        self.embedding = nn.Embedding(16, 8, dtype=torch.bfloat16)


class PredictorConfigTests(unittest.TestCase):
    def test_predictor_head_uses_extractor_dtype(self):
        with mock.patch(
            "transformers.AutoModel.from_pretrained", return_value=DummyExtractor()
        ):
            predictor = SparsityPredictor(
                num_layers=2,
                ffn_dim=4,
                extractor_path="unused",
                extractor_dtype=torch.bfloat16,
            )

        self.assertEqual(
            {parameter.dtype for parameter in predictor.parameters()},
            {torch.bfloat16},
        )


class DummyMLP(nn.Module):
    def __init__(self, hidden_size=5, intermediate_size=7):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SoftTopKTests(unittest.TestCase):
    def test_training_mask_is_sparse_soft_and_differentiable(self):
        scores = torch.randn(2, 3, 11, requires_grad=True)
        operation = BoundedSoftTopK(k=4)
        operation.train()
        mask = operation(scores)

        self.assertEqual(mask.shape, scores.shape)
        self.assertTrue(torch.equal((mask != 0).sum(dim=-1), torch.full((2, 3), 4)))
        self.assertTrue(torch.all(mask >= 0))
        self.assertTrue(torch.all(mask <= 1))
        self.assertTrue(torch.any((mask > 0) & (mask < 1)))

        (mask * torch.randn_like(mask)).sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(torch.count_nonzero(scores.grad).item(), 0)

    def test_eval_mask_is_exactly_binary(self):
        scores = torch.randn(2, 11)
        operation = BoundedSoftTopK(k=4)
        operation.eval()
        mask = operation(scores)
        self.assertTrue(torch.equal(mask.sum(dim=-1), torch.tensor([4.0, 4.0])))
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))

    def test_invalid_budget_fails_fast(self):
        operation = BoundedSoftTopK(k=12)
        with self.assertRaises(ValueError):
            operation(torch.randn(2, 11))


class DynamicFFNTests(unittest.TestCase):
    def test_full_budget_matches_dense_ffn_without_rescale(self):
        torch.manual_seed(0)
        original = DummyMLP()
        wrapped = DynamicMaskedFFN(
            original,
            target_dim=7,
        )
        wrapped.eval()
        wrapped.mask_alpha.fill_(1.0)
        wrapped.layer_scores = torch.randn(2, 7)
        inputs = torch.randn(2, 4, 5)
        self.assertTrue(torch.allclose(wrapped(inputs), original(inputs), atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
