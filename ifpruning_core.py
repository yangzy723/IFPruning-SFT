"""Shared IFPruning model components used by training and inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Return the longest ModuleList whose blocks expose an ``mlp`` module."""
    candidates = [
        module
        for module in model.modules()
        if isinstance(module, nn.ModuleList) and len(module) > 0 and hasattr(module[0], "mlp")
    ]
    if not candidates:
        raise ValueError("Could not find a transformer layer stack containing MLP blocks")
    return max(candidates, key=len)


class SparsityPredictor(nn.Module):
    """Prompt encoder plus an input-conditioned, layer-wise FFN scoring head."""

    def __init__(
        self,
        num_layers: int,
        ffn_dim: int,
        extractor_path: str,
        local_files_only: bool = True,
        cache_dir: str | None = None,
        hidden_dim: int = 128,
        output_init_std: float = 1e-4,
        extractor_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoModel

        self.num_layers = int(num_layers)
        self.ffn_dim = int(ffn_dim)
        self.hidden_dim = int(hidden_dim)

        self.extractor = AutoModel.from_pretrained(
            extractor_path,
            dtype=extractor_dtype,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.extractor.config.text_config.use_cache = False

        self.feature_dim = int(self.extractor.config.text_config.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(
                self.hidden_dim,
                self.num_layers * self.ffn_dim,
                bias=False,
            ),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=float(output_init_std))
        # ZeRO-3 requires every trainable parameter to use one dtype.
        self.mlp.to(dtype=extractor_dtype)

        self.register_buffer(
            "last_head_active_fraction", torch.tensor(float("nan")), persistent=False
        )
        self.register_buffer(
            "last_score_input_std", torch.tensor(float("nan")), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        hidden = output.last_hidden_state
        sequence_lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        features = hidden[batch_indices, sequence_lengths]
        features = features.to(dtype=self.mlp[0].weight.dtype)
        features = F.layer_norm(features.float(), (self.feature_dim,)).to(features.dtype)

        head_pre_activation = self.mlp[0](features)
        head_activation = self.mlp[1](head_pre_activation)
        scores = self.mlp[2](head_activation).view(-1, self.num_layers, self.ffn_dim)

        with torch.no_grad():
            active = head_activation.float().abs() > 1e-6
            self.last_head_active_fraction.copy_(active.float().mean())
            if scores.shape[0] > 1:
                centered = scores.float() - scores.float().mean(dim=0, keepdim=True)
                self.last_score_input_std.copy_(centered.square().mean().sqrt())
            else:
                self.last_score_input_std.fill_(float("nan"))
        return scores


class BoundedSoftTopK(nn.Module):
    """Differentiable SoftTopK with exactly ``k`` non-zero training entries."""

    def __init__(
        self,
        k: int,
        temperature: float = 1.0,
        iters: int = 32,
    ):
        super().__init__()
        self.k = int(k)
        self.temperature = max(float(temperature), 1e-6)
        self.iters = int(iters)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if not 0 < self.k <= scores.shape[-1]:
            raise ValueError(f"k={self.k} must be in [1, {scores.shape[-1]}]")

        logits = scores.float() / self.temperature
        with torch.no_grad():
            lower = logits.amin(dim=-1, keepdim=True) - 20.0
            upper = logits.amax(dim=-1, keepdim=True) + 20.0
            for _ in range(self.iters):
                threshold = (lower + upper) * 0.5
                mass = torch.sigmoid(logits - threshold).sum(dim=-1, keepdim=True)
                lower = torch.where(mass > self.k, threshold, lower)
                upper = torch.where(mass > self.k, upper, threshold)
            detached_threshold = (lower + upper) * 0.5

        # Restore the implicit derivative of sum(sigmoid(logits - tau)) = k
        # without retaining every binary-search iteration in the autograd graph.
        initial = torch.sigmoid(logits - detached_threshold)
        weights = (initial * (1.0 - initial)).detach()
        weighted_mean = (logits * weights).sum(dim=-1, keepdim=True) / weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        threshold = detached_threshold + weighted_mean - weighted_mean.detach()
        soft = torch.sigmoid(logits - threshold)

        top_indices = torch.topk(soft, self.k, dim=-1).indices
        indicator = torch.zeros_like(soft).scatter_(-1, top_indices, 1.0)
        if not self.training:
            return indicator.to(dtype=scores.dtype)
        mask = soft * indicator.detach()
        return mask.to(dtype=scores.dtype)


class DynamicMaskedFFN(nn.Module):
    """Apply a prompt-level mask to the intermediate channels of a gated FFN."""

    def __init__(
        self,
        mlp: nn.Module,
        target_dim: int,
        mask_temperature: float = 1.0,
        softtopk_iters: int = 32,
    ):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self.full_ffn_dim = int(self.gate_proj.out_features)
        self.target_ffn_dim = int(target_dim)
        if not 0 < self.target_ffn_dim <= self.full_ffn_dim:
            raise ValueError(
                f"target_dim={self.target_ffn_dim} must be in [1, {self.full_ffn_dim}]"
            )
        self.mask_op = BoundedSoftTopK(
            target_dim,
            mask_temperature,
            softtopk_iters,
        )
        self.layer_scores: torch.Tensor | None = None
        self.register_buffer(
            "mask_alpha", torch.tensor(0.0, dtype=torch.float32), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        if self.layer_scores is not None:
            mask = self.mask_op(self.layer_scores).to(dtype=hidden.dtype).unsqueeze(1)
            alpha = float(self.mask_alpha.item())
            sparse_hidden = hidden * mask
            if alpha >= 1.0:
                hidden = sparse_hidden
            elif alpha <= 0.0:
                hidden = hidden + mask * 0.0
            else:
                hidden = torch.lerp(hidden, sparse_hidden, alpha)
        return self.down_proj(hidden)
