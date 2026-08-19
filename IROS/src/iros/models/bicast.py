#!/usr/bin/env python3

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:

    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class BiCAST(nn.Module):

    VIS_DIM = 1024
    INTENT_DIM = 128
    SAFETY_DIM = 64

    def __init__(
        self,
        n_frames: int = 4,                              
        chunk_size: int = 5,                                          
        num_layers: int = 8,                                          
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        alpha_max: float = 0.9,                                      
        pretrained_backbone: bool = True,
        backbone_name: str = "resnet50",
        use_fsr: bool = True,
        use_safety: bool = True,
    ):

        super().__init__()
        self.n_frames = n_frames
        self.chunk_size = chunk_size
        self.alpha_max = alpha_max
        self.backbone_name = backbone_name.lower()
        self.use_fsr = bool(use_fsr)
        self.use_safety = bool(use_safety)

        if self.backbone_name == "resnet18":
            weights = (
                tv_models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained_backbone else None
            )
            backbone = tv_models.resnet18(weights=weights)
            backbone_dim = 512
        elif self.backbone_name == "resnet34":
            weights = (
                tv_models.ResNet34_Weights.IMAGENET1K_V1
                if pretrained_backbone else None
            )
            backbone = tv_models.resnet34(weights=weights)
            backbone_dim = 512
        elif self.backbone_name == "resnet50":
            weights = (
                tv_models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained_backbone else None
            )
            backbone = tv_models.resnet50(weights=weights)
            backbone_dim = 2048
        else:
            raise ValueError(
                "backbone_name must be one of: resnet18, resnet34, resnet50"
            )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.vis_proj = nn.Linear(backbone_dim, self.VIS_DIM)
        self.register_buffer(
            "pos_embed",
            sinusoidal_positional_encoding(n_frames, self.VIS_DIM),
            persistent=False,
        )

        self.intent_mlp = None
        if self.use_fsr:
            self.intent_mlp = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.INTENT_DIM),
            )

        self.safety_mlp = None
        if self.use_safety:
            self.safety_mlp = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.SAFETY_DIM),
            )

        self.d_model = (
            self.VIS_DIM
            + (self.INTENT_DIM if self.use_fsr else 0)
            + (self.SAFETY_DIM if self.use_safety else 0)
        )
        if self.d_model % nhead != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={nhead}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def make_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(self.d_model, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, chunk_size),
            )

        self.head_L = make_head()
        self.head_R = make_head()

    def forward(
        self,
        images: torch.Tensor,                             
        fsr_features: torch.Tensor,                                          
        safety_features: torch.Tensor,                               
    ) -> torch.Tensor:

        B, T, C, H, W = images.shape
        assert T == self.n_frames, f"expected {self.n_frames} frames, got {T}"

        feats = self.backbone(images.reshape(B * T, C, H, W))                          
        vis = self.vis_proj(feats).view(B, T, self.VIS_DIM)                     
        vis = vis + self.pos_embed                                               

        streams = [vis]
        if self.use_fsr:
            intent = self.intent_mlp(fsr_features.reshape(B, T, 6))
            streams.append(intent)
        if self.use_safety:
            safety = self.safety_mlp(safety_features)
            streams.append(safety)

        tokens = torch.cat(streams, dim=-1)                                        
        encoded = self.encoder(tokens)                                          
        rep = encoded.mean(dim=1)                                                

        logits_L = self.head_L(rep)                                       
        logits_R = self.head_R(rep)                                       
        alpha_L = self.alpha_max * torch.sigmoid(logits_L)                 
        alpha_R = self.alpha_max * torch.sigmoid(logits_R)
        return torch.stack([alpha_L, alpha_R], dim=-1)                       

class BiCASTLoss(nn.Module):

    def __init__(
        self,
        w_L: float = 0.6,
        w_R: float = 0.4,
        huber_delta: float = 0.1,
        lambda_s: float = 0.1,
        lambda_1: float = 1e-3,
    ):
        super().__init__()
        self.w = (w_L, w_R)
        self.lambda_s = lambda_s
        self.lambda_1 = lambda_1
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")

    def forward(
        self,
        alpha_pred: torch.Tensor,               
        alpha_target: torch.Tensor,             
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        loss = alpha_pred.new_zeros(())
        for i, w_i in enumerate(self.w):                
            pred_i, tgt_i = alpha_pred[..., i], alpha_target[..., i]
            huber_i = self.huber(pred_i, tgt_i)
            smooth_i = ((pred_i[:, 1:] - pred_i[:, :-1]) ** 2).sum(dim=1).mean()
            loss = loss + w_i * (huber_i + self.lambda_s * smooth_i)
        if model is not None and self.lambda_1 > 0:
            l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.lambda_1 * l1
        return loss

class ChunkAggregator:

    def __init__(self, chunk_size: int = 5, decay: float = 0.5):
        self.chunk_size = chunk_size
        self.decay = decay
        self._buffer: deque = deque(maxlen=chunk_size)                     

    def reset(self) -> None:
        self._buffer.clear()

    @torch.no_grad()
    def push(self, chunk: torch.Tensor) -> torch.Tensor:

        assert chunk.dim() == 2 and chunk.shape[0] == self.chunk_size, (
            f"expected [C={self.chunk_size}, 2] chunk, got {tuple(chunk.shape)}"
        )
        self._buffer.appendleft(chunk.detach())

        n = len(self._buffer)                      
        weights = torch.exp(
            -self.decay * torch.arange(n, dtype=chunk.dtype, device=chunk.device)
        )
        weights = weights / weights.sum()

        alpha = torch.zeros_like(chunk[0])
        for k in range(n):

            alpha = alpha + weights[k] * self._buffer[k][k]
        return alpha

def compute_fsr_features(force: np.ndarray, var_window: int = 5) -> np.ndarray:

    force = np.asarray(force, dtype=np.float32)
    single = force.ndim == 1
    if single:
        force = force[:, None]
    T, A = force.shape

    dF = np.zeros_like(force)
    dF[1:] = force[1:] - force[:-1]

    sigma = np.zeros_like(force)
    for t in range(T):
        lo = max(0, t - var_window + 1)
        sigma[t] = force[lo : t + 1].var(axis=0)

    feats = np.stack([force, dF, sigma], axis=-1)             
    return feats[:, 0, :] if single else feats
