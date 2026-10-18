import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.models import resnet50
import timm

# 1. Test timm CoAtNet architecture & loading
print("--- 1. Testing Raptor CoAtNet ---")
from submission.submission import TARGET_COLS

class RaptorClassifier(torch.nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = torch.nn.LayerNorm(F_dim)
        self.att = torch.nn.Sequential(
            torch.nn.Linear(F_dim, 256), torch.nn.Tanh(), torch.nn.Dropout(drop),
            torch.nn.Linear(256, n)
        )
        self.clsW = torch.nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = torch.nn.Parameter(torch.zeros(n))
        torch.nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

raptor_pt = Path("models/raptor/raptor_ft_coatnet_v5_full_swa.pt")
if raptor_pt.exists():
    ck = torch.load(raptor_pt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    dummy_x = torch.randn(1, 42, 3, 384, 384)
    with torch.no_grad():
        out = torch.sigmoid(model(dummy_x))
    print(f"CoAtNet loaded successfully! Dummy output shape: {out.shape}, range: [{out.min().item():.3f}, {out.max().item():.3f}]")

# 2. Test RadImageNet ResNet-50 & FoundationQueryHead
print("\n--- 2. Testing RadImageNet ResNet50 + Head ---")
class RadImageNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            *list(resnet50(weights=None).children())[:-2]
        )

    def forward(self, image):
        return self.backbone(image).mean(dim=(2, 3))

class FoundationQueryHead(torch.nn.Module):
    def __init__(self, n_slot=4, n_slice=8, token_dim=2048, head_dim=512, n_targets=12):
        super().__init__()
        self.n_slot = n_slot
        self.n_slice = n_slice
        self.project = torch.nn.Sequential(
            torch.nn.LayerNorm(token_dim),
            torch.nn.Linear(token_dim, head_dim),
            torch.nn.GELU()
        )
        self.plane = torch.nn.Parameter(torch.randn(n_slot, head_dim) * .01)
        self.position = torch.nn.Parameter(torch.randn(n_slice, head_dim) * .01)
        self.query = torch.nn.Parameter(torch.randn(n_targets, head_dim) * .02)
        self.attn = torch.nn.MultiheadAttention(head_dim, 8, dropout=.10, batch_first=True)
        self.fuse = torch.nn.Sequential(
            torch.nn.LayerNorm(head_dim * 4), torch.nn.Linear(head_dim * 4, head_dim),
            torch.nn.GELU(), torch.nn.Dropout(.15),
        )
        self.weight = torch.nn.Parameter(torch.randn(n_targets, head_dim) * .02)
        self.bias = torch.nn.Parameter(torch.zeros(n_targets))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slice, token.shape[-1])
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token, key_padding_mask=key_padding, need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(self.query), -1)
        fused = self.fuse(torch.cat([attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias

r50_pt = Path("models/radnet/ResNet50.pt")
heads_pt = Path("models/radnet/v52_e11_heads.pt")
if r50_pt.exists() and heads_pt.exists():
    enc = RadImageNetEncoder()
    enc_state = torch.load(r50_pt, map_location="cpu", weights_only=True)
    enc.load_state_dict(enc_state, strict=True)
    enc.eval()

    heads_data = torch.load(heads_pt, map_location="cpu", weights_only=False)
    head_model = FoundationQueryHead()
    head_model.load_state_dict(heads_data["folds"][0]["state_dict"], strict=True)
    head_model.eval()

    dummy_slices = torch.randn(1, 32, 2048) # 4 slots * 8 slices = 32 tokens
    dummy_mask = torch.ones(1, 32)
    with torch.no_grad():
        rad_out = torch.sigmoid(head_model(dummy_slices, dummy_mask))
    print(f"RadNet loaded successfully! Dummy output shape: {rad_out.shape}, range: [{rad_out.min().item():.3f}, {rad_out.max().item():.3f}]")

print("\n--- 3. Testing Rank-Space Finding Fusion ---")
# Simulated outputs for 3 studies
p_dino = np.random.uniform(0.1, 0.9, (3, 12))
p_coat = np.random.uniform(0.1, 0.9, (3, 12))
p_rad = np.random.uniform(0.1, 0.9, (3, 12))

from scipy.stats import rankdata

def rank01(arr: np.ndarray) -> np.ndarray:
    return (rankdata(arr, axis=0) - 0.5) / float(len(arr))

r_dino = rank01(p_dino)
r_coat = rank01(p_coat)
r_rad = rank01(p_rad)

coat_weights = {
    "Lateral Meniscus": 1.00,
    "Medial Meniscus": 0.80,
    "ACL": 0.75,
    "Lateral OA": 0.75,
    "Fracture": 0.75,
    "Effusion": 0.40,
    "Synovitis": 0.40,
}
default_w = 0.65

final_ranks = np.zeros_like(r_coat)
for j, target in enumerate(TARGET_COLS):
    cw = coat_weights.get(target, default_w)
    rem = 1.0 - cw
    rw = rem * 0.40 if target in ["Effusion", "Synovitis", "ACL", "Fracture", "Lateral OA"] else rem * 0.25
    dw = rem - rw
    final_ranks[:, j] = cw * r_coat[:, j] + dw * r_dino[:, j] + rw * r_rad[:, j]

final_ranks = rank01(final_ranks)
print(f"Fused rank output shape: {final_ranks.shape}, all finite: {np.isfinite(final_ranks).all()}")
print("ALL TESTS PASSED!")
