import os
import glob
import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

N_SLOT_TYPES, MASK_IDX = (6, 0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_SLOTS = [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]
DINO_CROP_MM = 130.0
DINO_SIZE = 336
DINO_SLICE_BAND = (0.12, 0.88)
DINO_N_SLICE = 16
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

class ViTSlotToken(nn.Module):
    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, 'num_prefix_tokens', 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, 'attn', None)
            if a is not None and hasattr(a, 'num_prefix_tokens'):
                a.num_prefix_tokens = a.num_prefix_tokens + 1

    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)

    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, 'patch_drop', None), x)
        x = self._maybe(getattr(v, 'norm_pre', None), x)
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        if rope is not None:
            if getattr(v, 'rope_mixed', False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)

def _seg_mean_max(v, sidx, B):
    D = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype))
    mean = torch.zeros(B, D, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, D), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), v, reduce='amax', include_self=True)
    return torch.cat([mean, mx], 1)

def _pad_kv(x, sidx, B, norm):
    T, P, D = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    pad = x.new_zeros(B, S, P, D)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return (norm(pad.reshape(B, S * P, D)), ~keep.repeat_interleave(P, dim=1))

class _GatedDelta(nn.Module):
    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))

    def delta(self, pat, sidx, B, return_attn):
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        return ((self.d_norm(att) * self.dw).sum(-1) + self.db, w)

class CodexResidualPool(_GatedDelta):
    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class Readout(nn.Module):
    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        self.pool_kind, self.k = (pool, n_labels)
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=0)
        self.pool = CodexResidualPool(d, n_labels, pe=pe)

    def forward(self, f, slot, sidx, B, return_attn=False):
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        return self.pool(f, slot, sidx, B, pres)[0]

class Net(nn.Module):
    def __init__(self, enc, cond, n_meta=0, pool='mean_max', stem='native', n_slice=16):
        super().__init__()
        self.enc, self.cond = (enc, cond)
        self.tokens = pool in ('xattn', 'xres', 'clsadd', 'xcodex')
        D = enc.num_features
        self.readout = Readout(pool, D)

    def forward(self, im, slot, smeta, sidx, B, vm=None):
        f = self.enc.forward_features(im, slot) if self.cond == 'token' else self.enc.forward_features(im)
        if self.tokens:
            inner = getattr(self.enc, 'vit', self.enc)
            orig = getattr(self.enc, '_orig_prefix', getattr(inner, 'num_prefix_tokens', 1))
            f = torch.cat([f[:, :1], f[:, orig:]], 1)
        return self.readout(f, slot, sidx, B)

def _dino_series_side(path: str) -> float:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return float(ds.ImagePositionPatient[0])
    except Exception:
        return 0.0

def _dino_ordered_files(sdir: str, cap: int = 64):
    keyed = []
    files = glob.glob(os.path.join(sdir, "*.dcm"))
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            keyed.append((int(getattr(ds, "InstanceNumber", 0) or 0), f))
        except Exception:
            continue
        if len(keyed) >= cap * 4:
            break
    keyed.sort(key=lambda x: x[0])
    return [f for _, f in keyed]

def _dino_read_crop(path: str):
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = DINO_CROP_MM / max(arr.shape)
    half = int(round(DINO_CROP_MM / max(ps, 1e-4) / 2))
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    y0, y1 = max(0, cy - half), min(arr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(arr.shape[1], cx + half)
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop

def _dino_window(crop: np.ndarray, lo: float, hi: float, flip: bool) -> np.ndarray:
    c = np.clip((crop - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = cv2.resize(c, (DINO_SIZE, DINO_SIZE), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if flip else img

def _dino_render(path: str, flip: bool):
    crop = _dino_read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1.0, 99.0])
    return _dino_window(crop, lo, hi, flip)

def build_dino_study(study_id: str, recs: list, series_dir: str):
    out = np.zeros((len(DINO_SLOTS), DINO_N_SLICE, DINO_SIZE, DINO_SIZE), dtype=np.float32)
    mask = np.zeros(len(DINO_SLOTS), dtype=np.uint8)
    if not recs:
        return out, mask
    
    rows = pd.DataFrame(recs)
    for s_i, (plane, fs) in enumerate(DINO_SLOTS):
        sub = rows[(rows["Anatomical_Plane"] == plane) & (rows["Fat_Suppression"].astype(int) == fs)]
        if sub.empty:
            continue
        ser_uid = sub.iloc[0]["SeriesInstanceUID"]
        s_path = os.path.join(series_dir, study_id, ser_uid)
        files = _dino_ordered_files(s_path)
        if not files:
            continue
        flip = (plane != "Sagittal") and (_dino_series_side(files[0]) < 0)
        lo_f, hi_f = DINO_SLICE_BAND
        i0 = int(round(lo_f * (len(files) - 1)))
        i1 = int(round(hi_f * (len(files) - 1)))
        avail = list(range(i0, i1 + 1))
        if len(avail) >= DINO_N_SLICE:
            picks = [avail[int(round(t))] for t in np.linspace(0, len(avail) - 1, DINO_N_SLICE)]
        else:
            picks = [int(round(t)) for t in np.linspace(0, len(files) - 1, DINO_N_SLICE)]
        
        slot_frames = []
        for p_idx in picks:
            fr = _dino_render(files[min(p_idx, len(files) - 1)], flip)
            if fr is None:
                fr = np.zeros((DINO_SIZE, DINO_SIZE), dtype=np.float32)
            slot_frames.append(fr)
        out[s_i] = np.stack(slot_frames, axis=0)
        mask[s_i] = 1
    return out, mask

print('Loading DINOv3 model...')
ckpt_path = 'checkpoints/m_f0.pt'
z = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = z['cfg']
enc = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0, in_chans=cfg.get('n_slice', 16), img_size=cfg['img'])
enc = ViTSlotToken(enc, N_SLOT_TYPES)
model = Net(enc, cfg['cond'], cfg.get('n_meta', 0), cfg['pool'], stem='native', n_slice=cfg.get('n_slice', 16))
model.load_state_dict(z['state_dict'], strict=False)
model.eval()

# Process 1 local study
df_series = pd.read_csv('data/test_series.csv')
studies = df_series['StudyInstanceUID'].unique()
study = studies[0]
recs = df_series[df_series['StudyInstanceUID'] == study].to_dict('records')

print(f'Building DINO inputs for study {study}...')
study_imgs, study_mask = build_dino_study(study, recs, 'data/test_series')
present = np.nonzero(study_mask > 0)[0]
print(f'Active slots: {present} (count: {len(present)})')

if len(present) > 0:
    blk = torch.from_numpy(study_imgs[present]).float()  # (n_present, 16, 336, 336)
    slots = torch.from_numpy(present + 1).long()
    sidx = torch.zeros(len(present), dtype=torch.long)
    smeta = torch.zeros(len(present), 0)
    with torch.no_grad():
        logits = model(blk, slots, smeta, sidx, 1)
        probs = torch.sigmoid(logits)[0].numpy()
        print('DINO Inference SUCCESS! Probs:', probs)
        assert len(probs) == 12, 'Must have 12 class probabilities'
