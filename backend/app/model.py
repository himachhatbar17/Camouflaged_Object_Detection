"""
CamoNet -- Delta-T-Guided Cross-Spectral Transfer network for camouflaged
object detection (COD).

Extracted from the original training notebook (Cell 4 module definitions +
Cell 5 assembly). Architecture is UNCHANGED from training. Only the
inference-time entry point (`forward_s4`, RGB-only) is used in deployment.

Stage 4 deployment path:
    RGB image -> rgb_enc -> pseudo_gen (predicts pseudo-ΔT from RGB) ->
    gate -> self-attention (Q=K=V=RGB, bias=pseudo-ΔT) -> fpn -> decoder
    (+ edge head) -> mask_head -> mask logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

IMG_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# MODULE A -- Dual Encoders
# --------------------------------------------------------------------------
class RGBEncoder(nn.Module):
    """ResNet-18 backbone, 4-stage multi-scale output."""

    def __init__(self, pretrained=False):
        super().__init__()
        net = resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_ch = [64, 128, 256, 512]

    def forward(self, x, use_ckpt=False):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class ThermalEncoder(nn.Module):
    """ResNet-34 backbone with a 1-channel stem. Unused at inference
    (Stage 4 is RGB-only) but kept so checkpoints load with strict=True."""

    def __init__(self, pretrained=False):
        super().__init__()
        net = resnet34(weights=None)
        c = net.conv1
        nc = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
        net.conv1 = nc
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_ch = [64, 128, 256, 512]

    def forward(self, x, use_ckpt=False):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


# --------------------------------------------------------------------------
# MODULE B -- Delta-T Residual Extractor
# --------------------------------------------------------------------------
class DeltaTExtractor(nn.Module):
    def __init__(self, kernel_sizes=(3, 7, 11)):
        super().__init__()
        self.ks = kernel_sizes

    def _local_mean(self, x, k):
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)

    def forward(self, x):
        return torch.cat([x - self._local_mean(x, k) for k in self.ks], dim=1)

    def multiscale(self, feats):
        return [self(f) for f in feats]


# --------------------------------------------------------------------------
# MODULE C -- Illumination Gate
# --------------------------------------------------------------------------
class IlluminationGate(nn.Module):
    def __init__(self, rgb_ch, dt_ch, hidden=32):
        super().__init__()
        self.rp = nn.Conv2d(rgb_ch, hidden, 1)
        self.dp = nn.Conv2d(dt_ch, hidden, 1)
        self.net = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_f, dt_f):
        if dt_f.shape[-2:] != rgb_f.shape[-2:]:
            dt_f = F.interpolate(dt_f, rgb_f.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([self.rp(rgb_f), self.dp(dt_f)], dim=1))


# --------------------------------------------------------------------------
# MODULE D -- Delta-T-Guided Attention (SRA, PVT-style)
# --------------------------------------------------------------------------
class DeltaTGuidedAttention(nn.Module):
    def __init__(self, channels, bias_ch, num_heads=4, sr_ratio=1):
        super().__init__()
        assert channels % num_heads == 0
        self.nh = num_heads
        self.hd = channels // num_heads
        self.sr = sr_ratio
        self.scale = self.hd ** -0.5

        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.bp = nn.Conv2d(bias_ch, num_heads, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, q_src, kv_src, bias_src):
        B, C, H, W = kv_src.shape
        if q_src.shape[-2:] != (H, W):
            q_src = F.interpolate(q_src, (H, W), mode="bilinear", align_corners=False)
        if bias_src.shape[-2:] != (H, W):
            bias_src = F.interpolate(bias_src, (H, W), mode="bilinear", align_corners=False)

        Q = self.q(q_src)
        if self.sr > 1:
            kv_r = F.avg_pool2d(kv_src, self.sr, self.sr, ceil_mode=True)
            bs_r = F.avg_pool2d(bias_src, self.sr, self.sr, ceil_mode=True)
        else:
            kv_r, bs_r = kv_src, bias_src
        Hr, Wr = kv_r.shape[-2:]

        Q = Q.view(B, self.nh, self.hd, H * W)
        K = self.k(kv_r).view(B, self.nh, self.hd, Hr * Wr)
        V = self.v(kv_r).view(B, self.nh, self.hd, Hr * Wr)
        bias = self.bp(bs_r).view(B, self.nh, 1, Hr * Wr)

        attn = torch.einsum("bhdn,bhdm->bhnm", Q, K) * self.scale + bias
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, V)
        out = out.reshape(B, C, H, W)
        out = self.out(out) + q_src
        out = self.norm(out)
        return out, attn


# --------------------------------------------------------------------------
# MODULE E -- Pseudo-Delta-T Generator
# --------------------------------------------------------------------------
class PseudoDeltaTGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# MODULE F -- FPN Multi-Scale Fusion
# --------------------------------------------------------------------------
class FPNFusion(nn.Module):
    def __init__(self, in_ch_list, out_ch=128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_ch_list])
        self.smo = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )
            for _ in in_ch_list
        ])

    def forward(self, feats):
        lats = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], lats[i].shape[-2:], mode="bilinear", align_corners=False)
        return [s(l) for s, l in zip(self.smo, lats)]


# --------------------------------------------------------------------------
# MODULE G -- Edge Attention Decoder
# --------------------------------------------------------------------------
class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EdgeAttentionDecoder(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.d3 = DecBlock(ch, ch, ch)
        self.d2 = DecBlock(ch, ch, ch)
        self.d1 = DecBlock(ch, ch, ch)
        self.edge_head = nn.Conv2d(ch, 1, 3, padding=1)
        self.edge_gate = nn.Sequential(nn.Conv2d(1, ch, 1), nn.Sigmoid())
        self.refine = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        x = self.d3(f4, f3)
        x = self.d2(x, f2)
        x = self.d1(x, f1)
        edge = self.edge_head(x)
        x = x * self.edge_gate(torch.sigmoid(edge)) + x
        return self.refine(x), edge


# --------------------------------------------------------------------------
# MODULE H -- Segmentation / Mask Head
# --------------------------------------------------------------------------
class MaskHead(nn.Module):
    def __init__(self, in_ch=128, out_size=IMG_SIZE):
        super().__init__()
        self.out_size = out_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return F.interpolate(self.conv(x), (self.out_size, self.out_size), mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
# AUX DETECTION HEAD -- present in checkpoint dict for strict loading only
# --------------------------------------------------------------------------
class AuxDetHead(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 5, 1),
        )

    def forward(self, x):
        return self.head(x)


# --------------------------------------------------------------------------
# FULL MODEL
# --------------------------------------------------------------------------
class CamoNet(nn.Module):
    WORK_DIM = 128
    N_KS = 3
    SR = [8, 4, 2, 1]

    def __init__(self, pretrained=False):
        super().__init__()
        WD = self.WORK_DIM
        DTC = WD * self.N_KS

        self.rgb_enc = RGBEncoder(pretrained)
        self.th_enc = ThermalEncoder(pretrained)
        self.rgb_align = nn.ModuleList([nn.Conv2d(c, WD, 1) for c in [64, 128, 256, 512]])
        self.th_align = nn.ModuleList([nn.Conv2d(c, WD, 1) for c in [64, 128, 256, 512]])

        self.delta_t = DeltaTExtractor(kernel_sizes=(3, 7, 11))
        self.gate = IlluminationGate(WD, DTC)
        self.attn = nn.ModuleList([
            DeltaTGuidedAttention(WD, DTC, num_heads=4, sr_ratio=self.SR[i]) for i in range(4)
        ])
        self.pseudo_gen = nn.ModuleList([PseudoDeltaTGenerator(WD, DTC) for _ in range(4)])

        self.rgb_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(WD, 256), nn.ReLU(), nn.Linear(256, 128))
        self.th_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(WD, 256), nn.ReLU(), nn.Linear(256, 128))

        self.aux_det = AuxDetHead(WD)
        self.fpn = FPNFusion([WD] * 4, WD)
        self.decoder = EdgeAttentionDecoder(WD)
        self.mask_head = MaskHead(WD, IMG_SIZE)

    def _encode(self, rgb, thermal=None, use_ckpt=False):
        rf = self.rgb_enc(rgb, use_ckpt)
        rf = [self.rgb_align[i](f) for i, f in enumerate(rf)]
        tf = None
        if thermal is not None:
            tf = self.th_enc(thermal, use_ckpt)
            tf = [self.th_align[i](f) for i, f in enumerate(tf)]
        return rf, tf

    def _gate_resize(self, g, size):
        if g.shape[-2:] != size:
            g = F.interpolate(g, size, mode="bilinear", align_corners=False)
        return g

    @torch.no_grad()
    def forward_s4(self, rgb):
        """Deployment path: RGB only. Pseudo-Delta-T self-attention
        (Q=K=V=RGB, bias=pseudo-ΔT). Returns (mask_logits, edge_logits,
        pred_dt, gate, attn_weights)."""
        rf, _ = self._encode(rgb, thermal=None)
        pred_dt = [self.pseudo_gen[i](rf[i]) for i in range(4)]
        gate = self.gate(rf[0], pred_dt[0])

        attn_out = []
        attn_weights = []
        for i in range(4):
            sa, aw = self.attn[i](rf[i], rf[i], pred_dt[i])
            g = self._gate_resize(gate, sa.shape[-2:])
            attn_out.append(g * sa + (1 - g) * rf[i])
            attn_weights.append(aw)

        fused = self.fpn(attn_out)
        dec_feat, edge_log = self.decoder(fused)
        mask_log = self.mask_head(dec_feat)
        return mask_log, edge_log, pred_dt, gate, attn_weights


def load_camonet(ckpt_path, device="cpu", strict=True):
    """Load a CamoNet checkpoint saved by the training notebook
    (dict with 'state_dict' + 'meta')."""
    model = CamoNet(pretrained=False)
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = obj["state_dict"] if "state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    model.to(device)
    model.eval()
    return model, obj.get("meta", {})


def attn_to_heatmap(attn_w, feat_hw, out_size=IMG_SIZE):
    """Turn a DeltaTGuidedAttention attn_weights tensor (B, heads, N, M)
    into a per-image [0,1] heatmap of shape (B, out_size, out_size).

    NOTE (bugfix): the original notebook version used
    `attn_w.mean(dim=1).mean(dim=-1)` -- averaging a post-softmax
    distribution over its own key dimension. Since every softmax row sums
    to exactly 1 by definition, that mean is *always* exactly 1/M for
    every query position, completely independent of the actual attention
    pattern. It renders as a flat, uninformative heatmap (verified
    empirically: std=0.0 across all query positions, at every scale).
    We use the per-query MAX attention weight instead (averaged over
    heads first) -- this measures how concentrated/confident each query's
    attention is, and genuinely varies across spatial positions."""
    H, W = feat_hw
    amap = attn_w.mean(dim=1).amax(dim=-1)  # (B, N=H*W) -- varies per query, unlike the old mean-over-M
    amap = amap.view(-1, 1, H, W)
    amap = F.interpolate(amap, size=(out_size, out_size), mode="bilinear", align_corners=False)
    amap = amap[:, 0]
    amin = amap.reshape(amap.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    amax = amap.reshape(amap.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    return (amap - amin) / (amax - amin + 1e-8)
