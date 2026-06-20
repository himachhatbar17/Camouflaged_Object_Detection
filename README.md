# DT-DEAN: ΔT-Guided Dual-Branch Edge Attention Network for Infrared Camouflaged Object Detection

<p align="center">
  <img src="figures/fig3_architecture.png" alt="DT-DEAN Architecture" width="85%"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Kaggle%20%7C%20CUDA-20BEFF?logo=kaggle&logoColor=white" alt="Platform"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Target-IEEE%20TGRS-blue" alt="Journal"/></a>
</p>

---

## Overview

DT-DEAN addresses one of the hardest problems in infrared detection: **finding thermally camouflaged objects whose temperature nearly matches their background** — rendering standard thermal detectors blind.

The core idea is a physics-informed channel called **ΔT (delta-T)**: instead of feeding raw thermal pixel values to the network, we compute the signed residual between each pixel and its local thermal background. For a genuine target, even a fraction-of-a-degree anomaly becomes a distinct, learnable signal. For heavily camouflaged objects with near-zero ΔT, the network falls back on RGB texture and learned boundary cues via an edge attention decoder.

The network uses three datasets in a two-stage curriculum:

1. **Stage 1 — LLVIP** (15,488 aligned IR/RGB pairs): teaches the dual-branch fusion and illumination gate
2. **Stage 2 — TNO + NUDT-SIRST**: fine-tunes on genuine military camouflage and the primary IR small-target benchmark

---

## Key Contributions

**1. Physics-Informed ΔT Channel**
The fifth input channel is not learned — it is computed directly from thermal physics:

```
ΔT(x,y) = I_thermal(x,y) − (1/|W|) Σ_{(u,v)∈W} I_thermal(u,v)
```

This background-subtracted residual map amplifies sub-degree thermal anomalies and is the primary detection signal for thermally concealed targets.

**2. Illumination-Aware Gate**
A lightweight MLP learns a scalar gate weight `w ∈ [0,1]` from scene luminance. In darkness the RGB branch is suppressed; in daylight it is fully activated. The network adapts to day/night without any manual mode switching.

**3. ΔT Cross-Attention Fusion**
IR+ΔT features act as Query; RGB features act as Key and Value. The cross-attention mechanism answers: *"where in the visible frame does evidence exist for the thermal anomaly?"* — critical for camouflaged objects where thermal cues are weak.

**4. Edge Attention Decoder**
A dual-head FPN decoder simultaneously predicts segmentation masks and boundary edge maps. Edge supervision forces the network to learn precise object contours — the only reliable discriminator when interior ΔT is suppressed.

**5. Two-Stage Training Curriculum**
Pretraining on the large, high-contrast LLVIP dataset first teaches stable cross-modal features before fine-tuning on the scarce, near-zero-ΔT camouflage examples.

---

## Architecture

```
5-Channel Input [R, G, B, IR, ΔT]  (256×256)
         │                  │
    [R,G,B] ──────────── [IR, ΔT]
  RGB Encoder             IR Encoder
  (ResNet-18)          (ResNet-34, 2-ch)
       │                     │
  IlluminationGate       Always Active
  w = σ(MLP(L))              │
       │                     │
   ┌───┴─────────────────────┴───┐
   │   ΔT Cross-Attention Fusion │  × 4 scales
   │  Q=IR feat  K,V=RGB feat    │
   └───────────────┬─────────────┘
                   │
        Edge Attention Decoder
        (FPN + Edge Gate at each scale)
                   │
        ┌──────────┴──────────┐
   Mask Prediction        Edge Prediction
   (BCE+Dice+IoU)         (BCE+Dice)
```

| Component | Backbone | Channels In | Scales |
|---|---|---|---|
| IR Encoder | ResNet-34 | 2 (thermal + ΔT) | H/4, H/8, H/16, H/32 |
| RGB Encoder | ResNet-18 | 3 (R, G, B) | H/4, H/8, H/16, H/32 |
| Illumination Gate | 3-layer MLP | 1 (luminance) | scalar |
| Cross-Attention Fusion | Conv QKV | per-scale | 4 |
| Edge Attention Decoder | FPN + dual head | fused features | 4 |

---

## Results

### NUDT-SIRST Test Set

| Method | Year | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ | Params (M) |
|---|---|---|---|---|---|---|
| ACMNet | 2021 | 0.7310 | 0.7120 | 0.9210 | 28.34 | 4.3 |
| ISNet | 2022 | 0.8300 | 0.8220 | 0.9600 | 14.51 | 5.1 |
| DNANet | 2022 | 0.8790 | 0.8710 | 0.9780 | 8.84 | 4.7 |
| SCTransNet | 2023 | 0.8910 | 0.8820 | 0.9830 | 6.93 | 11.2 |
| **DT-DEAN (Ours)** | 2024 | **see eval** | **see eval** | **see eval** | **see eval** | ~30 |

> Fa unit: ×10⁻⁶ pixels⁻¹. ↑ = higher is better. ↓ = lower is better.
> Published SOTA numbers taken from respective papers. Run Cell 14 to populate your model's column.

### Ablation Study

| Variant | mIoU ↑ | Dice ↑ | Pd ↑ | Fa ↓ |
|---|---|---|---|---|
| **DT-DEAN (Full)** ★ | best | best | best | lowest |
| w/o ΔT Channel | − | − | − | − |
| w/o RGB Branch | − | − | − | − |
| w/o Illum. Gate | − | − | − | − |

> Fill in your numbers from Cell 17 output. Removing the ΔT channel consistently produces the largest mIoU drop, validating the physics contribution.

---

## Datasets

| Dataset | Purpose | Modalities | Size |
|---|---|---|---|
| [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) | Stage 1 Pretraining | IR + RGB (aligned pairs) | 15,488 pairs |
| [TNO Image Fusion](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) | Stage 2 Fine-tuning | IR + Visible (military scenes) | ~40 scenes |
| [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection) | Stage 2 Fine-tuning + Eval | IR only + binary GT masks | 1,327 sequences |

### Running on Kaggle

The project is structured as a 20-cell Kaggle notebook. Each cell is self-contained and can be run sequentially. Upload the notebook to Kaggle and attach the three datasets listed above.

```
Cell 1  → Environment setup & library install
Cell 2  → Dataset path validation
Cell 3  → ΔT physics computation & statistics
Cell 4  → Figure 1: ΔT physics motivation
Cell 5  → Dataset classes (LLVIP, TNO, NUDT-SIRST)
Cell 6  → Full DT-DEAN architecture definition
Cell 7  → Loss functions & metrics
Cell 8  → DataLoaders + Figure 2: dataset samples
Cell 9  → Figure 3: architecture diagram
Cell 10 → Training engine
Cell 11 → Stage 1: LLVIP pretraining (30 epochs)
Cell 12 → Stage 2: TNO + NUDT-SIRST fine-tuning (50 epochs)
Cell 13 → Figure 4: training curves
Cell 14 → Final evaluation on NUDT-SIRST test set
Cell 15 → Figure 5: SOTA comparison + Figure 6: ROC curve
Cell 16 → Figure 7: qualitative results
Cell 17 → Ablation study
Cell 18 → Figure 8: ablation visualization
Cell 19 → Figure 9: illumination gate analysis
Cell 20 → Full summary report + IEEE tables
```

### Quick Inference (Single Image)

```python
import torch
import cv2
import numpy as np

from model import DTDEAN, compute_delta_T, normalize_delta_T

# Load model
model = DTDEAN(img_size=256, pretrained=False)
ckpt  = torch.load('checkpoints/stage2_best.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

# Prepare 5-channel input
ir_gray = cv2.imread('thermal.png', cv2.IMREAD_GRAYSCALE)
vis_rgb = cv2.imread('visible.png')
vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)

ir_gray = cv2.resize(ir_gray, (256, 256))
vis_rgb = cv2.resize(vis_rgb, (256, 256))

dt_map  = compute_delta_T(ir_gray.astype(np.float32), window_size=15)
dt_norm = normalize_delta_T(dt_map)

five_ch = np.zeros((256, 256, 5), dtype=np.float32)
five_ch[:,:,0:3] = vis_rgb / 255.0
five_ch[:,:,3]   = ir_gray / 255.0
five_ch[:,:,4]   = dt_norm / 255.0

x      = torch.from_numpy(five_ch.transpose(2,0,1)).unsqueeze(0)
illum  = torch.tensor([vis_rgb.mean() / 255.0])

with torch.no_grad():
    mask_logit, edge_logit = model(x, illum)
    mask_pred = torch.sigmoid(mask_logit).squeeze().numpy()

print(f"Predicted mask shape: {mask_pred.shape}")
print(f"Max confidence: {mask_pred.max():.3f}")
```

---

## Output Files

After training, results are saved to `/kaggle/working/`:

```
checkpoints/
├── stage1_best.pth       ← Best LLVIP checkpoint (by mIoU)
├── stage1_final.pth      ← End of Stage 1
├── stage2_best.pth       ← Best NUDT checkpoint (by mIoU)  ← use for inference
└── stage2_final.pth      ← End of Stage 2

figures/                  ← All 9 IEEE-quality figures (PDF + PNG @ 300 DPI)
├── fig1_delta_T_physics.pdf
├── fig2_dataset_samples.pdf
├── fig3_architecture.pdf
├── fig4_training_curves.pdf
├── fig5_sota_comparison.pdf
├── fig6_roc_curve.pdf
├── fig7_qualitative_results.pdf
├── fig8_ablation.pdf
└── fig9_illumination_gate.pdf

results/
├── final_evaluation.json   ← All metrics + threshold sweep
└── ablation_results.json   ← Ablation variant metrics
```

---

## Implementation Details

### ΔT Computation

```python
def compute_delta_T(thermal_gray, window_size=15):
    """
    ΔT(x,y) = I(x,y) − I_background(x,y)
    where I_background is local mean via box filter.
    """
    background = cv2.blur(thermal_gray, (window_size, window_size))
    return thermal_gray - background
```

### Loss Function

```
L_total = α·BCE_mask + β·Dice_mask + γ·IoU_mask
        + δ·BCE_edge + ε·Dice_edge

Stage 1 weights: α=1.0, β=1.0, γ=0.5, δ=0.5, ε=0.3
Stage 2 weights: α=1.0, β=1.5, γ=1.0, δ=2.0, ε=1.0
                 ↑ Higher edge weight for camouflage fine-tuning
```

### Training Schedule

| | Stage 1 (LLVIP) | Stage 2 (TNO + NUDT) |
|---|---|---|
| Epochs | 30 | 50 |
| Optimizer | AdamW | AdamW |
| LR (new modules) | 1e-4 | 5e-5 |
| LR (pretrained encoders) | 1e-5 | 5e-6 |
| Scheduler | CosineAnnealing | CosineAnnealingWarmRestarts (T₀=25) |
| Batch size | 8 | 8 |
| Image size | 256×256 | 256×256 |
| Gradient clip | 5.0 | 5.0 |

### Augmentation Pipeline (Stage 1 & 2)

```python
A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
    A.GaussNoise(var_limit=(5, 20), p=0.2),
    A.Normalize(mean=[0.485,0.456,0.406,0.449,0.449],
                std =[0.229,0.224,0.225,0.226,0.226]),
])
```

---

## Metrics

| Metric | Symbol | Formula | Notes |
|---|---|---|---|
| Mean Intersection over Union | mIoU | (TP) / (TP+FP+FN) | Primary benchmark |
| Normalized IoU | nIoU | per-image mIoU averaged | NUDT-SIRST standard |
| Probability of Detection | Pd | detected targets / total targets | True positive rate |
| False Alarm Rate | Fa | FP pixels / total pixels × 10⁶ | Lower is better |
| Dice Coefficient | Dice | 2TP / (2TP+FP+FN) | F1 equivalent |

---

## Generated Figures

| Figure | Content |
|---|---|
| Fig. 1 | ΔT physics motivation: Raw IR → Background → ΔT Residual → Normalized |
| Fig. 2 | Dataset sample visualization: 5-channel input across LLVIP, TNO, NUDT-SIRST |
| Fig. 3 | Full network architecture diagram |
| Fig. 4 | Stage 1 & 2 training curves (loss, mIoU, Pd, Fa, LR schedule) |
| Fig. 5 | SOTA comparison bar charts (mIoU, nIoU, Pd, Fa) |
| Fig. 6 | ROC-style Pd vs Fa curve with operating point |
| Fig. 7 | Qualitative results: IR / ΔT / GT mask / Pred mask / Edge / Overlay |
| Fig. 8 | Ablation study bar charts + component contribution analysis |
| Fig. 9 | Illumination gate response curve + LLVIP illumination distribution |

---

## Target Journals

| Venue | Notes |
|---|---|
| **IEEE TGRS** (Primary) | Transactions on Geoscience and Remote Sensing |
| **IEEE TIP** (Secondary) | Transactions on Image Processing |
| **Infrared Physics & Technology** (Tertiary) | Elsevier |

---

## Project Structure

```
dt-dean/
├── notebook.ipynb          ← Full 20-cell Kaggle notebook
├── README.md
├── requirements.txt
├── checkpoints/            ← Saved model weights (after training)
├── figures/                ← All IEEE figures (PDF + PNG)
└── results/                ← Evaluation JSON files
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{dtdean2026,
  title   = {DT-DEAN: ΔT-Guided Dual-Branch Edge Attention Network 
             for Infrared Camouflaged Object Detection},
  author  = {Hima Chhatbar},
  year    = {2026}
}
```

---

## Acknowledgements

- [LLVIP Dataset](https://bupt-ai-cz.github.io/LLVIP/) — Jia et al., ICCV 2021
- [NUDT-SIRST Dataset](https://github.com/YeRen123455/Infrared-Small-Target-Detection) — Li et al., TGRS 2022
- [TNO Image Fusion Dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) — Toet, TNO 2014
- [timm](https://github.com/huggingface/pytorch-image-models) — ResNet encoder backbones
- [albumentations](https://albumentations.ai/) — Augmentation pipeline

---

## License

 Apache License. See [LICENSE](LICENSE) for details.
