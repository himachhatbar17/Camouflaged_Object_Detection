# DT-DEAN

**ΔT-Guided Dual-Branch Edge Attention Network for Infrared Camouflaged Object Detection**

---

## Overview

DT-DEAN detects thermally concealed objects by computing a physics-informed residual — the difference between a pixel's thermal emission and its local background. This ΔT signal forms the fifth input channel alongside RGB and raw IR, enabling detection where temperature contrast is near-zero.

```
Input: [R, G, B, IR, ΔT]  →  Mask + Edge predictions
```

The network uses two parallel encoders — ResNet-34 for IR+ΔT, ResNet-18 for RGB — fused at four scales via cross-attention. An illumination gate learns to suppress the RGB branch in darkness, keeping the IR branch dominant at night.

---

## Architecture

```
[R, G, B]  →  RGB Encoder (ResNet-18)  ─┐
                                          ├─ ΔT Cross-Attention × 4 ─→ Edge Attention Decoder ─→ Mask
[IR, ΔT]   →  IR Encoder  (ResNet-34)  ─┘                                                      → Edge

Illumination Gate:  w = σ(MLP(luminance))  ∈ [0, 1]
```

**ΔT computation:**
```
ΔT(x, y) = I_thermal(x, y) − (1/|W|) Σ I_thermal(u, v)
```
where W is a 15×15 local window.

---

## Datasets

| Dataset | Role | Samples |
|---|---|---|
| LLVIP | Stage 1 pretraining | IR–RGB pairs |
| TNO Image Fusion | Stage 2 fine-tuning | Multi-sensor military scenes |
| NUDT-SIRST | Primary benchmark | Near-zero ΔT targets |

NUDT-SIRST split: 80/20 on 60 sample folders, `seed=42`, exact mask matching only.

---

## Training

Two-stage pipeline:

**Stage 1 — Cross-modal pretraining on LLVIP**
```
Epochs: 30  |  LR: 1e-4  |  Scheduler: CosineAnnealing
Loss: BCE + Dice + IoU (mask) + BCE + Dice (edge)
```

**Stage 2 — Fine-tuning on TNO + NUDT-SIRST**
```
Epochs: 50  |  LR: 5e-5  |  Scheduler: CosineAnnealingWarmRestarts
```

---

## Results on NUDT-SIRST

| Method | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ |
|---|---|---|---|---|
| ACMNet (2021) | 0.731 | 0.712 | 0.921 | 28.34 |
| ISNet (2022) | 0.830 | 0.822 | 0.960 | 14.51 |
| DNANet (2022) | 0.879 | 0.871 | 0.978 | 8.84 |
| SCTransNet (2023) | 0.891 | 0.882 | 0.983 | 6.93 |
| **DT-DEAN (ours)** | — | — | — | — |

*Fa unit: ×10⁻⁶ px⁻¹. Split may differ from published protocols.*

---

## Ablation

| Variant | ΔmIoU |
|---|---|
| Full model | — |
| w/o ΔT channel | − |
| w/o RGB branch | − |
| w/o Illumination Gate | − |

---

## Environment

```bash
pip install timm segmentation-models-pytorch albumentations opencv-python-headless
```

Tested on NVIDIA P100 16GB · PyTorch AMP · Batch size 4

---

## Figures

| | |
|---|---|
| Fig 1 | ΔT physics motivation |
| Fig 2 | Dataset sample visualization |
| Fig 3 | Architecture diagram |
| Fig 4 | Training curves |
| Fig 5 | SOTA comparison |
| Fig 6 | ROC curve (Pd vs Fa) |
| Fig 7 | Qualitative results |
| Fig 8 | Ablation study |
| Fig 9 | Illumination gate analysis |

---

*Target venues: IEEE TGRS · IEEE TIP · Infrared Physics and Technology*

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
