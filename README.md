<div align="center">

# Camouflaged Object Detection

### Thermal Edge Attention for Camouflaged Object Detection in Infrared Imagery (IR-COD)

*A delta-T–aware extension of YOLOv8-seg for detecting low-thermal-contrast objects in IR scenes*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research--prototype-yellow.svg)]()

</div>

---

## Overview

Most infrared object detectors assume a target is always thermally distinct from its background. In practice, that assumption breaks down — a person standing against a sun-warmed wall, or a vehicle that has cooled to ambient temperature, can be nearly invisible in raw thermal intensity even though it's plainly there.

**IR-COD** addresses this by giving the model an explicit signal for *thermal contrast*, not just thermal intensity, and an attention mechanism that learns to look harder exactly where that contrast disappears.

The pipeline:
1. Decomposes every IR frame into a **raw LWIR channel** and a **delta-T (ΔT) channel** — the local thermal anomaly after Gaussian background subtraction.
2. Uses **SAM2** to lift box-level annotations (LLVIP, FLIR) up to pixel-level instance masks.
3. Feeds both channels into a modified **YOLOv8-seg** backbone, with a **Thermal Edge Attention (TEA)** module inserted at the FPN neck that's explicitly driven by ΔT.
4. Trains with a composite loss that upweights pixels where ΔT is near zero — the hardest, most camouflage-like regions.
5. Evaluates with standard camouflaged-object-detection metrics (S-measure, weighted F-measure, MAE, E-measure) and a custom **ΔT sensitivity curve** that isolates how much of the model's advantage comes specifically from handling low-contrast targets.

---

## Why this matters

Thermal cameras are widely used in surveillance, search-and-rescue, autonomous driving, and defense — precisely the settings where a missed detection has real cost. Conventional IR detectors are tuned for the easy case: high-contrast, thermally salient targets. This project asks a narrower, more useful question: **how much worse does a detector get as an object's thermal contrast drops toward the noise floor, and can architecture-level changes slow that decline?**

---

## Method

### 1. Delta-T Decomposition

For every input image:

```
background = GaussianBlur(LWIR, sigma = 15)
ΔT         = LWIR - background
```

A large blur kernel captures the slowly varying thermal background (walls, ground, sky) while ignoring local detail. ΔT is therefore the *local thermal anomaly* — near zero where an object blends into its surroundings, large where it stands out. Each sample becomes a `[2, H, W]` tensor: raw LWIR + normalized ΔT.

### 2. Pixel-Level Annotation via SAM2

LLVIP and FLIR ship with bounding boxes only. Each box is used as a prompt for **SAM2**, producing an instance mask. A per-instance **contrast score** — mean `|ΔT|` inside the mask — is computed and stored alongside the YOLO-seg label, which downstream stages use to identify hard, low-contrast instances.

> **Note on KAIST:** the KAIST split used in this project ships without annotation files, so it is used unsupervised — for delta-T distribution calibration and as additional unlabeled context — rather than for direct training supervision.

### 3. Architecture — Thermal Edge Attention (TEA)

- The first convolution of YOLOv8m-seg is modified to accept **2 input channels** (LWIR + ΔT) instead of 3 (RGB), with weights initialized from the averaged original RGB filters.
- **TEA** sits at the FPN neck via a forward hook. It computes a spatial edge map (`1 − |ΔT|`, so near-zero ΔT becomes high activation), fuses it with channel-attentioned semantic features, and adds the result back as a residual — sharpening the network's attention exactly at thermal-camouflage boundaries.
- A composite loss (`L_mask + L_edge + L_dt`) combines standard segmentation BCE with a boundary-aware term and a ΔT-weighted consistency term that explicitly rewards correct predictions in low-contrast regions.

### 4. Curriculum Training

Training proceeds in three stages: backbone-only warm-up with TEA frozen, full joint training with TEA and the ΔT loss active, then low-LR fine-tuning — letting the model learn general IR features before being asked to specialize on thermal-boundary attention.

### 5. Evaluation

- **Standard COD metrics:** S-measure, weighted F-measure, MAE, E-measure, computed per-dataset and overall.
- **ΔT Sensitivity Curve:** test instances are binned by contrast score (high to low), and performance is compared between the full model and a ΔT-blind ablation. A larger gap in the low-contrast bins is the direct evidence that ΔT input + TEA specifically help with hard, low-thermal-contrast targets — rather than the model simply being better overall.
- **Ablation study** isolating the contribution of the ΔT input channel, the TEA module, and the ΔT-weighted loss term independently.

---

## Datasets

| Dataset | Role | Annotation format | Used for |
|---|---|---|---|
| [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) | Train / Test | VOC-XML (bounding boxes) | Supervised training, SAM2 prompting |
| [FLIR ADAS v2](https://www.flir.com/oem/adas/adas-dataset-form/) | Train / Val | COCO JSON | Supervised training, SAM2 prompting |
| [KAIST Multispectral Pedestrian](https://soonminhwang.github.io/rgbt-ped-detection/) | Calibration | Unannotated (this split) | ΔT distribution calibration, unsupervised context |

Datasets are not included in this repository due to size and license restrictions. See [`docs/DATASETS.md`](docs/DATASETS.md) for download links and the exact directory layout expected by the data loaders.

---

## Repository structure

```
Camouflaged_Object_Detection/
├── notebooks/
│   └── ir_cod_pipeline.ipynb      # End-to-end Kaggle/Colab notebook (all stages)
├── src/
│   ├── delta_t.py                 # ΔT decomposition + visualization
│   ├── dataset_parsers.py         # LLVIP / FLIR / KAIST loaders
│   ├── sam2_annotate.py           # Box-to-mask pipeline via SAM2
│   ├── model.py                   # 2-channel YOLOv8 + TEA module + composite loss
│   ├── dataset.py                 # Dataloader, augmentation, weighted sampling
│   ├── train.py                   # 3-stage curriculum training loop
│   ├── evaluate.py                # S-measure, wFm, MAE, E-measure
│   ├── sensitivity_curve.py       # ΔT-binned performance analysis
│   └── ablation.py                # Component-wise ablation study
├── figures/                       # Generated IEEE-style figures (per-figure exports)
├── configs/
│   └── config.yaml                # Hyperparameters, paths, dataset weights
├── docs/
│   └── DATASETS.md                # Dataset download + directory setup instructions
├── LICENSE
└── README.md
```

---

## Getting started

### Requirements

- Python 3.10+
- CUDA-capable GPU (developed and tested on Kaggle T4 ×2)
- `ultralytics`, `albumentations`, `segment-anything-2`, `torch`, `opencv-python`

```bash
pip install ultralytics albumentations opencv-python torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Running the pipeline

The full pipeline is designed to run end-to-end on Kaggle (GPU T4 ×2, internet on) or any CUDA machine with equivalent memory:

```bash
python src/sam2_annotate.py     # generate pixel masks + contrast scores
python src/train.py             # 3-stage curriculum training
python src/evaluate.py          # compute COD metrics
python src/sensitivity_curve.py # ΔT sensitivity analysis
python src/ablation.py          # ablation study
```

Configuration (paths, hyperparameters, dataset weights) lives in `configs/config.yaml` — update dataset paths there before running.

---

## Results

> Fill in once training and evaluation are finalized.

| Model variant | S-measure ↑ | wFm ↑ | MAE ↓ | E-measure ↑ |
|---|---|---|---|---|
| Baseline (1-ch, no TEA) | — | — | — | — |
| + ΔT input (2-ch) | — | — | — | — |
| + TEA | — | — | — | — |
| Full model (2-ch + TEA + L_dt) | — | — | — | — |

See [`figures/`](figures/) for the full ΔT sensitivity curve, training curves, qualitative detections, and per-dataset breakdowns generated for the manuscript.

---

## Citation

If you use this work, please cite:

```bibtex
@article{yourname2026ircod,
  title   = {Thermal Edge Attention for Camouflaged Object Detection in Infrared Imagery},
  author  = {Your Name},
  journal = {TBD},
  year    = {2026}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

Note: this license covers the **code** in this repository only. LLVIP, FLIR, and KAIST retain their own respective dataset licenses and usage terms — refer to each dataset's original distribution terms before use.

---

<div align="center">

*Final-year research project · IEEE-style camouflaged object detection in infrared imagery*

</div>
