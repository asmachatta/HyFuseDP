# HyFuseDP: A Hybrid Deep Learning Approach for Automated Date Palm Disease Recognition

This repository contains the core training and evaluation code for the paper:

> **HyFuseDP: A hybrid deep learning approach for automated date palm disease recognition**
> A. Chatta, B. Latreche, Y. Ghibeche, M. Bouakkaz, D. Sahraoui, A. A. Mir.

HyFuseDP combines two lightweight backbones — **RepVGG-A0** (reparameterized
convolutions, local texture) and **PoolFormer-S12** (pooling-based token mixing,
global context) — through **automatically optimized weighted soft voting**, with an
**XGBoost** classifier applied to the fused probability vectors.

---

## Reproducibility at a glance

All experiments use a **single fixed seed (42)** and a **stratified 70/15/15
train/validation/test split**. Every reported metric is computed **once on the
held-out 15% test partition**, which is never used for backbone training, fusion
weight selection, or model tuning. Fusion weights (alpha = 0.56, beta = 0.44) are
selected by grid search on the **Namoun validation split only** and transferred
unchanged to the Date Palm dataset to assess cross-dataset generalization.

---

## Datasets

| Dataset | Classes | Images | Conditions | Source |
|---|---|---|---|---|
| Namoun | 9 | 3,089 | Field (uncontrolled) | Namoun et al., *Data in Brief* 110933 (2024) |
| Date Palm | 3 | 2,631 | Laboratory (controlled) | Hamaidi, Kaggle (2025) |

Both are publicly available. Set the dataset path via the `INPUT_PATH` variable (or
environment variable) at the top of the script.

---

## Contents

This repository provides the core HyFuseDP pipeline:

| Script | Purpose | Paper outputs reproduced |
|---|---|---|
| `HyFuseDP_singleseed.py` | Full HyFuseDP pipeline: fine-tunes both backbones, performs weighted soft-vote fusion + XGBoost, runs the soft-voting-only ablation, McNemar's significance test, Grad-CAM visualization, and efficiency profiling | HyFuseDP and ablation rows of the comparison table; McNemar table; per-class report; confusion matrices; ROC curves; Grad-CAM figures; efficiency table |

> **Scope note.** This repository releases the core proposed method. The additional
> baseline rows reported in the paper (matched-protocol deep baselines and classical
> raw-pixel classifiers) were produced under the identical 70/15/15 protocol and seed
> described above; the scripts for those baselines can be made available on request.

---

## Environment

```bash
python >= 3.10
pip install torch torchvision timm scikit-learn xgboost pillow numpy matplotlib
```

Experiments were run on Kaggle (NVIDIA Tesla P100 / T4 GPUs). A CUDA-capable GPU is
recommended; the code falls back to CPU automatically.

---

## How to run

1. **Set the dataset path.** Edit `INPUT_PATH` in the script, or export it:
   ```bash
   export INPUT_PATH="/path/to/Namoun"      # or the Date Palm folder
   ```

2. **Run the full pipeline:**
   ```bash
   python HyFuseDP_singleseed.py
   ```

Run it once with `INPUT_PATH` pointing at the Namoun folder and once at the Date
Palm folder to reproduce the results for both datasets.

---

## Notes on protocol

The two backbones are fine-tuned under an identical protocol (same split and seed,
AdamW, cosine annealing, mild augmentation, batch size 32, early stopping on the
validation split). The fusion weights are selected on the Namoun validation split
and transferred unchanged to the Date Palm dataset.

The two datasets differ in difficulty: HyFuseDP yields a statistically significant
improvement on the hard **Namoun field** data, whereas on the controlled
**Date Palm** data performance saturates and the fusion offers no advantage over the
strongest single models. Both outcomes are reported as observed.

---

## License

Released under the MIT License. See `LICENSE`.
