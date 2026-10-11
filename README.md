# Nucleus Segmentation in Fluorescence Microscopy

A benchmark of **8 heavyweight ResNet-family U-Net backbones** for nucleus instance segmentation on the [BBBC039](https://bbbc.broadinstitute.org/BBBC039) dataset — from pixel-level accuracy down to the metrics a biologist actually cares about, plus full interpretability analysis of the winning architecture.

## Research Question

High-throughput screening (HTS) runs thousands of biological experiments automatically and produces imaging data far beyond what humans can annotate by hand. The nucleus is the primary anchor point for analyzing a cell: segmenting each nucleus creates a **biological coordinate system** that enables cell counting, morphological measurements (nucleus swelling or fragmentation → cell death), and extraction of localized fluorescence intensities of targeted protein biomarkers for phenotypic profiling.

**Which ResNet backbone U-Net segments nuclei best?** The project compares:

| Family | Backbones |
|---|---|
| ResNet | ResNet-50, ResNet-101 |
| Wide-ResNet | Wide-ResNet-50, Wide-ResNet-101 |
| ResNeXt | ResNeXt-50, ResNeXt-101 |
| SE-ResNeXt | SE-ResNeXt-50, SE-ResNeXt-101 |

## Dataset

- **Source:** [BBBC039](https://bbbc.broadinstitute.org/BBBC039) — DAPI-stained nuclei, fluorescence microscopy, with RGBA annotation masks.
- **Preprocessing:** percentile-based normalization, multi-head target construction (background / boundary / interior channels).
- **Split:** 50 / 25 / 25 train–val–test, seed 42.
- **Augmentation:** random crops, flips, noise, elastic transforms (`albumentations`).
- The notebook downloads and extracts the dataset itself, so nothing needs to be committed to this repo.

## Method

### Architecture

- U-Net with `timm` `features_only` encoders (ImageNet-pretrained) and a 1-channel stem.
- **Multi-head decoder:** predicts 3 channels per pixel — background, boundary, interior.
- **Boundary-subtraction instance segmentation:** instead of watershed, predicted boundaries are subtracted from predicted interiors, isolating each discrete cell core:

```
pred_cores = (interior_prob > 0.5) & ~(boundary_prob > 0.5)
```

This directly addresses the classic failure mode where two touching nuclei merge into one blob — a mistake pixel-level metrics would barely penalize but which ruins cell counts.

### Loss

Combined Binary Cross Entropy + Dice loss to counteract the extreme foreground/background class imbalance:

$$L_{Total} = -\frac{1}{N}\sum_{i=1}^{N} \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right] + \left(1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}\right)$$

### Training Pipeline

- All 8 backbones trained under identical settings (split, loss, LR policy) for a fair comparison.
- **Crash-safe experiment manager:** full checkpoints (weights + optimizer + scheduler + epoch) every epoch, `training_progress.json` manifest, skip-finished / resume-interrupted logic, early stopping (patience 10), `ReduceLROnPlateau`.
- **Colab Drive persistence:** local fast checkpoints synced to Google Drive every 5 epochs — a runtime disconnect costs at most 5 epochs.
- Encoder/decoder get separate optimizer LR groups.

## Evaluation

Models are evaluated at two stages, then on biological and hardware benchmarks:

### Semantic (pixel-level)
- **Dice coefficient (F1)** — industry standard for medical imaging, robust to class imbalance.
- **IoU (Jaccard)** — stricter; penalizes false positives/negatives more harshly.

### Instance (object-level)
- **mAP across IoU thresholds** (0.50 → 0.95, step 0.05) — the 2018 Data Science Bowl metric; over-split nuclei instantly trigger false-positive penalties.

### Biological benchmarks (what a biologist cares about)
- **Absolute cell count error** — how close detected counts are to ground truth.
- **Morphological variance** — Kolmogorov–Smirnov (KS) test on area and eccentricity distributions of predicted cells vs. human annotations.

### Hardware benchmarks
- Training time per model and peak VRAM.
- **Avg prediction time per model (ms/image)** — GPU-synchronized forward-pass timing, measured separately from CPU postprocessing.
- **Winner-vs-loser avg prediction time difference** — the interpretability setup auto-selects the winning (lowest val loss) and losing (highest val loss) models from the training manifest and reports the speed delta.

## Interpretability

"Black box" networks face heavy criticism in biomedical settings — biologists and regulators want to know *why* a prediction was made. Three analyses run automatically on the finished models:

1. **Grad-CAM** — gradient-weighted heatmaps over the original microscopy images, targeting the boundary channel. If the model learned real biology, the heatmap glows as a ring around the nucleus rather than highlighting background noise or sensor artifacts.
2. **Feature-map extraction** (PyTorch hooks) — grayscale plots from three locations: the ResNet stem (edge detectors), the encoder bottleneck (abstract semantic concepts), and the skip connections (high-resolution edges bypassing the bottleneck).
3. **Comparative error analysis** — test images sorted by worst image-level Dice, directly comparing the winning model's failure modes against the losing models to reveal architectural nuances (e.g., failure on dense overlapping clusters).

## Project Structure

```
├── src/
│   ├── Nucleus_Segmentation_in_Fluorescence_Microscopy.ipynb   # Full pipeline (Colab-ready)
│   ├── config.py                                                # Path configuration
│   ├── utils/dirtools.py                                        # File/IO utilities
│   └── dataset/                                                 # BBBC039 (gitignored, auto-downloaded)
├── tests/
│   └── test_checkpointing.py                                    # 4 regression scenarios for the training loop
├── model_evaluation_and_interpretability.md                     # Step-by-step explainer of every evaluation block
└── rationale_and_methodology.txt                                # Research rationale and metric derivations
```

## Quick Start

### Google Colab (recommended — GPU)

1. Open `src/Nucleus_Segmentation_in_Fluorescence_Microscopy.ipynb` in Colab.
2. Run the setup cell (installs dependencies, mounts Google Drive for checkpoints).
3. Choose a path:
   - **Option A — train:** run the data, model, and training sections. Training is fully resumable; re-running picks up where it left off.
   - **Option B — use finished models:** skip training. The loader lists Drive checkpoints and loads the best weights so evaluation and interpretability run without re-training.
4. Run the benchmarking and interpretability cells — the setup cell auto-selects the winning and losing models from `training_progress.json`.

### Local development

```bash
python tests/test_checkpointing.py   # 4/4 regression scenarios for checkpoint/resume/early-stopping logic
```

Dependencies: `torch`, `torchvision`, `timm`, `albumentations`, `tifffile`, `scikit-image`, `opencv-python` (installed automatically by the notebook's setup cell).

## Documentation

- [model_evaluation_and_interpretability.md](model_evaluation_and_interpretability.md) — what each evaluation/interpretability block is, how it works step by step, and why it exists.
- [rationale_and_methodology.txt](rationale_and_methodology.txt) — research motivation and metric derivations.

## Results

> ⏳ Training and evaluation have not yet been run for real (the pipeline logic is implemented and regression-tested). Benchmark results will be posted here once available.
