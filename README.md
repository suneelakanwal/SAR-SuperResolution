# SAR-SuperResolution# End-to-end super-resolution experimentation for **Sentinel-1 SAR (grayscale) imagery** using multiple model families:


End-to-end super-resolution experimentation for **Sentinel-1 SAR (grayscale) imagery** using multiple model families:

- **SRGAN / SRResNet** (adversarial + perceptual training)
- **ESRGAN** (RRDB generator + GAN fine-tuning)
- **EDSR** (high-PSNR baseline with custom grayscale patches)
- **SRCNN** (lightweight convolutional baseline)

The repository is primarily notebook-driven and includes preprocessing workflows, training/evaluation notebooks, and exported visual artifacts.

---

## Overview

This project explores how well deep super-resolution methods reconstruct high-resolution SAR texture from low-resolution inputs.

The workflow generally follows:

1. Prepare SAR patches and train/eval HDF5 datasets
2. Train baseline and GAN-based SR models
3. Evaluate PSNR/SSIM and visual quality
4. Compare outputs and summarize findings

Most experiments are designed for **Kaggle GPU environments**, then exported into this repository.

---

## Repository Structure

```text
SRGAN-Model/
├── README.md
├── data_preprocessing/
│   ├── edsr_data_preprocessing.ipynb
│   ├── srgan_data_preprocessing.ipynb
│   ├── esrgan                  # notebook-exported script (no .py extension)
│   └── srcnn                   # notebook-exported script (no .py extension)
├── models/
│   ├── esrgan.py               # notebook-style training script fragment
│   ├── srcnn.py                # notebook-style training script fragment
│   └── sar_swinir.py           # placeholder/empty
├── notebooks/
│   ├── SRGAN.ipynb
│   ├── srgan-experiment.ipynb
│   ├── srgan-phase-2.ipynb
│   ├── srgan_compare_phases.ipynb
│   ├── edsr-experiment.ipynb
│   ├── edsr-test.ipynb
│   ├── edsr-phase-2-test.ipynb
│   ├── edsr-phase-2-test_dummy.ipynb
│   ├── train_esrgan.ipynb
│   └── train_srcnn.ipynb
├── edsr_outputs/
├── srgan_outputs/
└── figures/
```

---

## Data Pipeline

### Input data assumptions

Preprocessing code assumes single-band SAR imagery (grayscale), with examples using:

- `image.tiff` from a Kaggle dataset path
- patch/tile extraction from large scenes

### Typical preprocessing behavior

Across preprocessing notebooks/scripts, you will find:

- percentile clipping + normalization
- tile extraction (example: `256x256`, stride overlap)
- flat-tile filtering (low standard deviation)
- train/eval split
- patch generation for SR pairs
- HDF5 export, commonly:
  - `/kaggle/working/sar_train_x2.h5`
  - `/kaggle/working/sar_eval_x2.h5`

---

## Model Workflows

## 1) SRGAN / SRResNet

Primary notebooks:

- `notebooks/SRGAN.ipynb`
- `notebooks/srgan-experiment.ipynb`
- `notebooks/srgan-phase-2.ipynb`
- `notebooks/srgan_compare_phases.ipynb`

Typical flow in notebooks:

- pretrain SRResNet (`train_net.py` with SRResNet config)
- adversarial fine-tuning (`train_gan.py` with SRGAN config)
- inference and output comparison plots

## 2) EDSR

Primary notebooks:

- `notebooks/edsr-experiment.ipynb`
- `notebooks/edsr-test.ipynb`
- `notebooks/edsr-phase-2-test.ipynb`

Typical flow includes:

- cloning EDSR-PyTorch in notebook runtime
- preparing DIV2K-style folder structure from SAR patches
- grayscale compatibility patches (in-runtime edits to EDSR code)
- training, continued training, and metrics plotting

## 3) ESRGAN

Related artifacts:

- `models/esrgan.py`
- `data_preprocessing/esrgan`
- `notebooks/train_esrgan.ipynb`

The exported ESRGAN code includes:

- RRDB generator (single-channel input/output)
- VGG-style discriminator
- warmup (pixel loss) stage
- GAN fine-tuning stage
- preview plotting for losses + PSNR + patch comparisons

## 4) SRCNN

Related artifacts:

- `models/srcnn.py`
- `notebooks/train_srcnn.ipynb`

The SRCNN workflow is a lightweight baseline with HDF5-based loading and standard PSNR tracking.

---

## Output Artifacts

The repo includes many pre-generated figures and comparison files, such as:

- `edsr_outputs/`
  - `edsr_2x_comparison.png`
  - `edsr_2x_metrics.png`
  - `edsr_full_summary.png`
  - `edsr_vs_bicubic.png`
- `srgan_outputs/`
  - `comparison.png`
  - `full_comparison.png`
  - `sr_result.png`
- `figures/`
  - dataset and patch visualizations
  - curve plots
  - SAR exploratory charts

These are useful for quick qualitative review without rerunning notebooks.

---

## Environment Setup

> Note: there is currently no pinned `requirements.txt` or `environment.yml` in this repository.

A practical local setup (Linux/macOS) is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib h5py scikit-image scikit-learn tqdm pillow ipython jupyter
```

Some notebooks may also need OpenCV:

```bash
pip install opencv-python
```

---

## Running Experiments

Since experiments are notebook-first, start with Jupyter:

```bash
jupyter lab
```

Recommended order:

1. Data prep notebook (`data_preprocessing/*.ipynb` or exported scripts)
2. Baseline training (SRCNN or EDSR)
3. GAN training (SRGAN / ESRGAN)
4. Comparison notebook (`notebooks/srgan_compare_phases.ipynb`)

---

## Important Notes / Caveats

- `models/esrgan.py` and `models/srcnn.py` are **not clean standalone modules**; they are notebook-exported scripts with runtime assumptions.
- `data_preprocessing/esrgan` and `data_preprocessing/srcnn` are files without `.py` extension (still readable as Python-like notebook exports).
- Several notebooks assume Kaggle-style absolute paths (`/kaggle/working/...`, `/kaggle/input/...`).
- `models/sar_swinir.py` is currently empty/placeholder.

---

## Reproducibility Tips

- Replace hardcoded Kaggle paths with configurable local paths first.
- Add a pinned dependency manifest (`requirements.txt` or `environment.yml`).
- Split notebook-exported code into importable modules under `models/` and `data_preprocessing/`.
- Store experiment config/version metadata with each result figure.

---

## Suggested Next Improvements

- Add a proper training CLI (`train.py`) with YAML configs
- Centralize metrics computation (PSNR, SSIM, optional LPIPS)
- Add checkpoint + artifact registry per experiment run
- Add unit tests for dataset loading and patch shape invariants

---

## Acknowledgment

This project builds on common SR research pipelines and uses open-source model implementations (notably EDSR/SRGAN-style workflows) adapted for grayscale SAR experimentation.

Thanks to the Kaggle community for providing GPU resources and datasets that enabled this exploration!

This project was developed in close collaboration with @marywagura.
