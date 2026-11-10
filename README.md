# Conditional microstructure processing response

Self-contained simulation, conditional flow-matching, sensitivity estimation and numerical analysis for Gaussian random fields, Cahn–Hilliard phase separation and Potts grain growth.

This is the **research-code package**. Manuscript text, LaTeX sources, typesetting/figure-layout scripts and trained checkpoints are excluded. The manuscript is distributed separately as an Overleaf archive. Numerical summaries and selected compact sample arrays are included.

## Installation

Use Python 3.11 or newer. Install a CUDA-enabled PyTorch build appropriate for your machine using the [official installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```text
python -m pip install -r requirements.txt
```

GPU simulations and training require an NVIDIA CUDA device. Reading summaries, the supplied result plot and the CPU checks do not require trained models. The local tested environment used PyTorch 2.14.1+cu132, NumPy 2.4.4, SciPy 1.18.0, Matplotlib 3.11.0 and Pillow 12.2.0. The minimum-version requirements are not an exhaustive compatibility claim.

The `code/genmicro` runtime is bundled; no neighbouring checkout, absolute user path or separately downloaded data are required. See `docs/runtime_validation.md` for the checkpoint regression check used when making it standalone.

## Quick checks and saved-results plot

From the extracted package root:

```text
python -m unittest discover -s tests -v
python code/plot_contrast_control.py
```

The plot is written into `figures/`, a generated output folder. It reads the included numerical control result and does not load a checkpoint.

## Reproduce the stronger-response control

```text
python code/contrast_control.py train
python code/contrast_control.py measure
python code/plot_contrast_control.py
```

This runs paired sinusoidal/direct models on the same Gaussian family with ring amplitude 0.8 and frequency 8, one image per uniformly sampled setting and one training pool. Initialization seed is 3400. The original nested sizes 16–512 and training-step schedule are retained. A sweep is skipped only if both its result and checkpoint already exist. Because checkpoints are omitted here, the training command regenerates them and rewrites the corresponding numerical result. Use a copy of this package if you want to preserve the distributed result files unchanged. Model files and simulator caches are generated locally.

PyTorch compilation is enabled by default. If your platform lacks a configured compiler, set `GENMICRO_COMPILE=0` before starting Python to use eager execution. For example, in PowerShell use `$env:GENMICRO_COMPILE='0'`; in a POSIX shell prefix the command with `GENMICRO_COMPILE=0`.

## Other experiments and analyses

`systems.py`, `grf.py` and `potts.py` define the systems. `study.py` performs nested training sweeps, for example `python code/study.py gA:0 gA_raw:0`. `fisher_v5.py` implements exact, denoiser-based and descriptor estimates; `geometry_v5.py` measures them for saved models. Original numerical analyses are `analyze_v3.py`, `analyze_v4.py` and `analyze_v5.py`. They consume the included results; model-dependent measurements require rerunning training first. The legacy stage names identify separate experiments and are preserved to avoid changing recorded measurements.

`results/` contains numerical measurements, summaries and sample arrays. Large `.pt` files (trained weights, reference-image caches and intermediate visualization tensors) are omitted. Original and stronger-response experiments have different names and result files. No original experiment is relabelled as the stronger control.

Manual GitHub upload instructions are in `docs/GITHUB_UPLOAD.md`. Upload the extracted files, not this ZIP, and keep the separate Overleaf package out of the code repository.
