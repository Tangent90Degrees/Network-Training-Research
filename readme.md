# Network Training — Course Project

Research project for COMP-4471: Deep Learning in Computer Vision.

## Todo

- [ ] Do more experiments
- [ ] Draw the model architecture figure for the paper
- [ ] Make presentation slides

## Research Question

Do the fast-training tricks from [94% on CIFAR-10 in 3.29 Seconds on a Single GPU](docs/2404.00498v2.pdf) transfer to Vision Transformer (ViT) architectures?

The original paper introduces four tricks for CNN training on CIFAR-10:

- Frozen patch-whitening initialisation
- Dirac (identity) initialisation
- ScaleBias output layer + Lookahead optimiser
- Alternating flip augmentation

This project tests each trick on a small ViT (~4M parameters) under both a weak training recipe (100 epochs) and a strong training recipe (300 epochs, RandAugment, Mixup/CutMix, DropPath).

## Key Results

| Trick                          | Weak recipe ($\Delta$) | Strong recipe ($\Delta$) |
| ------------------------------ | ---------------------- | ------------------------ |
| Frozen patch-whitening         | +1.31pp                | −4.39pp                  |
| Alternating flip               | +0.80pp                | −0.82pp                  |
| Soft Dirac init ($\alpha$=0.3) | −1.16pp                | **+0.41pp**              |
| Dirac init (full identity)     | −7.90pp                | −2.22pp                  |
| ScaleBias + Lookahead          | −15.97pp               | −5.03pp                  |
| All methods combined           | −23.63pp               | −9.68pp                  |

Baselines: 61.68% (weak, 100 epochs) and 81.42% (strong, 300 epochs).

The main finding: trick transferability depends on the **trick × recipe interaction**, not just architectural compatibility. Frozen whitening helps under basic augmentation but hurts under strong augmentation (Mixup/CutMix) because the frozen patch embedding is mismatched to heavily transformed inputs.

## Setup

```bash
conda activate vision
```

All experiments use PyTorch 2.5.1 on NVIDIA RTX 4090. CIFAR-10 data is in `datasets/Cifar-10/`.

## Directory Structure

```txt
Network-Training/
├── datasets/
│   └── Cifar-10/           # CIFAR-10 training and test data
├── docs/                   # Documentation and reference paper
├── experiments/
│   ├── common/             # Shared code: model, data, trainer, tricks
│   ├── baseline/           # 100-epoch weak-recipe baseline
│   ├── baseline_improved/  # 300-epoch strong-recipe baseline
│   ├── whitening_v3/       # Frozen patch-whitening (strong recipe)
│   ├── altflip_v3/         # Alternating flip (strong recipe)
│   ├── dirac_v3/           # Full Dirac init (strong recipe)
│   ├── dirac_soft_v3/      # Soft Dirac init α=0.3 (strong recipe)
│   ├── sbl_v3/             # ScaleBias + Lookahead (strong recipe)
│   └── all_methods_v3/     # All tricks combined (strong recipe)
├── external/
│   └── Cifar10CNN/         # Reference CNN codebase
└── reports/
    ├── paper.tex           # LaTeX report
    └── paper.pdf           # Compiled PDF
```

## Running Experiments

Each experiment folder has a `run.py`. From the project root:

```bash
conda run -n vision python experiments/baseline_improved/run.py
```

Results are saved to `experiments/<name>/outputs/results.json` and `best.pt`.

To compare all completed runs:

```bash
conda run -n vision python experiments/common/analyze.py --plot experiments/comparison_v3.png
```

## Report

The full write-up is in [`reports/paper.tex`](reports/paper.tex), compiled to [`reports/paper.pdf`](reports/paper.pdf).
