# Sort, Partition, Randomize

Code for **Sort, Partition, Randomize: Optimal Binary Hypothesis Testing under Local Differential Privacy**  
Elena Ghazi, Jawad Nasser, Flavio Calmon, and Ibrahim Issa  
**NeurIPS 2026 (Spotlight)** | [arXiv version](https://arxiv.org/abs/2606.07443)

This repository reproduces the numerical experiments in Figures 1 and 3 of the NeurIPS paper.

## Run

The following commands, run from this folder, create an environment, install the dependencies, and open the notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab reproduce_figures.ipynb
```

In JupyterLab, **Kernel → Restart Kernel and Run All Cells** runs the complete notebook. The notebook recomputes both experiments, compares every trial with the saved results, and displays and saves the four panels. No LaTeX installation is needed.

The panels can also be generated directly:

```bash
python reproduce_kl_vs_eps_k6_with_spr.py
python reproduce_kl_vs_eps_k100_spr.py
```

## Experiments and outputs

For each alphabet size, the code draws 100 independent pairs from Dirichlet(1), with NumPy seed 0, and evaluates 101 privacy values from 0 to 10. The same pairs are shared across mechanisms and privacy values. At k=6, SPR is compared with the KOV staircase LP, the binary mechanism, and randomized response. At k=100, the LP is omitted because it would have 2^100 variables.

Files in `figures/` are regenerated on each run:

| Paper figure | PDF panels |
| --- | --- |
| Figure 1: mean normalized KL utility | `kl_vs_eps_k6_panel.pdf`, `kl_vs_eps_k100_panel.pdf` |
| Figure 3: pointwise 95% confidence intervals | `kl_vs_eps_k6_panel_with_ci.pdf`, `kl_vs_eps_k100_panel_with_ci.pdf` |

The confidence intervals use 5,000 percentile-bootstrap resamples with seed 0 and the same resampling indices across mechanisms. Plot fonts use Matplotlib's built-in rendering, so typography differs slightly from the paper.

The two files in `reference_results/` contain the original per-trial utilities and their averages. The notebook uses them only for comparison; it recomputes the experiments from scratch. Comparisons allow an absolute difference of 1e-10 for floating-point variation.

## Tested environment

Tested on macOS (Apple Silicon) with Python 3.14.7 and the package versions in `requirements.txt`. The complete notebook took about 50 seconds in this environment; runtime varies by machine. All per-trial comparisons passed, with a largest difference of 2.6e-15 from the saved results.
