# Causality-Encoded Diffusion Models (CEDM)

---

## Overview

This repository provides the implementation for the paper:

> **Causality-Encoded Diffusion Models for Interventional Sampling and Edge Inference**
> Li Chen, Xiaotong Shen, Wei Pan (2026)

Standard diffusion models learn observational distributions but are causally agnostic -- they do not distinguish between observationally equivalent graphs and cannot answer interventional queries. This work introduces:

- **CEDM** (Causality-Encoded Diffusion Models): a score-based generative framework that incorporates a known directed acyclic graph (DAG) by training one conditional diffusion model per node, each conditioned on its parent nodes. The resulting sampler recovers the observational distribution and supports interventional sampling via the do-calculus by clamping intervened variables and propagating effects downstream through the graph during reverse diffusion.

- **CEDMI** (CEDM-based Inference): a resampling-based procedure for testing directed edges in a given DAG. The test generates null replicates of a child node under the candidate graph using CEDM's interventional sampler, then compares the observed **MCODEC** statistic to its null distribution to produce a Monte Carlo p-value.

- **MCODEC** (Multivariate Conditional Dependence Coefficient): a new rank-based, tuning-free test statistic for multivariate conditional dependence, extending the Azadkia-Chatterjee coefficient to the multivariate setting.

**Key theoretical results**: CEDM achieves score-matching and total-variation convergence rates governed by the maximum local dimension (a node and its parents) rather than the ambient dimension, yielding provable gains over causally agnostic baselines on sparse DAGs. CEDMI achieves asymptotic type-I error control under sample splitting.

---

## Installation

This project uses **Python 3.10.19**. Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

The Python dependencies are `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`seaborn`, `hyppo`, and `tqdm`. See `requirements.txt` for the tested versions.

For the flow cytometry inference step, **R** is also required. The script `flow_cytometry_inference/Edge_Inference.Rmd` uses the R package `RCIT`.

---

## Repository Structure

```
CEDM/
|
+-- utils/
|   +-- conditional_ddpm.py         # CEDM: per-node conditional diffusion models
|   +-- ddpm.py                     # Causally agnostic joint diffusion baseline
|   +-- utils_model.py              # Masked MLP backbone (standard framework)
|   +-- utils_data.py               # Synthetic data generators and ground-truth conditionals
|   +-- mcodec.py                   # MCODEC test statistic
|
+-- simulations/
|   |   --- Figure 1: do-distributional comparison ---
|   +-- *_data_comparison.py
|   +-- *_distribution_comparison.py
|   +-- summary_distribution_comparison.ipynb
|   |   --- Figures 2-3: inference with sample splitting ---
|   +-- DataGen_inference_split.py
|   +-- inference_*_split.py
|   +-- summary_inference_split.ipynb
|   |   --- Supplementary simulation studies ---
|   +-- *_comparison.py
|   +-- inference_size_*.py
|   +-- inference_power_*.py
|   +-- inference_sachs_misspecification.py
|   +-- summary_comparison.ipynb
|   +-- summary_inference.ipynb
|
+-- flow_cytometry_inference/
|   +-- cytometry_diffusion_training.py  # Train CEDM on flow cytometry data
|   +-- Edge_Inference.Rmd               # R script for edge inference
|   +-- Flow_Cytometry.csv               # Pre-processed Sachs dataset (n=1755)
|   +-- model_check.py                   # Model diagnostics
|   +-- plot_histograms.py               # Result visualisation
|   |   --- Figure 5 (main text): shuffled Sachs experiments ---
|   +-- p44_42_to_pakts473_shuffled.py
|   +-- pip2_plcg_pkc_shuffled.py
|   +-- pip3_pakts473_shuffled.py
|   +-- pkc_pka_shuffled.py
|   +-- flow_cytometry_shuffled_summary.ipynb  # Generates Figure 5
|
+-- requirements.txt
```

---

## Reproducing the Paper Results

Run the commands below from the repository root. Simulation entry points also normalise
their import and output paths to the repository root when launched from another directory.

### Figure 1 (Main Text) -- Data and Distributional Comparison

Compares CEDM against causally agnostic and VACA baselines on four graph structures
(hub, random DAG, long chain, Sachs) in terms of distribution-level metrics.

```bash
python simulations/hub_data_comparison.py
python simulations/hub_distribution_comparison.py
python simulations/long_chain_data_comparison.py
python simulations/long_chain_distribution_comparison.py
python simulations/random_dag_data_comparison.py
python simulations/random_dag_distribution_comparison.py
python simulations/sachs_data_comparison.py
python simulations/sachs_distribution_comparison.py
```

Then open `simulations/summary_distribution_comparison.ipynb` to generate Figure 1.
The notebook can also read separately generated VACA CSV files from the documented
`results/*_distribution_comparison/vaca/` locations; VACA source code is not bundled here.

---

### Figures 2-3 (Main Text) -- Conditional Independence Testing

These experiments use sample splitting: CEDM is trained on a held-out training set and the
MCODEC statistic is evaluated on a separate inference set.

**Step 1 -- Generate data (run once):**

```bash
python simulations/DataGen_inference_split.py
```

**Step 2 -- Run CEDMI:**

```bash
python simulations/inference_size_gamma_split.py    # Empirical size, Chain null
python simulations/inference_size_eta_split.py      # Empirical size, Fork null
python simulations/inference_power_gamma_split.py   # Power, varying gamma
python simulations/inference_power_eta_split.py     # Power, varying eta
```

Then open `simulations/summary_inference_split.ipynb` to produce Figures 2-3. Implementations
of comparison methods are not bundled; obtain them from the original sources listed below.

---

### Figure 5 (Main Text) -- Flow Cytometry Analysis on Shuffled Data

Applies CEDMI to shuffled replicates of the Sachs et al. (2005) protein-signalling dataset
to assess four disputed edges between the literature consensus network and the Sachs et al.
Bayesian network (p44/42 -> pakts473, PIP3 -> pakts473, PIP2 <- Plcg/PKC, PKC -> PKA).

```bash
python flow_cytometry_inference/p44_42_to_pakts473_shuffled.py
python flow_cytometry_inference/pip2_plcg_pkc_shuffled.py
python flow_cytometry_inference/pip3_pakts473_shuffled.py
python flow_cytometry_inference/pkc_pka_shuffled.py
```

Then open `flow_cytometry_inference/flow_cytometry_shuffled_summary.ipynb` to generate
Figure 5.

---

### Flow Cytometry Data Analysis (Original, Non-Shuffled)

Applies CEDMI to the full Sachs et al. (2005) protein-signalling dataset (n = 1755
single-cell measurements, 11 proteins) without shuffling.

**Step 1 -- Train CEDM (Python):**

```bash
python flow_cytometry_inference/cytometry_diffusion_training.py
```

**Step 2 -- Edge inference (R):**

Open and run `flow_cytometry_inference/Edge_Inference.Rmd` in R.

---

### Supplementary Figure 1 -- Conditional Expectation Estimation

Evaluates CEDM versus a causally agnostic diffusion baseline (using RePaint for conditional
generation) on four graph structures across training sizes n in {500, 1000, 2000, 5000}
(50 seeds each).

```bash
python simulations/hub_comparison.py
python simulations/random_dag_comparison.py
python simulations/long_chain_comparison.py
python simulations/sachs_comparison.py
```

Then open `simulations/summary_comparison.ipynb` to generate Supplementary Figure 1.

---

### Supplementary Figures 2-4 -- Inference Without Sample Splitting

```bash
python simulations/inference_size_gamma.py
python simulations/inference_size_eta.py
python simulations/inference_power_gamma.py
python simulations/inference_power_eta.py
```

Then open `simulations/summary_inference.ipynb` to produce Supplementary Figures 2-4.

---

### Supplementary Figure 6 -- Model Misspecification on Sachs

Evaluates CEDMI under three conditioning-set specifications for the null edge P38 -> Mek on
synthetic Sachs data (true parent set, super-set with extra parents, missing-parent subset).

```bash
python simulations/inference_sachs_misspecification.py
```

---

## Method Details

### CEDM Architecture

Each node group `j` has its own `ConditionalDDPM` with an MLP noise-prediction network:

```
Input: [noisy X_j | X_{pa(j)} | sinusoidal time embedding]
  -> SiLU MLP (hidden dims: 512 -> 256 -> 256 -> 256 -> 128)
Output: predicted noise for X_j
```

Training uses denoising score matching with a linear beta schedule
(beta_1 = 1e-4 to beta_T = 0.02, T = 1000 steps). Node groups are trained in
topological order; each model conditions on the parent groups' observed (not noisy) values.

### CEDMI Procedure

Given a candidate edge (k -> j) to test:

1. Fit CEDM on the training split encoding the null graph (edge removed).
2. On the inference split, compute the observed MCODEC statistic tau.
3. Generate D = 100 Monte Carlo resamples of X_j under do(X_{pa'(j)} = observed values).
4. Compute tau^(m) for each resample.
5. Monte Carlo p-value: p = (1 + #{tau^(m) >= tau}) / (D + 1).

### MCODEC

For multivariate X_j, X_k, X_{pa'(j)\k}:

```
MCODEC(X_j, X_k | X_{pa'(j)\k}) =
    [ xi(X_j, (X_{pa'(j)\k}, X_k)) - xi(X_j, X_{pa'(j)\k}) ]
    / [ 1 - xi(X_j, X_{pa'(j)\k}) ]
```

where `xi` is the Ansari-Fuchs multi-output extension of the Azadkia-Chatterjee rank
correlation. MCODEC lies in [0, 1], equals 0 if and only if X_j is conditionally independent
of X_k given X_{pa'(j)\k}, and is strongly consistent.

---

## External Comparison Methods

Third-party competitor implementations are intentionally not included in this repository.
The paper's comparison methods are available from their original sources:

| Method  | Reference                    | Source                                          |
|---------|------------------------------|-------------------------------------------------|
| SGMCIT  | Ren et al., KDD 2025         | https://github.com/jinchenghou123/SGMCIT        |
| NNLSCIT | Li et al., NeurIPS 2023      | https://github.com/LeeShuai-kenwitch/NNLSCIT    |
| CDCIT   | Yang et al., AAAI 2025       | https://github.com/Yanfeng-Yang-0316/CDCIT      |
| CCIT    | Sen et al., NeurIPS 2017     | PyPI package `ccit`                             |
| RCoT / RCIT | Strobl et al., JCI 2019  | R package `RCIT`                                |
| VACA    | Sanchez-Martin et al., AAAI 2022 | https://github.com/psanch21/VACA  |

The authors thank the original developers for the competing methods.

---

## Citation

If you use this code, please cite:

```bibtex
@article{chen2026causality,
  title   = {Causality-Encoded Diffusion Models for Interventional Sampling and Edge Inference},
  author  = {Chen, Li and Shen, Xiaotong and Pan, Wei},
  journal = {arXiv preprint arXiv:2604.21843},
  year    = {2026}
}
```
