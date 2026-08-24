"""
inference_sachs_misspecification.py

Train local conditional diffusion models for Mek under three conditioning-set
specifications, then run standardized-MCODEC inference for the null edge
P38 -> Mek on synthetic Sachs data.

Per repetition and sample size n:
1. Sample n observations from TrueSampler_sachs
2. Compute tau = MCODEC(z(Mek), z(P38), z(Z_case))
3. Train a local conditional diffusion model for Mek | Z_case
4. Sample synthetic Mek conditional on the observed Z_case
5. Compute tau(d) for each synthetic Mek draw
6. Form p-value = (1 + #{tau(d) >= tau}) / (D + 1)

Cases:
  1. True parent set of Mek: {PKC, PKA, Raf}
  2. True parent set plus extras: {PKC, PKA, Raf, PIP3, PIP2, Plcg, Jnk}
  3. Missing Raf: {PKC, PKA}
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from _project_setup import PROJECT_ROOT
from utils.conditional_ddpm import generate_group_samples, train_conditional_ddpm_group
from utils.mcodec import mcodec
from utils.utils_data import TrueSampler_sachs


NODE_NAMES = [
    "PKC",
    "Plcg",
    "PKA",
    "PIP3",
    "Raf",
    "Jnk",
    "P38",
    "PIP2",
    "Mek",
    "Erk",
    "Akt",
]
NODE_TO_IDX = {name: idx for idx, name in enumerate(NODE_NAMES)}


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardize_with_reference(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardize x using provided center/scale, with zero-scale protection."""
    safe_scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - center) / safe_scale


def get_case_configs() -> list[dict]:
    """Return the three Mek-conditioning cases to evaluate."""
    return [
        {
            "case_id": 1,
            "case_name": "true_parents",
            "parent_names": ["PKC", "PKA", "Raf"],
        },
        {
            "case_id": 2,
            "case_name": "true_plus_extra",
            "parent_names": ["PKC", "PKA", "Raf", "PIP3", "PIP2", "Plcg", "Jnk"],
        },
        {
            "case_id": 3,
            "case_name": "missing_raf",
            "parent_names": ["PKC", "PKA"],
        },
    ]


def fit_local_scaler(data_local: np.ndarray, use_standard_scaler: bool, seed: int):
    """Fit the configured scaler on the local Mek-conditioning dataset."""
    if use_standard_scaler:
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data_local)
    else:
        n_quantiles = min(1000, len(data_local))
        scaler = QuantileTransformer(
            output_distribution="normal",
            random_state=seed,
            n_quantiles=n_quantiles,
            subsample=min(100000, len(data_local)),
        )
        data_norm = scaler.fit_transform(data_local)
    return scaler, data_norm


def make_data_seed(seed_offset: int, n_curr: int, rep: int) -> int:
    """Build a deterministic seed for the shared dataset at (n, repetition)."""
    return int(seed_offset + n_curr * 1_000 + rep)


def make_case_seed(seed_offset: int, case_id: int, n_curr: int, rep: int) -> int:
    """Build a deterministic seed for case-specific training/sampling randomness."""
    return int(seed_offset + case_id * 1_000_000 + n_curr * 1_000 + rep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Sachs misspecification inference for the null edge P38 -> Mek "
            "using local conditional diffusion models for Mek."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="One or more sample sizes to evaluate. Default: 100 200 500 1000.",
    )
    parser.add_argument("--n-rep", type=int, default=100, help="Number of repetitions per n and case.")
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Subset of case ids to run. Default: 1 2 3.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Training/sampling device.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise scale for TrueSampler_sachs.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Rejection threshold for empirical size.")
    parser.add_argument(
        "--monte-carlo-size",
        type=int,
        default=100,
        help="Number of synthetic Mek samples per repetition (D).",
    )
    parser.add_argument(
        "--monte-carlo-batch-size",
        type=int,
        default=100,
        help="Batch size for Monte Carlo sampling.",
    )
    parser.add_argument(
        "--use-standard-scaler",
        action="store_true",
        help="Use StandardScaler instead of QuantileTransformer.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3000,
        help="Training epochs for the local Mek diffusion model.",
    )
    parser.add_argument("--n-steps", type=int, default=1000, help="Diffusion steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 256, 256, 128],
        help="Hidden dimensions for the local conditional diffusion MLP.",
    )
    parser.add_argument("--dim-t", type=int, default=128, help="Time embedding dimension.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sachs_misspecification",
        help="Directory for detailed and summary results.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Optional integer offset added to all repetition seeds.",
    )
    parser.add_argument(
        "--verbose-training",
        action="store_true",
        help="Show per-epoch training output for the local Mek model.",
    )
    args = parser.parse_args()

    n_values = list(dict.fromkeys(args.n))
    if any(n <= 1 for n in n_values):
        raise ValueError("All --n values must be >= 2.")
    if args.n_rep <= 0:
        raise ValueError("--n-rep must be positive.")
    if args.monte_carlo_size <= 0:
        raise ValueError("--monte-carlo-size must be positive.")
    if args.monte_carlo_batch_size <= 0:
        raise ValueError("--monte-carlo-batch-size must be positive.")

    case_configs = {cfg["case_id"]: cfg for cfg in get_case_configs()}
    requested_case_ids = list(dict.fromkeys(args.cases))
    invalid_case_ids = [case_id for case_id in requested_case_ids if case_id not in case_configs]
    if invalid_case_ids:
        raise ValueError(f"Unknown case ids requested: {invalid_case_ids}")
    selected_cases = [case_configs[case_id] for case_id in requested_case_ids]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA not available, switching device from {args.device} to cpu")
        device = "cpu"
    else:
        device = args.device

    args.results_dir.mkdir(parents=True, exist_ok=True)
    scaler_suffix = "_std" if args.use_standard_scaler else "_qt"

    print("=" * 80)
    print("Sachs Misspecification Inference: P38 -> Mek")
    print("=" * 80)
    print(f"n values={n_values}, n_rep={args.n_rep}, cases={requested_case_ids}")
    print(f"device={device}, sigma={args.sigma}, alpha={args.alpha}")
    print(f"D={args.monte_carlo_size}, D_batch={args.monte_carlo_batch_size}")
    print(f"use_standard_scaler={args.use_standard_scaler}")
    print(f"diffusion: n_epochs={args.n_epochs}, n_steps={args.n_steps}, lr={args.lr}")
    print(f"hidden_dims={args.hidden_dims}, dim_t={args.dim_t}")
    print(f"results_dir={args.results_dir}")

    D = args.monte_carlo_size
    D_batch = args.monte_carlo_batch_size
    mc_batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])

    all_results_paths = []
    all_results: list[dict] = []

    target_idx = NODE_TO_IDX["Mek"]
    x_idx = NODE_TO_IDX["P38"]

    for n_curr in n_values:
        results_path = args.results_dir / (
            f"sachs_misspecification_n{n_curr}_rep{args.n_rep}{scaler_suffix}.csv"
        )
        if results_path.exists():
            results_path.unlink()
        all_results_paths.append(results_path)

        print("\n" + "#" * 80)
        print(f"RUNNING n={n_curr}")
        print("#" * 80)
        print(f"results_path={results_path}")

        results_n: list[dict] = []

        for rep in range(1, args.n_rep + 1):
            data_seed = make_data_seed(args.seed_offset, n_curr, rep)
            print(f"\n{'#' * 80}")
            print(f"n={n_curr} | Shared dataset for Rep {rep}/{args.n_rep} | data_seed={data_seed}")
            print(f"{'#' * 80}")

            seed_everything(data_seed)
            sampler = TrueSampler_sachs(sigma=args.sigma)
            data = sampler.sample(n_curr).astype(np.float32)
            print(f"Sampled shared Sachs data with shape {data.shape}")

            for case_cfg in selected_cases:
                case_id = case_cfg["case_id"]
                case_name = case_cfg["case_name"]
                parent_names = case_cfg["parent_names"]
                parent_indices = [NODE_TO_IDX[name] for name in parent_names]
                local_columns = parent_indices + [target_idx]
                case_seed = make_case_seed(args.seed_offset, case_id, n_curr, rep)

                print("\n" + "-" * 80)
                print(
                    f"Case {case_id}: {case_name} | "
                    f"parents={parent_names} | local_dim={len(local_columns)} | "
                    f"case_seed={case_seed}"
                )
                print("-" * 80)

                groups = torch.zeros(len(local_columns), dtype=torch.long)
                groups[-1] = 1  # Mek is the only target group.
                A = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

                seed_everything(case_seed)

                z_obs = data[:, parent_indices]
                x_obs = data[:, [x_idx]]
                y_obs = data[:, [target_idx]]

                z_center = z_obs.mean(axis=0, keepdims=True)
                z_scale = z_obs.std(axis=0, ddof=0, keepdims=True)
                x_center = x_obs.mean(axis=0, keepdims=True)
                x_scale = x_obs.std(axis=0, ddof=0, keepdims=True)
                y_center = y_obs.mean(axis=0, keepdims=True)
                y_scale = y_obs.std(axis=0, ddof=0, keepdims=True)

                z_obs_std = standardize_with_reference(z_obs, z_center, z_scale)
                x_obs_std = standardize_with_reference(x_obs, x_center, x_scale)
                y_obs_std = standardize_with_reference(y_obs, y_center, y_scale)

                tau_observed = mcodec(y_obs_std, x_obs_std, z_obs_std)
                print(f"tau_observed = {tau_observed:.6f}")

                data_local = data[:, local_columns]
                scaler, data_local_norm = fit_local_scaler(
                    data_local=data_local,
                    use_standard_scaler=args.use_standard_scaler,
                    seed=case_seed,
                )
                data_local_norm_t = torch.tensor(data_local_norm, dtype=torch.float32)

                print("Training local conditional diffusion model for Mek...")
                mek_model = train_conditional_ddpm_group(
                    group_id=1,
                    train_data=data_local_norm_t,
                    groups=groups,
                    A=A,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    hidden_dims=args.hidden_dims,
                    dim_t=args.dim_t,
                    n_steps=args.n_steps,
                    device=device,
                    verbose=args.verbose_training,
                )

                z_obs_norm = torch.tensor(
                    data_local_norm[:, :-1],
                    dtype=torch.float32,
                    device=mek_model.device,
                )

                print(f"Sampling synthetic Mek conditional on observed Z_case, D={D}...")
                tau_samples: list[float] = []

                for batch_idx, mc_batch_size in enumerate(mc_batch_sizes):
                    print(f"  MC batch {batch_idx + 1}/{len(mc_batch_sizes)}: {mc_batch_size} samples")

                    parent_values_batch = z_obs_norm.repeat(mc_batch_size, 1)
                    mek_batch_norm = generate_group_samples(
                        ddpm=mek_model,
                        n_samples=mc_batch_size * n_curr,
                        parent_values=parent_values_batch,
                        device=device,
                        verbose=False,
                    )

                    mek_batch_norm = mek_batch_norm.detach().cpu().numpy().reshape(mc_batch_size, n_curr, 1)

                    for j in range(mc_batch_size):
                        local_full_norm = np.empty((n_curr, len(local_columns)), dtype=np.float32)
                        local_full_norm[:, :-1] = data_local_norm[:, :-1]
                        local_full_norm[:, -1] = mek_batch_norm[j, :, 0]

                        local_full_orig = scaler.inverse_transform(local_full_norm)
                        y_sampled = local_full_orig[:, [-1]]
                        y_sampled_std = standardize_with_reference(y_sampled, y_center, y_scale)

                        tau_d = mcodec(y_sampled_std, x_obs_std, z_obs_std)
                        tau_samples.append(float(tau_d))

                tau_samples_arr = np.asarray(tau_samples, dtype=np.float64)
                valid_tau_samples = tau_samples_arr[np.isfinite(tau_samples_arr)]

                if not np.isfinite(tau_observed) or len(valid_tau_samples) == 0:
                    num_larger_or_equal = np.nan
                    p_value = np.nan
                    print(
                        "Warning: tau_observed or tau_samples contain only non-finite values; "
                        "p-value set to NaN."
                    )
                else:
                    num_larger_or_equal = int(np.sum(valid_tau_samples >= tau_observed))
                    p_value = (1 + num_larger_or_equal) / (len(valid_tau_samples) + 1)

                print(f"tau_samples valid: {len(valid_tau_samples)}/{len(tau_samples_arr)}")
                if len(valid_tau_samples) > 0:
                    print(
                        f"tau(d) range: [{valid_tau_samples.min():.6f}, {valid_tau_samples.max():.6f}], "
                        f"mean: {valid_tau_samples.mean():.6f}"
                    )
                print(f"p-value: {p_value}")

                row = {
                    "rep": rep,
                    "data_seed": data_seed,
                    "seed": case_seed,
                    "n": n_curr,
                    "case_id": case_id,
                    "case_name": case_name,
                    "parent_names": ",".join(parent_names),
                    "n_parents": len(parent_names),
                    "target_var": "Mek",
                    "edge_var": "P38",
                    "D": D,
                    "D_effective": int(len(valid_tau_samples)),
                    "tau_observed": float(tau_observed) if np.isfinite(tau_observed) else np.nan,
                    "num_larger_or_equal": num_larger_or_equal,
                    "p_value": p_value,
                    "alpha": args.alpha,
                    "use_standard_scaler": args.use_standard_scaler,
                    "stats_standardized": True,
                }

                for d_idx, tau_d in enumerate(tau_samples_arr, start=1):
                    row[f"tau_d_{d_idx}"] = tau_d

                results_n.append(row)
                all_results.append(row)
                pd.DataFrame(results_n).to_csv(results_path, index=False)
                print(f"Saved cumulative results to: {results_path}")

        results_df = pd.DataFrame(results_n)
        print("\n" + "-" * 80)
        print(f"SUMMARY FOR n={n_curr}")
        print("-" * 80)
        print(f"Total rows: {len(results_df)}")

        for case_cfg in selected_cases:
            case_df = results_df[results_df["case_id"] == case_cfg["case_id"]]
            finite_p = case_df["p_value"].dropna()
            print(
                f"Case {case_cfg['case_id']} ({case_cfg['case_name']}): "
                f"rows={len(case_df)}, finite_p={len(finite_p)}"
            )
            if len(finite_p) > 0:
                print(f"  Mean p-value: {finite_p.mean():.6f}")
                print(f"  Rejection rate (alpha={args.alpha}): {(finite_p <= args.alpha).mean():.4f}")
                print(f"  Min/Max p-value: [{finite_p.min():.6f}, {finite_p.max():.6f}]")
            else:
                print("  No finite p-values were produced.")

    all_results_df = pd.DataFrame(all_results)
    summary_rows = []

    if len(all_results_df) > 0:
        for (n_curr, case_id, case_name, parent_names), group_df in all_results_df.groupby(
            ["n", "case_id", "case_name", "parent_names"],
            sort=True,
        ):
            finite_p = group_df["p_value"].dropna()
            summary_rows.append(
                {
                    "n": int(n_curr),
                    "case_id": int(case_id),
                    "case_name": case_name,
                    "parent_names": parent_names,
                    "n_rep_total": int(len(group_df)),
                    "n_rep_finite": int(len(finite_p)),
                    "alpha": args.alpha,
                    "mean_p_value": float(finite_p.mean()) if len(finite_p) > 0 else np.nan,
                    "rejection_rate": float((finite_p <= args.alpha).mean()) if len(finite_p) > 0 else np.nan,
                    "min_p_value": float(finite_p.min()) if len(finite_p) > 0 else np.nan,
                    "max_p_value": float(finite_p.max()) if len(finite_p) > 0 else np.nan,
                    "mean_tau_observed": float(group_df["tau_observed"].dropna().mean())
                    if group_df["tau_observed"].notna().any()
                    else np.nan,
                    "std_tau_observed": float(group_df["tau_observed"].dropna().std(ddof=0))
                    if group_df["tau_observed"].notna().any()
                    else np.nan,
                    "use_standard_scaler": args.use_standard_scaler,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.results_dir / f"sachs_misspecification_summary{scaler_suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE")
    print("=" * 80)
    print("Per-n results files:")
    for path in all_results_paths:
        print(f"  - {path}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
