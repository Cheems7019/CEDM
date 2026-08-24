"""
pip3_pakts473_shuffled.py

Train Cluster-1 style conditional diffusion models on shuffled Flow Cytometry
datasets, then run MCODEC inference for the edge PIP3 -> pakts473.

Per iteration i:
1. Load flow_cytometry_inference/shuffled_data/Flow_Cytometry_i.csv
2. Take first n rows and keep columns: (PIP3, PKA, p44/42, pakts473)
3. Train conditional diffusion with DAG:
      G0=[PIP3] -> G1=[PKA, p44/42] -> G2=[pakts473]
4. Standardize observed variables and compute
      tau = MCODEC(z(pakts473), z(PIP3), z([PKA, p44/42]))
5. Sample synthetic pakts473 conditional on observed (PIP3, PKA, p44/42),
   standardize synthetic pakts473 with observed pakts473 center/scale,
   compute tau(d), and obtain p-value:
      p = (1 + #{tau(d) >= tau}) / (D + 1)
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.append(str(Path(__file__).parent.parent))

from utils.conditional_ddpm import generate_conditional_samples, train_all_conditional_ddpms
from utils.mcodec import mcodec


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


def load_iteration_data(shuffled_dir: Path, iteration: int, n: int) -> tuple[np.ndarray, Path]:
    """Load first n rows from shuffled dataset i with required columns."""
    data_path = shuffled_dir / f"Flow_Cytometry_{iteration}.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing shuffled dataset: {data_path}. "
            "Generate shuffled files first with data_shuffling.py."
        )

    df = pd.read_csv(data_path)
    required_cols = ["PIP3", "PKA", "p44/42", "pakts473"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    if len(df) < n:
        raise ValueError(f"{data_path} has only {len(df)} rows, but n={n} was requested.")

    data = df.loc[: n - 1, required_cols].to_numpy(dtype=np.float32)
    return data, data_path


def standardize_with_reference(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardize x using provided center/scale, with zero-scale protection."""
    safe_scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - center) / safe_scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Cluster-1 style diffusion models on shuffled Flow Cytometry data "
            "and run standardized-MCODEC inference for PIP3 -> pakts473."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="One or more n values (first n observations) to evaluate. Default: 100 200 500 1000.",
    )
    parser.add_argument("--n-rep", type=int, default=100, help="Number of shuffled datasets (iterations).")
    parser.add_argument("--start-iteration", type=int, default=1, help="Starting shuffled index i (default: 1).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training/sampling device.")
    parser.add_argument("--monte-carlo-size", type=int, default=100, help="Number of synthetic pakts473 samples (D).")
    parser.add_argument("--monte-carlo-batch-size", type=int, default=100, help="Batch size for Monte Carlo sampling.")
    parser.add_argument("--use-standard-scaler", action="store_true", help="Use StandardScaler instead of QuantileTransformer.")
    parser.add_argument("--n-epochs", type=int, default=3000, help="Training epochs per group.")
    parser.add_argument("--n-steps", type=int, default=1000, help="Diffusion steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[640, 320, 320, 320, 160],
        help="Hidden dimensions for conditional diffusion MLP.",
    )
    parser.add_argument("--dim-t", type=int, default=64, help="Time embedding dimension.")
    parser.add_argument(
        "--shuffled-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "shuffled_data",
        help="Directory containing Flow_Cytometry_i.csv files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_shuffled",
        help="Directory for inference results.",
    )
    parser.add_argument("--save-checkpoints", action="store_true", help="Save trained models for each iteration.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints_shuffled",
        help="Directory for optional per-iteration checkpoints.",
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA not available, switching device from {args.device} to cpu")
        device = "cpu"
    else:
        device = args.device

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler_suffix = "_std" if args.use_standard_scaler else "_qt"

    print("=" * 80)
    print("Shuffled Flow Cytometry Inference (standardized stats): PIP3 -> pakts473")
    print("=" * 80)
    print(f"n values={n_values}, n_rep={args.n_rep}, start_iteration={args.start_iteration}")
    print(f"device={device}, D={args.monte_carlo_size}, D_batch={args.monte_carlo_batch_size}")
    print(f"use_standard_scaler={args.use_standard_scaler}")
    print(f"results_dir={args.results_dir}")

    # Cluster-1 setup on 4 variables: [PIP3, PKA, p44/42, pakts473]
    groups = torch.zeros(4, dtype=torch.long)
    groups[0] = 0  # PIP3
    groups[1] = 1  # PKA
    groups[2] = 1  # p44/42
    groups[3] = 2  # pakts473

    A = np.array(
        [
            [0, 1, 0],  # G0 -> G1
            [0, 0, 1],  # G1 -> G2
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    D = args.monte_carlo_size
    D_batch = args.monte_carlo_batch_size
    mc_batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])

    all_results_paths = []

    for n_curr in n_values:
        results_path = args.results_dir / (
            f"pip3_pakts473_shuffled_n{n_curr}_rep{args.n_rep}{scaler_suffix}.csv"
        )
        if results_path.exists():
            results_path.unlink()
        all_results_paths.append(results_path)

        print("\n" + "#" * 80)
        print(f"RUNNING n={n_curr}")
        print("#" * 80)
        print(f"results_path={results_path}")

        results_n = []

        for iteration in range(args.start_iteration, args.start_iteration + args.n_rep):
            print(f"\n{'=' * 80}")
            print(f"n={n_curr} | Iteration {iteration}")
            print(f"{'=' * 80}")

            seed_everything(iteration)

            data, data_path = load_iteration_data(args.shuffled_dir, iteration, n_curr)
            print(f"Loaded {data_path} with subset shape {data.shape}")

            if args.use_standard_scaler:
                scaler = StandardScaler()
                data_norm = scaler.fit_transform(data)
            else:
                n_quantiles = min(1000, n_curr)
                scaler = QuantileTransformer(
                    output_distribution="normal",
                    random_state=iteration,
                    n_quantiles=n_quantiles,
                    subsample=min(100000, n_curr),
                )
                data_norm = scaler.fit_transform(data)

            data_norm_t = torch.tensor(data_norm, dtype=torch.float32)

            print("Training conditional diffusion models...")
            models = train_all_conditional_ddpms(
                train_data=data_norm_t,
                groups=groups,
                A=A,
                hidden_dims=args.hidden_dims,
                dim_t=args.dim_t,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                lr=args.lr,
                device=device,
                verbose=True,
            )

            if args.save_checkpoints:
                iter_ckpt_dir = args.checkpoint_dir / f"n{n_curr}" / f"iter_{iteration}"
                iter_ckpt_dir.mkdir(parents=True, exist_ok=True)

                with open(iter_ckpt_dir / "models.pkl", "wb") as f:
                    pickle.dump(models, f)
                with open(iter_ckpt_dir / "metadata.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "iteration": iteration,
                            "n": n_curr,
                            "groups": groups,
                            "A": A,
                            "hidden_dims": args.hidden_dims,
                            "dim_t": args.dim_t,
                            "n_steps": args.n_steps,
                            "n_epochs": args.n_epochs,
                            "lr": args.lr,
                            "use_standard_scaler": args.use_standard_scaler,
                            "scaler": scaler,
                        },
                        f,
                    )
                print(f"Saved checkpoint to {iter_ckpt_dir}")

            pip3 = data[:, 0:1]
            pka_p44_42 = data[:, 1:3]
            pakts473_obs = data[:, 3:4]

            pip3_center = pip3.mean(axis=0, keepdims=True)
            pip3_scale = pip3.std(axis=0, ddof=0, keepdims=True)
            pka_p44_42_center = pka_p44_42.mean(axis=0, keepdims=True)
            pka_p44_42_scale = pka_p44_42.std(axis=0, ddof=0, keepdims=True)
            pakts473_center = pakts473_obs.mean(axis=0, keepdims=True)
            pakts473_scale = pakts473_obs.std(axis=0, ddof=0, keepdims=True)

            pip3_z = standardize_with_reference(pip3, pip3_center, pip3_scale)
            pka_p44_42_z = standardize_with_reference(pka_p44_42, pka_p44_42_center, pka_p44_42_scale)
            pakts473_obs_z = standardize_with_reference(pakts473_obs, pakts473_center, pakts473_scale)

            tau_observed = mcodec(pakts473_obs_z, pip3_z, pka_p44_42_z)
            print(f"tau_observed = {tau_observed:.6f}")

            print(f"Sampling synthetic pakts473 conditional on observed (PIP3, PKA, p44/42), D={D}...")
            tau_samples = []

            for batch_idx, mc_batch_size in enumerate(mc_batch_sizes):
                print(f"  MC batch {batch_idx + 1}/{len(mc_batch_sizes)}: {mc_batch_size} samples")

                do_vars = {
                    0: data_norm[:, 0],  # PIP3
                    1: data_norm[:, 1],  # PKA
                    2: data_norm[:, 2],  # p44/42
                }
                sample_vars = [3]  # pakts473

                pakts473_batch_norm = generate_conditional_samples(
                    models=models,
                    groups=groups,
                    A=A,
                    do_vars=do_vars,
                    sample_vars=sample_vars,
                    n_samples=mc_batch_size,
                    device=device,
                )

                pakts473_batch_norm = pakts473_batch_norm.cpu().detach().numpy()  # [mc_batch_size, n, 1]

                for j in range(mc_batch_size):
                    full_norm = np.zeros((n_curr, 4), dtype=np.float32)
                    full_norm[:, 0:3] = data_norm[:, 0:3]
                    full_norm[:, 3] = pakts473_batch_norm[j, :, 0]

                    full_sample = scaler.inverse_transform(full_norm)
                    pakts473_sampled = full_sample[:, 3:4]

                    pakts473_sampled_z = standardize_with_reference(
                        pakts473_sampled, pakts473_center, pakts473_scale
                    )
                    tau_d = mcodec(pakts473_sampled_z, pip3_z, pka_p44_42_z)
                    tau_samples.append(float(tau_d))

            tau_samples_arr = np.asarray(tau_samples, dtype=np.float64)
            valid_tau_samples = tau_samples_arr[np.isfinite(tau_samples_arr)]

            if not np.isfinite(tau_observed) or len(valid_tau_samples) == 0:
                num_larger_or_equal = np.nan
                p_value = np.nan
                print("Warning: tau_observed or tau_samples contain only non-finite values; p-value set to NaN.")
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
                "iteration": iteration,
                "shuffled_file": data_path.name,
                "n": n_curr,
                "D": D,
                "D_effective": int(len(valid_tau_samples)),
                "tau_observed": float(tau_observed) if np.isfinite(tau_observed) else np.nan,
                "num_larger_or_equal": num_larger_or_equal,
                "p_value": p_value,
                "use_standard_scaler": args.use_standard_scaler,
                "stats_standardized": True,
            }

            for d_idx, tau_d in enumerate(tau_samples_arr, start=1):
                row[f"tau_d_{d_idx}"] = tau_d

            results_n.append(row)
            pd.DataFrame(results_n).to_csv(results_path, index=False)
            print(f"Saved cumulative results to: {results_path}")

        results_df = pd.DataFrame(results_n)
        print("\n" + "-" * 80)
        print(f"SUMMARY FOR n={n_curr}")
        print("-" * 80)
        print(f"Total iterations: {len(results_df)}")

        finite_p = results_df["p_value"].dropna()
        if len(finite_p) > 0:
            print(f"Mean p-value: {finite_p.mean():.6f}")
            print(f"Rejection rate (alpha=0.05): {(finite_p <= 0.05).mean():.4f}")
            print(f"Min/Max p-value: [{finite_p.min():.6f}, {finite_p.max():.6f}]")
        else:
            print("No finite p-values were produced.")

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE")
    print("=" * 80)
    print("Per-n results files:")
    for path in all_results_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
