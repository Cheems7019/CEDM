"""
sachs_data_comparison.py

Generate fixed data assets for the Sachs distribution comparison under:

    do(PIP3 = pip3_value, P38 = p38_value, Raf = raf_value, Erk = erk_value)

The saved target sample contains all non-intervened variables:
PKC, Plcg, PKA, Jnk, PIP2, Mek, and Akt.

This script also saves per-seed/per-training-size observational training data
and graph metadata so diffusion frameworks and competitors can use identical
data and DAG structures for each configuration.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

from _project_setup import PROJECT_ROOT
from utils.utils_data import TrueSampler_sachs


NODE_NAMES = ["PKC", "Plcg", "PKA", "PIP3", "Raf", "Jnk", "P38", "PIP2", "Mek", "Erk", "Akt"]
INTERVENTION_INDICES = np.array([3, 6, 4, 9], dtype=int)
INTERVENTION_NAMES = ["PIP3", "P38", "Raf", "Erk"]
TARGET_INDICES = np.array([0, 1, 2, 5, 7, 8, 10], dtype=int)
TARGET_COLUMNS = [NODE_NAMES[i] for i in TARGET_INDICES]
FEATURE_COLUMNS = NODE_NAMES
DEFAULT_TRAINING_SAMPLES = [500, 1000, 2000, 5000]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def sachs_graph_arrays():
    groups_conditional = np.arange(len(NODE_NAMES), dtype=np.int64)
    adjacency_conditional = np.zeros((len(NODE_NAMES), len(NODE_NAMES)), dtype=np.float32)

    adjacency_conditional[0, 2] = 1
    adjacency_conditional[0, 4] = 1
    adjacency_conditional[0, 5] = 1
    adjacency_conditional[0, 6] = 1
    adjacency_conditional[0, 8] = 1
    adjacency_conditional[1, 3] = 1
    adjacency_conditional[1, 7] = 1
    adjacency_conditional[2, 4] = 1
    adjacency_conditional[2, 5] = 1
    adjacency_conditional[2, 6] = 1
    adjacency_conditional[2, 8] = 1
    adjacency_conditional[2, 9] = 1
    adjacency_conditional[2, 10] = 1
    adjacency_conditional[3, 7] = 1
    adjacency_conditional[4, 8] = 1
    adjacency_conditional[8, 9] = 1
    adjacency_conditional[9, 10] = 1

    groups_standard = np.zeros(len(NODE_NAMES), dtype=np.int64)
    adjacency_standard = np.ones((1, 1), dtype=np.float32)

    return {
        "groups_conditional": groups_conditional,
        "adjacency_conditional": adjacency_conditional,
        "variable_adjacency_conditional": adjacency_conditional.copy(),
        "groups_standard": groups_standard,
        "adjacency_standard": adjacency_standard,
        "feature_columns": np.array(FEATURE_COLUMNS),
        "node_names": np.array(NODE_NAMES),
        "target_indices": TARGET_INDICES,
        "target_columns": np.array(TARGET_COLUMNS),
        "intervention_indices": INTERVENTION_INDICES,
        "intervention_names": np.array(INTERVENTION_NAMES),
    }


def save_graph_structure(output_path, save_csv=True):
    graph = sachs_graph_arrays()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(output_path, **graph)

    if save_csv:
        base_dir = output_dir if output_dir else "."
        pd.DataFrame(
            graph["adjacency_conditional"],
            index=graph["node_names"],
            columns=graph["node_names"],
        ).to_csv(os.path.join(base_dir, "conditional_adjacency.csv"))

        pd.DataFrame(
            graph["variable_adjacency_conditional"],
            index=graph["node_names"],
            columns=graph["node_names"],
        ).to_csv(os.path.join(base_dir, "conditional_variable_adjacency.csv"))

        pd.DataFrame(
            {
                "feature": graph["feature_columns"],
                "conditional_group": graph["groups_conditional"],
                "standard_group": graph["groups_standard"],
                "is_target": np.isin(np.arange(len(NODE_NAMES)), TARGET_INDICES),
                "is_intervention": np.isin(np.arange(len(NODE_NAMES)), INTERVENTION_INDICES),
            }
        ).to_csv(os.path.join(base_dir, "feature_metadata.csv"), index=False)

    return output_path


def training_data_path(training_output_dir, n_samples_train, seed):
    return os.path.join(training_output_dir, f"train_n{n_samples_train}_seed{seed}.npz")


def generate_and_save_training_data(
    n_seeds,
    training_samples,
    training_output_dir,
    sigma=1.0,
    save_csv=False,
):
    os.makedirs(training_output_dir, exist_ok=True)
    manifest_rows = []

    for seed in range(n_seeds):
        seed_everything(seed)
        sampler = TrueSampler_sachs(sigma=sigma)

        for n_samples_train in training_samples:
            train_data = sampler.sample(n_samples_train)
            output_path = training_data_path(training_output_dir, n_samples_train, seed)

            np.savez(
                output_path,
                train_data=train_data,
                n_samples_train=np.array(n_samples_train, dtype=int),
                seed=np.array(seed, dtype=int),
                sigma=np.array(sigma, dtype=float),
                feature_columns=np.array(FEATURE_COLUMNS),
            )

            csv_path = ""
            if save_csv:
                csv_path = os.path.splitext(output_path)[0] + ".csv"
                pd.DataFrame(train_data, columns=FEATURE_COLUMNS).to_csv(csv_path, index=False)

            manifest_rows.append(
                {
                    "seed": seed,
                    "n_samples_train": n_samples_train,
                    "path": output_path,
                    "csv_path": csv_path,
                    "sigma": sigma,
                }
            )
            print(f"Saved training data: seed={seed}, n={n_samples_train}, path={output_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(training_output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def generate_sachs_do_distribution(
    n_samples,
    pip3_value=0.5,
    p38_value=0.5,
    raf_value=0.5,
    erk_value=0.5,
    sigma=1.0,
    seed=12345,
    return_full=False,
):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, len(NODE_NAMES)))

    x[:, 0] = rng.normal(0, sigma, n_samples)  # PKC
    x[:, 1] = rng.normal(0, sigma, n_samples)  # Plcg

    x[:, 2] = (
        0.5 * np.tanh(x[:, 0] ** 2)
        + 0.4 * x[:, 0]
        + 0.3 * np.sin(x[:, 0]) ** 2
        + 0.2 * x[:, 0] ** 2
        + rng.normal(0, sigma, n_samples)
    )  # PKA

    x[:, 3] = pip3_value  # PIP3 intervention
    x[:, 4] = raf_value  # Raf intervention

    x[:, 5] = (
        0.5 * np.tanh(x[:, 0] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.sin(x[:, 0] * x[:, 2]) * np.cos(x[:, 2])
        + 0.3 * np.tanh(x[:, 0] ** 2)
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0, sigma, n_samples)
    )  # Jnk

    x[:, 6] = p38_value  # P38 intervention

    x[:, 7] = (
        0.5 * np.tanh(x[:, 1] ** 2 + x[:, 3] ** 2)
        + 0.4 * np.sin(x[:, 1]) * np.tanh(x[:, 3])
        + 0.3 * np.cos(x[:, 3]) * np.tanh(x[:, 1])
        + 0.2 * x[:, 1] * x[:, 3]
        + rng.normal(0, sigma, n_samples)
    )  # PIP2

    x[:, 8] = (
        0.4 * np.tanh(x[:, 0] ** 2 + x[:, 4] ** 2)
        + 0.4 * np.tanh(x[:, 2] * x[:, 4])
        + 0.3 * np.sin(x[:, 0]) * np.tanh(x[:, 4] * x[:, 2])
        + 0.2 * x[:, 0] * x[:, 2]
        + rng.normal(0, sigma, n_samples)
    )  # Mek

    x[:, 9] = erk_value  # Erk intervention

    x[:, 10] = (
        0.5 * np.tanh(x[:, 9] ** 2 + x[:, 2] ** 2)
        + 0.4 * np.cos(x[:, 2]) ** 2 * np.tanh(x[:, 9])
        + 0.3 * np.tanh(x[:, 9] * x[:, 2]) * np.sin(x[:, 2]) ** 2
        + 0.2 * np.tanh(x[:, 9]) * np.cos(x[:, 2])
        + rng.normal(0, sigma, n_samples)
    )  # Akt

    target = x[:, TARGET_INDICES]
    if return_full:
        return target, x
    return target


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed true do-reference data, graph metadata, and training "
            "datasets for Sachs distribution comparison."
        )
    )
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument(
        "--training_samples",
        type=int,
        nargs="+",
        default=DEFAULT_TRAINING_SAMPLES,
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--pip3_value", type=float, default=0.5)
    parser.add_argument("--p38_value", type=float, default=0.5)
    parser.add_argument("--raf_value", type=float, default=0.5)
    parser.add_argument("--erk_value", type=float, default=0.5)
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/sachs_distribution_comparison/true_do_reference.npz",
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="./data/sachs_distribution_comparison/training",
    )
    parser.add_argument(
        "--graph_output_path",
        type=str,
        default="./data/sachs_distribution_comparison/graph_structure.npz",
    )
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--save_training_csv", action="store_true")
    parser.add_argument("--skip_reference", action="store_true")
    parser.add_argument("--skip_training_data", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    parser.add_argument("--no_graph_csv", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    intervention_values = np.array(
        [args.pip3_value, args.p38_value, args.raf_value, args.erk_value],
        dtype=float,
    )

    if not args.skip_reference:
        target = generate_sachs_do_distribution(
            n_samples=args.n_samples,
            pip3_value=args.pip3_value,
            p38_value=args.p38_value,
            raf_value=args.raf_value,
            erk_value=args.erk_value,
            sigma=args.sigma,
            seed=args.seed,
        )

        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.savez(
            args.output_path,
            target_samples=target,
            target_indices=TARGET_INDICES,
            target_columns=np.array(TARGET_COLUMNS),
            intervention_indices=INTERVENTION_INDICES,
            intervention_names=np.array(INTERVENTION_NAMES),
            intervention_values=intervention_values,
            n_samples=np.array(args.n_samples, dtype=int),
            sigma=np.array(args.sigma, dtype=float),
            seed=np.array(args.seed, dtype=int),
        )

        if args.save_csv:
            csv_path = args.csv_path
            if csv_path is None:
                csv_path = os.path.splitext(args.output_path)[0] + ".csv"
            csv_dir = os.path.dirname(csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            pd.DataFrame(target, columns=TARGET_COLUMNS).to_csv(csv_path, index=False)
            print(f"Saved CSV reference data to: {csv_path}")

        print("\nSaved fixed true Sachs do-reference sample")
        print(f"Path: {args.output_path}")
        print(f"Shape: {target.shape}")
        for name, value in zip(INTERVENTION_NAMES, intervention_values):
            print(f"{name} intervention: {value}")

        summary = pd.DataFrame(target, columns=TARGET_COLUMNS).describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print("\nTarget sample summary:")
            print(summary.round(4))

    if not args.skip_graph:
        save_graph_structure(args.graph_output_path, save_csv=not args.no_graph_csv)
        print(f"\nSaved graph structure to: {args.graph_output_path}")
        if not args.no_graph_csv:
            print(
                "Saved graph CSV helpers to: "
                f"{os.path.dirname(args.graph_output_path) or '.'}"
            )

    if not args.skip_training_data:
        manifest_path = generate_and_save_training_data(
            n_seeds=args.n_seeds,
            training_samples=args.training_samples,
            training_output_dir=args.training_output_dir,
            sigma=args.sigma,
            save_csv=args.save_training_csv,
        )
        print(f"\nSaved training data manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
