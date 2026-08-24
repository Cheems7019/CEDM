"""
hub_data_comparison.py

Generate fixed data assets for the hub distribution comparison:

    Y1 -> (Y2, Y3, Y4, Y5, Y6)

The intervention is do(Y2 = y2_values, Y5 = y5_values), where both values are
5-dimensional slate vectors. By default both intervention vectors are set to
(1.0, 1.0, 1.0, 1.0, 1.0).

Because Y2 and Y5 are leaf nodes in the hub graph, this intervention does not
change the true target distribution of (Y1, Y3, Y4, Y6). The intervention values
are still stored so the diffusion comparison can use the same fixed Y2/Y5
values for conditional and interventional sampling.

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
from utils.utils_data import TrueSampler_hub


N_FEATURES = 30
Y2_INDICES = np.arange(5, 10)
Y5_INDICES = np.arange(20, 25)
TARGET_INDICES = np.array(
    list(range(0, 5)) + list(range(10, 15)) + list(range(15, 20)) + list(range(25, 30)),
    dtype=int,
)
TARGET_COLUMNS = (
    [f"Y1_X{i}" for i in range(5)]
    + [f"Y3_X{i}" for i in range(10, 15)]
    + [f"Y4_X{i}" for i in range(15, 20)]
    + [f"Y6_X{i}" for i in range(25, 30)]
)
FEATURE_COLUMNS = [f"X{i}" for i in range(N_FEATURES)]
SLATE_SIZE = 5
DEFAULT_TRAINING_SAMPLES = [500, 1000, 2000, 5000]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def intervention_vector(name, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,):
        raise ValueError(f"{name} must be a 5-dimensional vector")
    return values


def hub_graph_arrays():
    groups_conditional = np.zeros(N_FEATURES, dtype=np.int64)
    groups_conditional[0:5] = 0
    groups_conditional[5:10] = 1
    groups_conditional[10:15] = 2
    groups_conditional[15:20] = 3
    groups_conditional[20:25] = 4
    groups_conditional[25:30] = 5

    adjacency_conditional = np.array(
        [
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    variable_adjacency_conditional = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float32)
    for child_start in [5, 10, 15, 20, 25]:
        variable_adjacency_conditional[
            0:SLATE_SIZE,
            child_start : child_start + SLATE_SIZE,
        ] = 1.0

    groups_standard = np.zeros(N_FEATURES, dtype=np.int64)
    adjacency_standard = np.ones((1, 1), dtype=np.float32)
    group_names = np.array(["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"])

    return {
        "groups_conditional": groups_conditional,
        "adjacency_conditional": adjacency_conditional,
        "variable_adjacency_conditional": variable_adjacency_conditional,
        "groups_standard": groups_standard,
        "adjacency_standard": adjacency_standard,
        "group_names": group_names,
        "feature_columns": np.array(FEATURE_COLUMNS),
        "target_indices": TARGET_INDICES,
        "target_columns": np.array(TARGET_COLUMNS),
        "y2_indices": Y2_INDICES,
        "y5_indices": Y5_INDICES,
    }


def save_graph_structure(output_path, save_csv=True):
    graph = hub_graph_arrays()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(output_path, **graph)

    if save_csv:
        base_dir = output_dir if output_dir else "."
        pd.DataFrame(
            graph["adjacency_conditional"],
            index=graph["group_names"],
            columns=graph["group_names"],
        ).to_csv(os.path.join(base_dir, "conditional_adjacency.csv"))

        pd.DataFrame(
            graph["variable_adjacency_conditional"],
            index=graph["feature_columns"],
            columns=graph["feature_columns"],
        ).to_csv(os.path.join(base_dir, "conditional_variable_adjacency.csv"))

        pd.DataFrame(
            {
                "feature": graph["feature_columns"],
                "conditional_group": graph["groups_conditional"],
                "standard_group": graph["groups_standard"],
                "is_target": np.isin(np.arange(N_FEATURES), TARGET_INDICES),
                "is_y2_intervention": np.isin(np.arange(N_FEATURES), Y2_INDICES),
                "is_y5_intervention": np.isin(np.arange(N_FEATURES), Y5_INDICES),
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
    correlation=0.5,
    save_csv=False,
):
    os.makedirs(training_output_dir, exist_ok=True)
    manifest_rows = []

    for seed in range(n_seeds):
        seed_everything(seed)
        sampler = TrueSampler_hub(sigma=sigma, correlation=correlation)

        for n_samples_train in training_samples:
            train_data = sampler.sample(n_samples_train)
            output_path = training_data_path(training_output_dir, n_samples_train, seed)

            np.savez(
                output_path,
                train_data=train_data,
                n_samples_train=np.array(n_samples_train, dtype=int),
                seed=np.array(seed, dtype=int),
                sigma=np.array(sigma, dtype=float),
                correlation=np.array(correlation, dtype=float),
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
                    "correlation": correlation,
                }
            )
            print(f"Saved training data: seed={seed}, n={n_samples_train}, path={output_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(training_output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def generate_hub_do_distribution(
    n_samples,
    y2_values,
    y5_values,
    sigma=1.0,
    correlation=0.5,
    seed=12345,
    return_full=False,
):
    y2_values = intervention_vector("y2_values", y2_values)
    y5_values = intervention_vector("y5_values", y5_values)

    seed_everything(seed)
    sampler = TrueSampler_hub(sigma=sigma, correlation=correlation)
    full = sampler.sample(n_samples)

    # Y2 and Y5 are leaves, so overwriting them is enough to represent do(Y2,Y5)
    # in the saved full data; TARGET_INDICES excludes them.
    full[:, Y2_INDICES] = y2_values
    full[:, Y5_INDICES] = y5_values

    target = full[:, TARGET_INDICES]
    if return_full:
        return target, full
    return target


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed true do-reference data, graph metadata, and training "
            "datasets for hub distribution comparison."
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
    parser.add_argument("--correlation", type=float, default=0.5)
    parser.add_argument(
        "--y2_values",
        type=float,
        nargs=5,
        default=[1.0, 1.0, 1.0, 1.0, 1.0],
        help="Five intervention values for Y2; defaults to all 1.0.",
    )
    parser.add_argument(
        "--y5_values",
        type=float,
        nargs=5,
        default=[1.0, 1.0, 1.0, 1.0, 1.0],
        help="Five intervention values for Y5; defaults to all 1.0.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/hub_distribution_comparison/true_do_reference.npz",
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="./data/hub_distribution_comparison/training",
    )
    parser.add_argument(
        "--graph_output_path",
        type=str,
        default="./data/hub_distribution_comparison/graph_structure.npz",
    )
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--save_training_csv", action="store_true")
    parser.add_argument("--skip_reference", action="store_true")
    parser.add_argument("--skip_training_data", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    parser.add_argument("--no_graph_csv", action="store_true")
    args = parser.parse_args()

    if not args.skip_reference:
        target = generate_hub_do_distribution(
            n_samples=args.n_samples,
            y2_values=args.y2_values,
            y5_values=args.y5_values,
            sigma=args.sigma,
            correlation=args.correlation,
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
            y2_values=np.asarray(args.y2_values, dtype=float),
            y5_values=np.asarray(args.y5_values, dtype=float),
            n_samples=np.array(args.n_samples, dtype=int),
            sigma=np.array(args.sigma, dtype=float),
            correlation=np.array(args.correlation, dtype=float),
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

        print("\nSaved fixed true hub do-reference sample")
        print(f"Path: {args.output_path}")
        print(f"Shape: {target.shape}")
        print(f"Y2 intervention: {np.asarray(args.y2_values, dtype=float)}")
        print(f"Y5 intervention: {np.asarray(args.y5_values, dtype=float)}")

        summary = pd.DataFrame(target, columns=TARGET_COLUMNS).describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        with pd.option_context("display.max_columns", None, "display.width", 160):
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
            correlation=args.correlation,
            save_csv=args.save_training_csv,
        )
        print(f"\nSaved training data manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
