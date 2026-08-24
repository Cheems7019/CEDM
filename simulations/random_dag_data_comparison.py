"""
random_dag_data_comparison.py

Generate fixed data assets for random-DAG distribution comparison.

For each seed, this script:
  1. Samples the random 6-slate DAG.
  2. Saves the DAG structure for competitors.
  3. Generates a fixed true do-reference sample under do(Y2, Y5).
  4. Generates per-training-size observational training datasets.

The default intervention is:
    Y2 = (0.5, 0.5, 0.5, 0.5, 0.5)
    Y5 = (0.5, 0.5, 0.5, 0.5, 0.5)

The evaluation target is (Y1, Y3, Y4, Y6), matching the long-chain and hub
distribution comparisons.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

from _project_setup import PROJECT_ROOT
from utils.utils_data import TrueSampler_random, sample_random_dag


SLATE_SIZE = 5
N_SLATES = 6
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
GROUP_NAMES = np.array(["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"])
DEFAULT_TRAINING_SAMPLES = [500, 1000, 2000, 5000]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def edge_prob_token(edge_prob):
    return f"edge{edge_prob:g}".replace("-", "m").replace(".", "p")


def reference_data_path(reference_output_dir, edge_prob, seed):
    return os.path.join(
        reference_output_dir, f"reference_{edge_prob_token(edge_prob)}_seed{seed}.npz"
    )


def training_data_path(training_output_dir, edge_prob, n_samples_train, seed):
    return os.path.join(
        training_output_dir,
        f"train_{edge_prob_token(edge_prob)}_n{n_samples_train}_seed{seed}.npz",
    )


def graph_structure_path(graph_output_dir, edge_prob, seed):
    return os.path.join(
        graph_output_dir, f"graph_{edge_prob_token(edge_prob)}_seed{seed}.npz"
    )


def intervention_vector(name, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (SLATE_SIZE,):
        raise ValueError(f"{name} must be a {SLATE_SIZE}-dimensional vector")
    return values


def slate_covariance(sigma=1.0, correlation=0.5):
    correlation_matrix = np.eye(SLATE_SIZE)
    for i in range(SLATE_SIZE):
        for j in range(SLATE_SIZE):
            if i != j:
                correlation_matrix[i, j] = correlation ** abs(i - j)
    return correlation_matrix * sigma**2


def sample_root_slate(n_samples, sigma=1.0, correlation=0.5):
    mean = np.zeros(SLATE_SIZE)
    cov = slate_covariance(sigma=sigma, correlation=correlation)
    return np.random.multivariate_normal(mean, cov, size=n_samples)


def groups_array():
    groups = np.zeros(N_FEATURES, dtype=np.int64)
    for slate_idx in range(N_SLATES):
        groups[slate_idx * SLATE_SIZE : (slate_idx + 1) * SLATE_SIZE] = slate_idx
    return groups


def variable_adjacency_from_slate_dag(dag_adjacency):
    variable_adjacency = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float32)
    for parent in range(N_SLATES):
        for child in range(N_SLATES):
            if dag_adjacency[parent, child] == 1:
                parent_start = parent * SLATE_SIZE
                child_start = child * SLATE_SIZE
                variable_adjacency[
                    parent_start : parent_start + SLATE_SIZE,
                    child_start : child_start + SLATE_SIZE,
                ] = 1.0
    return variable_adjacency


def random_dag_graph_arrays(dag_adjacency):
    groups_conditional = groups_array()
    groups_standard = np.zeros(N_FEATURES, dtype=np.int64)
    adjacency_conditional = np.asarray(dag_adjacency, dtype=np.float32)
    adjacency_standard = np.ones((1, 1), dtype=np.float32)
    variable_adjacency = variable_adjacency_from_slate_dag(adjacency_conditional)

    return {
        "groups_conditional": groups_conditional,
        "adjacency_conditional": adjacency_conditional,
        "variable_adjacency_conditional": variable_adjacency,
        "groups_standard": groups_standard,
        "adjacency_standard": adjacency_standard,
        "group_names": GROUP_NAMES,
        "feature_columns": np.array(FEATURE_COLUMNS),
        "target_indices": TARGET_INDICES,
        "target_columns": np.array(TARGET_COLUMNS),
        "y2_indices": Y2_INDICES,
        "y5_indices": Y5_INDICES,
    }


def save_graph_structure(graph_output_dir, edge_prob, seed, dag_adjacency, save_csv=True):
    os.makedirs(graph_output_dir, exist_ok=True)
    output_path = graph_structure_path(graph_output_dir, edge_prob, seed)
    graph = random_dag_graph_arrays(dag_adjacency)
    n_edges = int(np.sum(dag_adjacency))

    np.savez(
        output_path,
        **graph,
        edge_prob=np.array(edge_prob, dtype=float),
        seed=np.array(seed, dtype=int),
        function_seed=np.array(seed, dtype=int),
        n_edges=np.array(n_edges, dtype=int),
    )

    csv_paths = {"adjacency_csv": "", "variable_adjacency_csv": "", "feature_metadata_csv": ""}
    if save_csv:
        token = f"{edge_prob_token(edge_prob)}_seed{seed}"
        adjacency_csv = os.path.join(graph_output_dir, f"conditional_adjacency_{token}.csv")
        variable_csv = os.path.join(
            graph_output_dir, f"conditional_variable_adjacency_{token}.csv"
        )
        metadata_csv = os.path.join(graph_output_dir, f"feature_metadata_{token}.csv")

        pd.DataFrame(
            graph["adjacency_conditional"],
            index=graph["group_names"],
            columns=graph["group_names"],
        ).to_csv(adjacency_csv)
        pd.DataFrame(
            graph["variable_adjacency_conditional"],
            index=graph["feature_columns"],
            columns=graph["feature_columns"],
        ).to_csv(variable_csv)
        pd.DataFrame(
            {
                "feature": graph["feature_columns"],
                "conditional_group": graph["groups_conditional"],
                "standard_group": graph["groups_standard"],
                "is_target": np.isin(np.arange(N_FEATURES), TARGET_INDICES),
                "is_y2_intervention": np.isin(np.arange(N_FEATURES), Y2_INDICES),
                "is_y5_intervention": np.isin(np.arange(N_FEATURES), Y5_INDICES),
            }
        ).to_csv(metadata_csv, index=False)

        csv_paths = {
            "adjacency_csv": adjacency_csv,
            "variable_adjacency_csv": variable_csv,
            "feature_metadata_csv": metadata_csv,
        }

    return output_path, csv_paths


def generate_random_dag_do_distribution(
    n_samples,
    dag_adjacency,
    y2_values,
    y5_values,
    sigma=1.0,
    correlation=0.5,
    function_seed=0,
    sample_seed=12345,
    return_full=False,
):
    y2_values = intervention_vector("y2_values", y2_values)
    y5_values = intervention_vector("y5_values", y5_values)
    dag_adjacency = np.asarray(dag_adjacency)

    sampler = TrueSampler_random(
        dag_adjacency=dag_adjacency,
        sigma=sigma,
        correlation=correlation,
        function_seed=function_seed,
    )

    np.random.seed(sample_seed)
    full = np.zeros((n_samples, N_FEATURES))

    for slate_idx in range(N_SLATES):
        slate_start = slate_idx * SLATE_SIZE
        slate_end = slate_start + SLATE_SIZE

        if slate_idx == 1:
            full[:, slate_start:slate_end] = y2_values
        elif slate_idx == 4:
            full[:, slate_start:slate_end] = y5_values
        else:
            parents = np.where(dag_adjacency[:, slate_idx] == 1)[0]
            if len(parents) == 0:
                full[:, slate_start:slate_end] = sample_root_slate(
                    n_samples, sigma=sigma, correlation=correlation
                )
            else:
                full[:, slate_start:slate_end] = sampler._generate_slate(
                    full, slate_idx, n_samples
                )

    target = full[:, TARGET_INDICES]
    if return_full:
        return target, full
    return target


def save_references(
    n_seeds,
    edge_prob,
    reference_output_dir,
    n_samples,
    y2_values,
    y5_values,
    sigma=1.0,
    correlation=0.5,
    reference_seed=12345,
    save_csv=False,
):
    os.makedirs(reference_output_dir, exist_ok=True)
    manifest_rows = []

    for seed in range(n_seeds):
        dag_adjacency = sample_random_dag(n_slates=N_SLATES, edge_prob=edge_prob, seed=seed)
        target = generate_random_dag_do_distribution(
            n_samples=n_samples,
            dag_adjacency=dag_adjacency,
            y2_values=y2_values,
            y5_values=y5_values,
            sigma=sigma,
            correlation=correlation,
            function_seed=seed,
            sample_seed=reference_seed + seed,
        )

        output_path = reference_data_path(reference_output_dir, edge_prob, seed)
        np.savez(
            output_path,
            target_samples=target,
            target_indices=TARGET_INDICES,
            target_columns=np.array(TARGET_COLUMNS),
            y2_values=np.asarray(y2_values, dtype=float),
            y5_values=np.asarray(y5_values, dtype=float),
            dag_adjacency=np.asarray(dag_adjacency, dtype=np.float32),
            edge_prob=np.array(edge_prob, dtype=float),
            n_edges=np.array(int(np.sum(dag_adjacency)), dtype=int),
            n_samples=np.array(n_samples, dtype=int),
            sigma=np.array(sigma, dtype=float),
            correlation=np.array(correlation, dtype=float),
            seed=np.array(seed, dtype=int),
            function_seed=np.array(seed, dtype=int),
            reference_seed=np.array(reference_seed, dtype=int),
            sample_seed=np.array(reference_seed + seed, dtype=int),
        )

        csv_path = ""
        if save_csv:
            csv_path = os.path.splitext(output_path)[0] + ".csv"
            pd.DataFrame(target, columns=TARGET_COLUMNS).to_csv(csv_path, index=False)

        manifest_rows.append(
            {
                "seed": seed,
                "edge_prob": edge_prob,
                "n_edges": int(np.sum(dag_adjacency)),
                "path": output_path,
                "csv_path": csv_path,
                "n_samples": n_samples,
                "reference_seed": reference_seed,
                "sample_seed": reference_seed + seed,
            }
        )
        print(f"Saved reference data: seed={seed}, path={output_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(reference_output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def save_training_data(
    n_seeds,
    edge_prob,
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
        dag_adjacency = sample_random_dag(n_slates=N_SLATES, edge_prob=edge_prob, seed=seed)
        sampler = TrueSampler_random(
            dag_adjacency=dag_adjacency,
            sigma=sigma,
            correlation=correlation,
            function_seed=seed,
        )

        for n_samples_train in training_samples:
            train_data = sampler.sample(n_samples_train)
            output_path = training_data_path(training_output_dir, edge_prob, n_samples_train, seed)

            np.savez(
                output_path,
                train_data=train_data,
                dag_adjacency=np.asarray(dag_adjacency, dtype=np.float32),
                n_edges=np.array(int(np.sum(dag_adjacency)), dtype=int),
                n_samples_train=np.array(n_samples_train, dtype=int),
                edge_prob=np.array(edge_prob, dtype=float),
                seed=np.array(seed, dtype=int),
                function_seed=np.array(seed, dtype=int),
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
                    "edge_prob": edge_prob,
                    "n_edges": int(np.sum(dag_adjacency)),
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


def save_graphs(n_seeds, edge_prob, graph_output_dir, save_csv=True):
    os.makedirs(graph_output_dir, exist_ok=True)
    manifest_rows = []

    for seed in range(n_seeds):
        dag_adjacency = sample_random_dag(n_slates=N_SLATES, edge_prob=edge_prob, seed=seed)
        output_path, csv_paths = save_graph_structure(
            graph_output_dir=graph_output_dir,
            edge_prob=edge_prob,
            seed=seed,
            dag_adjacency=dag_adjacency,
            save_csv=save_csv,
        )
        manifest_rows.append(
            {
                "seed": seed,
                "edge_prob": edge_prob,
                "n_edges": int(np.sum(dag_adjacency)),
                "path": output_path,
                **csv_paths,
            }
        )
        print(f"Saved graph structure: seed={seed}, path={output_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(graph_output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate random-DAG reference data, graph metadata, and training "
            "datasets for distribution comparison."
        )
    )
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--edge_prob", type=float, default=0.5)
    parser.add_argument("--reference_seed", type=int, default=12345)
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
        default=[0.5, 0.5, 0.5, 0.5, 0.5],
        help="Five intervention values for Y2; defaults to all 0.5.",
    )
    parser.add_argument(
        "--y5_values",
        type=float,
        nargs=5,
        default=[0.5, 0.5, 0.5, 0.5, 0.5],
        help="Five intervention values for Y5; defaults to all 0.5.",
    )
    parser.add_argument(
        "--reference_output_dir",
        type=str,
        default="./data/random_dag_distribution_comparison/references",
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="./data/random_dag_distribution_comparison/training",
    )
    parser.add_argument(
        "--graph_output_dir",
        type=str,
        default="./data/random_dag_distribution_comparison/graphs",
    )
    parser.add_argument("--save_reference_csv", action="store_true")
    parser.add_argument("--save_training_csv", action="store_true")
    parser.add_argument("--skip_reference", action="store_true")
    parser.add_argument("--skip_training_data", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    parser.add_argument("--no_graph_csv", action="store_true")
    args = parser.parse_args()

    y2_values = intervention_vector("y2_values", args.y2_values)
    y5_values = intervention_vector("y5_values", args.y5_values)

    print("\nRandom-DAG data generation")
    print(f"n_seeds: {args.n_seeds}")
    print(f"edge_prob: {args.edge_prob}")
    print(f"training sizes: {args.training_samples}")
    print(f"Y2 intervention: {y2_values}")
    print(f"Y5 intervention: {y5_values}")

    if not args.skip_reference:
        manifest_path = save_references(
            n_seeds=args.n_seeds,
            edge_prob=args.edge_prob,
            reference_output_dir=args.reference_output_dir,
            n_samples=args.n_samples,
            y2_values=y2_values,
            y5_values=y5_values,
            sigma=args.sigma,
            correlation=args.correlation,
            reference_seed=args.reference_seed,
            save_csv=args.save_reference_csv,
        )
        print(f"\nSaved reference manifest to: {manifest_path}")

    if not args.skip_graph:
        manifest_path = save_graphs(
            n_seeds=args.n_seeds,
            edge_prob=args.edge_prob,
            graph_output_dir=args.graph_output_dir,
            save_csv=not args.no_graph_csv,
        )
        print(f"\nSaved graph manifest to: {manifest_path}")

    if not args.skip_training_data:
        manifest_path = save_training_data(
            n_seeds=args.n_seeds,
            edge_prob=args.edge_prob,
            training_samples=args.training_samples,
            training_output_dir=args.training_output_dir,
            sigma=args.sigma,
            correlation=args.correlation,
            save_csv=args.save_training_csv,
        )
        print(f"\nSaved training data manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
