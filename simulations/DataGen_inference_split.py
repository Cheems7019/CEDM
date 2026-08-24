"""
DataGen_inference_split.py

Generate deterministic train/inference split datasets for:
- inference_size_eta_split.py
- inference_size_gamma_split.py
- inference_power_eta_split.py
- inference_power_gamma_split.py

Rules:
- For each (scenario, parameter, n_train, seed), sample once with size n_train + n_infer.
- Deterministic split: first n_train rows -> train, remaining n_infer rows -> infer.
- Save one CSV per combination with columns: split, X0..X29.
"""

import argparse
import os
import random
from typing import Iterable, List

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # torch is optional; data generation does not require it.
    torch = None

from _project_setup import PROJECT_ROOT
from utils.utils_data import TrueSampler_inference_power, TrueSampler_inference_size


TRAIN_TO_INFER = {
    200: 200,
    500: 300,
    1000: 400,
}

DEFAULT_TRAIN_SIZES = [200, 500, 1000]
DEFAULT_ETA_VALUES = [0.5, 0.7, 1.0, 2.0, 3.0]
DEFAULT_GAMMA_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_param(value: float) -> str:
    return f"{value:.10g}"


def validate_train_sizes(train_sizes: Iterable[int]) -> List[int]:
    out = []
    for n_train in train_sizes:
        if n_train not in TRAIN_TO_INFER:
            raise ValueError(
                f"Unsupported n_train={n_train}. Supported keys are {sorted(TRAIN_TO_INFER.keys())}."
            )
        out.append(n_train)
    return out


def save_split_csv(data: np.ndarray, n_train: int, n_infer: int, out_path: str) -> None:
    feature_cols = [f"X{i}" for i in range(30)]
    split = np.array(["train"] * n_train + ["infer"] * n_infer, dtype=object)
    df = pd.DataFrame(data, columns=feature_cols)
    df.insert(0, "split", split)
    df.to_csv(out_path, index=False)


def maybe_generate_file(
    sampler,
    n_train: int,
    n_infer: int,
    out_path: str,
    seed: int,
    overwrite: bool,
) -> bool:
    if os.path.exists(out_path) and not overwrite:
        print(f"  [skip] {out_path}")
        return False

    seed_everything(seed)
    total_n = n_train + n_infer
    data = sampler.sample(total_n)
    save_split_csv(data, n_train, n_infer, out_path)
    print(f"  [write] {out_path}  (train={n_train}, infer={n_infer})")
    return True


def gen_size_eta(n_seeds: int, train_sizes: List[int], overwrite: bool) -> int:
    out_dir = "./data/inference_size_eta_split"
    os.makedirs(out_dir, exist_ok=True)

    sigma = 1.0
    correlation = 0.5
    gamma = 1.0
    eta = 0.0

    writes = 0
    print("\nGenerating: inference_size_eta_split")
    for n_train in train_sizes:
        n_infer = TRAIN_TO_INFER[n_train]
        for seed in range(n_seeds):
            sampler = TrueSampler_inference_size(
                sigma=sigma,
                correlation=correlation,
                gamma=gamma,
                eta=eta,
            )
            out_path = os.path.join(
                out_dir,
                f"size_eta_n{n_train}_m{n_infer}_seed_{seed}.csv",
            )
            writes += int(
                maybe_generate_file(
                    sampler=sampler,
                    n_train=n_train,
                    n_infer=n_infer,
                    out_path=out_path,
                    seed=seed,
                    overwrite=overwrite,
                )
            )
    return writes


def gen_size_gamma(n_seeds: int, train_sizes: List[int], overwrite: bool) -> int:
    out_dir = "./data/inference_size_gamma_split"
    os.makedirs(out_dir, exist_ok=True)

    sigma = 1.0
    correlation = 0.5
    gamma = 0.0
    eta = 1.0

    writes = 0
    print("\nGenerating: inference_size_gamma_split")
    for n_train in train_sizes:
        n_infer = TRAIN_TO_INFER[n_train]
        for seed in range(n_seeds):
            sampler = TrueSampler_inference_size(
                sigma=sigma,
                correlation=correlation,
                gamma=gamma,
                eta=eta,
            )
            out_path = os.path.join(
                out_dir,
                f"size_gamma_n{n_train}_m{n_infer}_seed_{seed}.csv",
            )
            writes += int(
                maybe_generate_file(
                    sampler=sampler,
                    n_train=n_train,
                    n_infer=n_infer,
                    out_path=out_path,
                    seed=seed,
                    overwrite=overwrite,
                )
            )
    return writes


def gen_power_eta(
    n_seeds: int,
    train_sizes: List[int],
    eta_values: List[float],
    overwrite: bool,
) -> int:
    out_dir = "./data/inference_power_eta_split"
    os.makedirs(out_dir, exist_ok=True)

    sigma = 1.0
    correlation = 0.5
    gamma = 1.0
    y3_noise_scale = 0.5

    writes = 0
    print("\nGenerating: inference_power_eta_split")
    for eta in eta_values:
        eta_str = format_param(eta)
        for n_train in train_sizes:
            n_infer = TRAIN_TO_INFER[n_train]
            for seed in range(n_seeds):
                sampler = TrueSampler_inference_power(
                    sigma=sigma,
                    correlation=correlation,
                    gamma=gamma,
                    eta=eta,
                    Y3_noise_scale=y3_noise_scale,
                )
                out_path = os.path.join(
                    out_dir,
                    f"power_eta_eta{eta_str}_n{n_train}_m{n_infer}_seed_{seed}.csv",
                )
                writes += int(
                    maybe_generate_file(
                        sampler=sampler,
                        n_train=n_train,
                        n_infer=n_infer,
                        out_path=out_path,
                        seed=seed,
                        overwrite=overwrite,
                    )
                )
    return writes


def gen_power_gamma(
    n_seeds: int,
    train_sizes: List[int],
    gamma_values: List[float],
    overwrite: bool,
) -> int:
    out_dir = "./data/inference_power_gamma_split"
    os.makedirs(out_dir, exist_ok=True)

    sigma = 1.0
    correlation = 0.5
    eta = 1.0
    y3_noise_scale = 0.5

    writes = 0
    print("\nGenerating: inference_power_gamma_split")
    for gamma in gamma_values:
        gamma_str = format_param(gamma)
        for n_train in train_sizes:
            n_infer = TRAIN_TO_INFER[n_train]
            for seed in range(n_seeds):
                sampler = TrueSampler_inference_power(
                    sigma=sigma,
                    correlation=correlation,
                    gamma=gamma,
                    eta=eta,
                    Y3_noise_scale=y3_noise_scale,
                )
                out_path = os.path.join(
                    out_dir,
                    f"power_gamma_gamma{gamma_str}_n{n_train}_m{n_infer}_seed_{seed}.csv",
                )
                writes += int(
                    maybe_generate_file(
                        sampler=sampler,
                        n_train=n_train,
                        n_infer=n_infer,
                        out_path=out_path,
                        seed=seed,
                        overwrite=overwrite,
                    )
                )
    return writes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic train/inference split datasets for inference experiments."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "size_eta", "size_gamma", "power_eta", "power_gamma"],
        help="Which dataset family to generate. Default: all.",
    )
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of seeds.")
    parser.add_argument(
        "--train_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TRAIN_SIZES,
        help="Training sizes to generate. Must be subset of [200, 500, 1000].",
    )
    parser.add_argument(
        "--eta_values",
        type=float,
        nargs="+",
        default=DEFAULT_ETA_VALUES,
        help="Eta grid for power_eta task.",
    )
    parser.add_argument(
        "--gamma_values",
        type=float,
        nargs="+",
        default=DEFAULT_GAMMA_VALUES,
        help="Gamma grid for power_gamma task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files.",
    )
    args = parser.parse_args()

    train_sizes = validate_train_sizes(args.train_sizes)
    total_writes = 0

    print("=" * 80)
    print("Data generation for inference split experiments")
    print("=" * 80)
    print(f"task:       {args.task}")
    print(f"n_seeds:    {args.n_seeds}")
    print(f"train_sizes:{train_sizes}")
    print(f"infer map:  {TRAIN_TO_INFER}")
    print(f"overwrite:  {args.overwrite}")

    if args.task in ["all", "size_eta"]:
        total_writes += gen_size_eta(args.n_seeds, train_sizes, args.overwrite)
    if args.task in ["all", "size_gamma"]:
        total_writes += gen_size_gamma(args.n_seeds, train_sizes, args.overwrite)
    if args.task in ["all", "power_eta"]:
        total_writes += gen_power_eta(args.n_seeds, train_sizes, args.eta_values, args.overwrite)
    if args.task in ["all", "power_gamma"]:
        total_writes += gen_power_gamma(args.n_seeds, train_sizes, args.gamma_values, args.overwrite)

    print("\n" + "=" * 80)
    print(f"Done. Files written: {total_writes}")
    print("=" * 80)


if __name__ == "__main__":
    main()
