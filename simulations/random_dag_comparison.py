"""
random_dag_comparison.py

Compares the conditional diffusion framework against the standard bidirectional
framework on randomly sampled DAG structures.

Frameworks compared:
  - Conditional: 6 groups (G0: X0-X4 ... G5: X25-X29), each group's model
    conditions on its parent groups according to the sampled DAG.
  - Standard: single joint model over all 30 features.

Data: TrueSampler_random (6 slates x 5 nodes, random DAG per seed).
Conditioning on Y1 (X0-X4); estimating E[Y2-Y6 | Y1].
Metric: MSE/MAE averaged over M conditioning vectors and 25 target variables.

Loop order: outer seed (each seed draws a new random DAG), inner training size.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import argparse
import pickle

from _project_setup import PROJECT_ROOT
from utils.utils_model import MLPDiffusionContinuous
from utils.ddpm import MyDDPM, training_loop, generate_imputation
from utils.utils_data import sample_random_dag, TrueSampler_random, compute_conditional_expectation_random

from utils.conditional_ddpm import (
    train_all_conditional_ddpms,
    generate_conditional_samples,
)


def seed_everything(seed):
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Random DAG structure: Conditional framework vs Standard framework comparison")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=50, help="Number of seeds (each with different random DAG)")
    parser.add_argument("--edge_prob", type=float, default=0.5, help="Probability of edge in random DAG")
    parser.add_argument("--n_samples_test", type=int, default=100, help="Test samples for conditional estimation")
    parser.add_argument("--monte_carlo_size", type=int, default=100, help="Monte Carlo samples")
    parser.add_argument("--monte_carlo_batch_size", type=int, default=100, help="Batch size for Monte Carlo sampling")
    parser.add_argument("--use_standard_scaler", action="store_true", help="Use StandardScaler instead of QuantileTransformer")
    args = parser.parse_args()

    device = args.device
    n_seeds = args.n_seeds
    edge_prob = args.edge_prob
    n_samples_test = args.n_samples_test
    D = args.monte_carlo_size
    D_batch = args.monte_carlo_batch_size if args.monte_carlo_batch_size is not None else D

    training_samples_list = [500, 1000, 2000, 5000]
    print(f"Testing training sample sizes: {training_samples_list}")
    print(f"Random DAG edge probability: {edge_prob}")

    sigma = 1.0
    correlation = 0.5
    d_in = 30  # 6 slates x 5 nodes

    d_time_standard = 128
    hidden_dims_standard = [1024, 512, 512, 512, 256]

    # Smaller per-group capacity since the conditional framework trains 6 separate models.
    hidden_dims_conditional = [512, 256, 256, 256, 128]
    dim_t_conditional = 128

    n_steps = 1000
    n_epochs = 3000
    lr = 5e-5

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/random_dag_comparison", exist_ok=True)
    os.makedirs("./results/random_dag_comparison/dags", exist_ok=True)

    scaler_name = "ss" if args.use_standard_scaler else "qt"
    results_file = f"./results/random_dag_comparison/random_dag_comparison_results_{scaler_name}.csv"

    # Conditional framework: 6 groups, one per slate.
    groups_conditional = torch.zeros(d_in, dtype=torch.long)
    for i in range(0, 5):
        groups_conditional[i] = 0    # Y1 (X0-X4)  -> G0
    for i in range(5, 10):
        groups_conditional[i] = 1    # Y2 (X5-X9)  -> G1
    for i in range(10, 15):
        groups_conditional[i] = 2    # Y3 (X10-X14) -> G2
    for i in range(15, 20):
        groups_conditional[i] = 3    # Y4 (X15-X19) -> G3
    for i in range(20, 25):
        groups_conditional[i] = 4    # Y5 (X20-X24) -> G4
    for i in range(25, 30):
        groups_conditional[i] = 5    # Y6 (X25-X29) -> G5

    # Standard framework: all features in a single group.
    groups_standard = torch.zeros(d_in, dtype=torch.long)
    A_standard = torch.ones(1, 1, dtype=torch.float32)

    print(f"\nConditioning on M={n_samples_test} randomly sampled Y1 vectors per seed")
    print(f"Estimating: Y2-Y6 (X5-X29, 25 features total) for each conditioning vector")
    print(f"Monte Carlo size D={D} per conditioning vector")

    results = []

    for seed in range(n_seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed}/{n_seeds-1}")
        print(f"{'='*80}")
        
        seed_everything(seed)

        print(f"\nSampling random DAG with edge_prob={edge_prob}...")
        dag_adjacency = sample_random_dag(n_slates=6, edge_prob=edge_prob, seed=seed)

        print("\nDAG Adjacency Matrix (Y_i -> Y_j):")
        print("     Y1  Y2  Y3  Y4  Y5  Y6")
        for i in range(6):
            row_str = f"Y{i+1}: "
            row_str += "  ".join([str(dag_adjacency[i, j]) for j in range(6)])
            print(row_str)

        n_edges = np.sum(dag_adjacency)
        print(f"\nNumber of edges: {n_edges} / {6*5//2} possible")

        dag_file = f"./results/random_dag_comparison/dags/dag_seed_{seed}.npy"
        np.save(dag_file, dag_adjacency)
        print(f"Saved DAG to {dag_file}")

        # function_seed matches seed for deterministic functional forms per DAG.
        sampler = TrueSampler_random(
            dag_adjacency=dag_adjacency,
            sigma=sigma,
            correlation=correlation,
            function_seed=seed
        )

        # Sample conditioning vectors once per seed; reused across all training sizes.
        M = n_samples_test
        cond_data = sampler.sample(M)
        Y1_raw = cond_data[:, 0:5]  # Y1 conditioning vectors, shape (M, 5)

        print(f"\nSampled M={M} conditioning vectors for this seed")
        print(f"Y1 range: [{Y1_raw.min():.3f}, {Y1_raw.max():.3f}]")
        print(f"Y1 mean: {Y1_raw.mean(axis=0)}")

        # Compute true conditional means once per seed.
        print(f"\nComputing true conditional means for M={M} vectors...")
        true_means = np.zeros((M, 25))

        for m in range(M):
            if m % 50 == 0:
                print(f"  Progress: {m}/{M}")

            true_conditional = compute_conditional_expectation_random(
                Y1_raw[m],
                dag_adjacency=dag_adjacency,
                sigma=sigma,
                correlation=correlation,
                function_seed=seed,
                n_monte_carlo=100000
            )
            true_means[m] = np.concatenate([
                true_conditional['Y2_mean'],
                true_conditional['Y3_mean'],
                true_conditional['Y4_mean'],
                true_conditional['Y5_mean'],
                true_conditional['Y6_mean']
            ])

        print(f"True means computed. Shape: {true_means.shape}")
        print(f"True means[0] (first 3): {true_means[0, 0:3]}")

        A_conditional = torch.tensor(dag_adjacency, dtype=torch.float32)

        for n_samples_train in training_samples_list:
            print(f"\n{'-'*80}")
            print(f"SEED {seed} | Training with {n_samples_train} samples")
            print(f"{'-'*80}")
            
            print(f"Generating {n_samples_train} training samples...")
            train_data = sampler.sample(n_samples_train)

            print(f"Train data shape: {train_data.shape}, range: [{train_data.min():.2f}, {train_data.max():.2f}]")
            print(f"Using fixed M={M} conditioning vectors from this seed")

            if args.use_standard_scaler:
                scaler = StandardScaler()
            else:
                scaler = QuantileTransformer(n_quantiles=min(1000, n_samples_train), output_distribution='normal')

            scaler.fit(train_data)

            train_data_scaled = scaler.transform(train_data)
            X_train = torch.tensor(train_data_scaled, dtype=torch.float32)

            cond_data_scaled = scaler.transform(cond_data)

            print(f"\n{'*'*80}")
            print(f"CONDITIONAL FRAMEWORK")
            print(f"{'*'*80}")

            seed_everything(seed)

            print(f"\nTraining conditional DDPMs for {len(torch.unique(groups_conditional))} groups...")
            conditional_models = train_all_conditional_ddpms(
                train_data=X_train,
                groups=groups_conditional,
                A=A_conditional.cpu().numpy() if isinstance(A_conditional, torch.Tensor) else A_conditional,
                hidden_dims=hidden_dims_conditional,
                dim_t=dim_t_conditional,
                n_steps=n_steps,
                n_epochs=n_epochs,
                lr=lr,
                device=device,
                verbose=False
            )

            print(f"\nGenerating D={D} samples for all M={M} conditioning vectors...")

            do_vars = {}
            for i in range(5):
                do_vars[i] = torch.tensor(cond_data_scaled[:, i], dtype=torch.float32, device=device)

            sample_vars = list(range(5, 30))

            # Returns shape [D, M, 25] in normalized space.
            samples_norm = generate_conditional_samples(
                models=conditional_models,
                groups=groups_conditional,
                A=A_conditional.cpu().numpy() if isinstance(A_conditional, torch.Tensor) else A_conditional,
                do_vars=do_vars,
                sample_vars=sample_vars,
                n_samples=D,
                device=device,
                verbose=False
            )

            samples_norm_np = samples_norm.cpu().numpy()  # [D, M, 25]
            print(f"Generated samples shape (normalized): {samples_norm_np.shape}")

            print(f"Converting to original scale...")
            samples_orig = np.empty((D, M, 25))

            for d in range(D):
                # Reconstruct the full 30-dim vector to use the scaler's inverse_transform.
                full_norm = np.zeros((M, 30))
                full_norm[:, 0:5] = cond_data_scaled[:, 0:5]
                full_norm[:, 5:30] = samples_norm_np[d]

                full_orig = scaler.inverse_transform(full_norm)

                samples_orig[d] = full_orig[:, 5:30]

            print(f"Samples in original space. Shape: {samples_orig.shape}")

            print(f"\n=== SHAPE CHECK (Conditional Framework) ===")
            print(f"true_means.shape == (M={M}, 25): {true_means.shape}")
            print(f"samples_norm.shape == (D={D}, M={M}, 25): {samples_norm_np.shape}")
            print(f"samples_orig.shape == (D={D}, M={M}, 25): {samples_orig.shape}")
            assert true_means.shape == (M, 25), f"Expected true_means shape ({M}, 25), got {true_means.shape}"
            assert samples_orig.shape == (D, M, 25), f"Expected samples_orig shape ({D}, {M}, 25), got {samples_orig.shape}"
            print(f"All shapes correct.\n")

            est_means = samples_orig.mean(axis=0)
            conditional_error = np.mean((est_means - true_means)**2)
            conditional_mae = np.mean(np.abs(est_means - true_means))

            print(f"\nConditional Framework Results:")
            print(f"  MSE: {conditional_error:.6f}")
            print(f"  MAE: {conditional_mae:.6f}")
            print(f"  Est means[0] (first conditioning vector, first 3): {est_means[0, 0:3]}")
            print(f"  True means[0] (first conditioning vector, first 3): {true_means[0, 0:3]}")

            print(f"\n{'*'*80}")
            print(f"STANDARD BIDIRECTIONAL FRAMEWORK")
            print(f"{'*'*80}")

            seed_everything(seed)

            print(f"\nTraining standard bidirectional DDPM...")
            model_standard = MLPDiffusionContinuous(
                d_in=d_in,
                hidden_dims=hidden_dims_standard,
                dim_t=d_time_standard,
                groups_0=groups_standard,
                A=A_standard
            ).to(device)
            
            ddpm_standard = MyDDPM(model_standard, n_steps=n_steps, device=device)
            optimizer_standard = optim.Adam(model_standard.parameters(), lr=lr)

            training_loop(
                X_train,
                ddpm_standard,
                n_epochs,
                optimizer_standard,
                device=device,
                store_path=None
            )

            print(f"\nGenerating D={D} samples for all M={M} conditioning vectors...")

            batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])
            all_samples_orig = []

            for batch_idx, mc_batch_size in enumerate(batch_sizes):
                print(f"  Batch {batch_idx+1}/{len(batch_sizes)}: Generating {mc_batch_size} samples...")

                # np.tile repeats rows mc_batch_size times (mc-major order),
                # matching the reshape(mc_batch_size, M, 25) below.
                batch_data = np.tile(cond_data, (mc_batch_size, 1))

                batch_data_norm = scaler.transform(batch_data)
                input_norm = torch.tensor(batch_data_norm, dtype=torch.float32)

                # Observe Y1 (X0-X4), impute Y2-Y6 (X5-X29).
                input_mask = torch.ones(input_norm.shape, dtype=torch.float32)
                input_mask[:, 5:30] = 0

                output_norm = generate_imputation(
                    ddpm_standard, input_norm, input_mask, resampling_steps=20
                )
                output_norm_np = output_norm.cpu().detach().numpy()

                output_orig = scaler.inverse_transform(output_norm_np)

                Y2_Y6 = output_orig[:, 5:30]
                Y2_Y6_reshaped = Y2_Y6.reshape(mc_batch_size, M, 25)

                all_samples_orig.append(Y2_Y6_reshaped)

            samples_orig = np.concatenate(all_samples_orig, axis=0)  # [D, M, 25]
            print(f"Generated samples shape: {samples_orig.shape}")

            print(f"\n=== SHAPE CHECK (Standard Framework) ===")
            print(f"true_means.shape == (M={M}, 25): {true_means.shape}")
            print(f"samples_orig.shape == (D={D}, M={M}, 25): {samples_orig.shape}")
            assert true_means.shape == (M, 25), f"Expected true_means shape ({M}, 25), got {true_means.shape}"
            assert samples_orig.shape == (D, M, 25), f"Expected samples_orig shape ({D}, {M}, 25), got {samples_orig.shape}"
            print(f"All shapes correct.\n")

            est_means = samples_orig.mean(axis=0)
            standard_error = np.mean((est_means - true_means)**2)
            standard_mae = np.mean(np.abs(est_means - true_means))

            print(f"\nStandard Framework Results:")
            print(f"  MSE: {standard_error:.6f}")
            print(f"  MAE: {standard_mae:.6f}")
            print(f"  Est means[0] (first conditioning vector, first 3): {est_means[0, 0:3]}")
            print(f"  True means[0] (first conditioning vector, first 3): {true_means[0, 0:3]}")

            result = {
                'seed': seed,
                'n_train': n_samples_train,
                'edge_prob': edge_prob,
                'n_edges': n_edges,
                'conditional_mse': conditional_error,
                'conditional_mae': conditional_mae,
                'standard_mse': standard_error,
                'standard_mae': standard_mae,
            }
            results.append(result)

            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            print(f"\nResults saved to {results_file}")
            print(f"Completed: {len(results)} / {n_seeds * len(training_samples_list)} experiments")

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")

    df = pd.DataFrame(results)

    print("\nOverall Summary (averaged over all seeds and training sizes):")
    print(f"  Conditional MSE: {df['conditional_mse'].mean():.6f} ± {df['conditional_mse'].std():.6f}")
    print(f"  Standard MSE:    {df['standard_mse'].mean():.6f} ± {df['standard_mse'].std():.6f}")
    print(f"  Conditional MAE: {df['conditional_mae'].mean():.6f} ± {df['conditional_mae'].std():.6f}")
    print(f"  Standard MAE:    {df['standard_mae'].mean():.6f} ± {df['standard_mae'].std():.6f}")

    print("\nSummary by training size:")
    summary_by_n = df.groupby('n_train').agg({
        'conditional_mse': ['mean', 'std'],
        'standard_mse': ['mean', 'std'],
        'conditional_mae': ['mean', 'std'],
        'standard_mae': ['mean', 'std'],
    })
    print(summary_by_n)

    print(f"\nFinal results saved to: {results_file}")


if __name__ == "__main__":
    main()
