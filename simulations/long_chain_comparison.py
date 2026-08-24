"""
long_chain_comparison.py

Compares the conditional diffusion framework against the standard bidirectional
framework on a long chain causal structure: Y1->Y2->Y3->Y4->Y5->Y6.

Frameworks compared:
  - Conditional: 6 groups (G0: X0-X4 ... G5: X25-X29), each group's model
    conditions only on its immediate parent in the chain.
  - Standard: single joint model over all 30 features.

Data: TrueSampler_long_chain (6 slates x 5 nodes, chain structure).
Conditioning on Y1 (X0-X4); estimating E[Y2-Y6 | Y1].
Metric: MSE/MAE averaged over M conditioning vectors and 25 target variables.

Loop order: outer seed, inner training size (conditioning vectors are reused
across training sizes within the same seed).
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
from utils.utils_data import TrueSampler_long_chain, compute_conditional_expectation_long_chain

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
    parser = argparse.ArgumentParser(description="Long chain structure: Conditional framework vs Standard framework comparison")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=50, help="Number of seeds for comparison")
    parser.add_argument("--n_samples_test", type=int, default=100, help="Test samples for conditional estimation")
    parser.add_argument("--monte_carlo_size", type=int, default=100, help="Monte Carlo samples")
    parser.add_argument("--monte_carlo_batch_size", type=int, default=100, help="Batch size for Monte Carlo sampling")
    parser.add_argument("--use_standard_scaler", action="store_true", help="Use StandardScaler instead of QuantileTransformer")
    args = parser.parse_args()

    device = args.device
    n_seeds = args.n_seeds
    n_samples_test = args.n_samples_test
    D = args.monte_carlo_size
    D_batch = args.monte_carlo_batch_size if args.monte_carlo_batch_size is not None else D

    training_samples_list = [500, 1000, 2000, 5000]
    print(f"Testing training sample sizes: {training_samples_list}")

    sigma = 1.0
    correlation = 0.5
    d_in = 30  # 6 slates x 5 nodes

    # Standard bidirectional model.
    d_time_standard = 128
    hidden_dims_standard = [1024, 512, 512, 512, 256]

    # Conditional model (6 separate per-group models, smaller capacity each).
    hidden_dims_conditional = [512, 256, 256, 256, 128]
    dim_t_conditional = 128

    n_steps = 1000
    n_epochs = 3000
    lr = 5e-5

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/long_chain_comparison", exist_ok=True)

    # Conditional framework: G0=Y1(X0-X4) ... G5=Y6(X25-X29).
    groups_conditional = torch.zeros(d_in, dtype=torch.long)
    groups_conditional[0:5] = 0    # Group 0: Y1 (X0-X4)
    groups_conditional[5:10] = 1   # Group 1: Y2 (X5-X9)
    groups_conditional[10:15] = 2  # Group 2: Y3 (X10-X14)
    groups_conditional[15:20] = 3  # Group 3: Y4 (X15-X19)
    groups_conditional[20:25] = 4  # Group 4: Y5 (X20-X24)
    groups_conditional[25:30] = 5  # Group 5: Y6 (X25-X29)

    # Chain adjacency: G0->G1->G2->G3->G4->G5.
    A_conditional = torch.tensor([
        [0, 1, 0, 0, 0, 0],  # G0 → G1
        [0, 0, 1, 0, 0, 0],  # G1 → G2
        [0, 0, 0, 1, 0, 0],  # G2 → G3
        [0, 0, 0, 0, 1, 0],  # G3 → G4
        [0, 0, 0, 0, 0, 1],  # G4 → G5
        [0, 0, 0, 0, 0, 0]   # G5 has no children
    ], dtype=torch.float32)

    # Standard framework: all features in a single group.
    groups_standard = torch.zeros(d_in, dtype=torch.long)
    A_standard = torch.ones((1, 1), dtype=torch.float32)

    patterns = {
        "conditional": {
            "groups": groups_conditional,
            "A": A_conditional,
            "framework": "conditional"
        },
        "standard": {
            "groups": groups_standard,
            "A": A_standard,
            "framework": "standard"
        }
    }
    
    print("\n" + "="*70)
    print("FRAMEWORK CONFIGURATIONS:")
    print("="*70)
    
    print("\nCONDITIONAL FRAMEWORK (6-group long chain):")
    print("  Groups: [0: X0-X4 (Y1)], [1: X5-X9 (Y2)], [2: X10-X14 (Y3)],")
    print("          [3: X15-X19 (Y4)], [4: X20-X24 (Y5)], [5: X25-X29 (Y6)]")
    print("  Structure: Y1 → Y2 → Y3 → Y4 → Y5 → Y6 (long chain)")
    print("  Adjacency Matrix:")
    print("      G0 G1 G2 G3 G4 G5")
    for i in range(6):
        row_str = f"  G{i}    "
        for j in range(6):
            row_str += f"{int(A_conditional[i,j].item())}  "
        print(row_str)
    print("  Training: Separate conditional model per group (6 models)")
    print("  Conditioning: Each group conditions ONLY on parent group")
    print(f"  Capacity per group: hidden_dims={hidden_dims_conditional}, dim_t={dim_t_conditional}")
    
    print("\nSTANDARD FRAMEWORK (bidirectional):")
    print("  Groups: All features (X0-X29) in Group 0")
    print("  Structure: Fully connected")
    print("  Adjacency Matrix:")
    print("      G0")
    print(f"  G0    {int(A_standard[0,0].item())}")
    print("  Training: Single joint model")
    print(f"  Capacity: hidden_dims={hidden_dims_standard}, dim_t={d_time_standard}")

    results = []

    print(f"\nConditioning on M={n_samples_test} randomly sampled Y1 vectors")
    print(f"Estimating: Y2-Y6 (X5-X29, 25 features total) for each conditioning vector")
    print(f"Monte Carlo size D={D} per conditioning vector")

    scaler_suffix = "_std" if args.use_standard_scaler else "_qt"
    results_file = f"./results/long_chain_comparison/long_chain_comparison_results{scaler_suffix}.csv"

    if os.path.exists(results_file):
        os.remove(results_file)

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        
        seed_everything(seed)

        # Sample conditioning vectors once per seed; reused across all training sizes.
        sampler = TrueSampler_long_chain(sigma=sigma, correlation=correlation)
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
            true_result = compute_conditional_expectation_long_chain(
                Y1_raw[m],
                sigma=sigma,
                correlation=correlation,
                n_monte_carlo=100000
            )
            true_means[m] = np.concatenate([
                true_result['Y2_mean'],
                true_result['Y3_mean'],
                true_result['Y4_mean'],
                true_result['Y5_mean'],
                true_result['Y6_mean']
            ])
        
        print(f"True means computed. Shape: {true_means.shape}")
        print(f"True means[0] (first 3): {true_means[0, 0:3]}")
        
        for n_samples_train in training_samples_list:
            print(f"\n{'='*70}")
            print(f"SEED {seed} | TRAINING SAMPLE SIZE: {n_samples_train}")
            print(f"{'='*70}")

            train_data = sampler.sample(n_samples_train)

            print(f"\nTraining data shape: {train_data.shape}")
            print(f"Y1 (X0-X4) mean: {train_data[:, 0:5].mean(axis=0)}")
            print(f"Y6 (X25-X29) mean: {train_data[:, 25:30].mean(axis=0)}")
            
            if args.use_standard_scaler:
                print("Using StandardScaler for normalization")
                scaler = StandardScaler()
                train_data_norm = scaler.fit_transform(train_data)
            else:
                n_quantiles = min(1000, n_samples_train)
                print(f"Using QuantileTransformer with {n_quantiles} quantiles")
                qt = QuantileTransformer(output_distribution="normal", random_state=seed, 
                                       n_quantiles=n_quantiles,
                                       subsample=min(100000, n_samples_train))
                train_data_norm = qt.fit_transform(train_data)
            
            train_data_norm_tensor = torch.tensor(train_data_norm, dtype=torch.float32)
            
            for pattern_name, config in patterns.items():
                print(f"\n{'='*50}")
                print(f"Testing {pattern_name.upper()} framework")
                print(f"{'='*50}")

                seed_everything(seed)
                
                groups = config["groups"]
                A = config["A"]
                framework = config["framework"]
                
                if framework == "conditional":
                    print(f"Training 6 conditional models...")
                    models = train_all_conditional_ddpms(
                        train_data=train_data_norm_tensor,
                        groups=groups,
                        A=A.cpu().numpy() if isinstance(A, torch.Tensor) else A,
                        hidden_dims=hidden_dims_conditional,
                        dim_t=dim_t_conditional,
                        n_steps=n_steps,
                        n_epochs=n_epochs,
                        lr=lr,
                        device=device,
                        verbose=True
                    )
                    
                    # Models trained - no checkpoint saving
                    
                    print(f"\nEstimating E[Y2-Y6 | Y1] using conditional framework...")
                    print(f"  Using fixed M={M} conditioning vectors from this seed")

                    if args.use_standard_scaler:
                        cond_data_norm = scaler.transform(cond_data)
                    else:
                        cond_data_norm = qt.transform(cond_data)

                    print(f"  Generating D={D} samples for all M={M} conditioning vectors...")

                    do_vars = {}
                    for i in range(5):
                        do_vars[i] = torch.tensor(cond_data_norm[:, i], dtype=torch.float32, device=device)

                    sample_vars = list(range(5, 30))

                    # Returns shape [D, M, 25] in normalized space.
                    samples_norm = generate_conditional_samples(
                        models=models,
                        groups=groups,
                        A=A.cpu().numpy() if isinstance(A, torch.Tensor) else A,
                        do_vars=do_vars,
                        sample_vars=sample_vars,
                        n_samples=D,
                        device=device
                    )
                    
                    samples_norm_np = samples_norm.cpu().numpy()  # [D, M, 25]
                    print(f"  Generated samples shape (normalized): {samples_norm_np.shape}")

                    print(f"  Converting to original scale...")
                    samples_orig = np.empty((D, M, 25))

                    for d in range(D):
                        if d % 10 == 0:
                            print(f"    Draw {d}/{D}")

                        # Reconstruct the full 30-dim vector to use the scaler's inverse_transform.
                        full_norm = np.zeros((M, 30))
                        full_norm[:, 0:5] = cond_data_norm[:, 0:5]
                        full_norm[:, 5:30] = samples_norm_np[d]

                        if args.use_standard_scaler:
                            full_orig = scaler.inverse_transform(full_norm)
                        else:
                            full_orig = qt.inverse_transform(full_norm)

                        samples_orig[d] = full_orig[:, 5:30]

                    print(f"  Samples in original space. Shape: {samples_orig.shape}")

                    print(f"\n  === SHAPE CHECK (Conditional Framework) ===")
                    print(f"  true_means.shape == (M={M}, 25): {true_means.shape}")
                    print(f"  samples_norm.shape == (D={D}, M={M}, 25): {samples_norm_np.shape}")
                    print(f"  samples_orig.shape == (D={D}, M={M}, 25): {samples_orig.shape}")
                    assert true_means.shape == (M, 25), f"Expected true_means shape ({M}, 25), got {true_means.shape}"
                    assert samples_orig.shape == (D, M, 25), f"Expected samples_orig shape ({D}, {M}, 25), got {samples_orig.shape}"
                    print(f"  All shapes correct.\n")

                    est_means = samples_orig.mean(axis=0)
                    mse = np.mean((est_means - true_means)**2)
                    mae = np.mean(np.abs(est_means - true_means))

                    result = {
                        'n_samples_train': n_samples_train,
                        'seed': seed,
                        'pattern': pattern_name,
                        'framework': framework,
                        'mse': mse,
                        'mae': mae,
                        'use_standard_scaler': args.use_standard_scaler,
                    }
                    
                    results.append(result)
                    
                    print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
                    print(f"  Est means[0] (first conditioning vector, first 3): {est_means[0, 0:3]}")
                    print(f"  True means[0] (first conditioning vector, first 3): {true_means[0, 0:3]}")
                
                else:  # framework == "standard"
                    noise_pred_network = MLPDiffusionContinuous(
                        d_in=d_in,
                        hidden_dims=hidden_dims_standard,
                        dim_t=d_time_standard,
                        groups_0=groups,
                        A=A
                    )
                    tabular_ddpm = MyDDPM(network=noise_pred_network, n_steps=n_steps, device=device)
                    optimizer = optim.Adam(tabular_ddpm.network.parameters(), lr=lr)
                    
                    print(f"Model parameters: {sum(p.numel() for p in noise_pred_network.parameters()):,}")
                    print(f"Training for {n_epochs} epochs...")
                    
                    training_loop(
                        train_data_norm_tensor,
                        tabular_ddpm,
                        n_epochs,
                        optimizer,
                        store_path=None
                    )

                    print(f"\nEstimating E[Y2-Y6 | Y1] using standard framework...")
                    print(f"  Using fixed M={M} conditioning vectors from this seed")

                    if args.use_standard_scaler:
                        cond_data_norm = scaler.transform(cond_data)
                    else:
                        cond_data_norm = qt.transform(cond_data)

                    print(f"  Generating D={D} samples for all M={M} conditioning vectors...")

                    batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])
                    all_samples_orig = []

                    for batch_idx, mc_batch_size in enumerate(batch_sizes):
                        print(f"    Batch {batch_idx+1}/{len(batch_sizes)}: Generating {mc_batch_size} samples...")

                        # np.tile repeats rows mc_batch_size times (mc-major order),
                        # matching the reshape(mc_batch_size, M, 25) below.
                        batch_data = np.tile(cond_data, (mc_batch_size, 1))

                        if args.use_standard_scaler:
                            batch_data_norm = scaler.transform(batch_data)
                        else:
                            batch_data_norm = qt.transform(batch_data)

                        input_norm = torch.tensor(batch_data_norm, dtype=torch.float32)

                        # Observe Y1 (X0-X4), impute Y2-Y6 (X5-X29).
                        input_mask = torch.ones(input_norm.shape, dtype=torch.float32)
                        input_mask[:, 5:30] = 0

                        output_norm = generate_imputation(
                            tabular_ddpm, input_norm, input_mask, resampling_steps=20
                        )
                        output_norm_np = output_norm.cpu().detach().numpy()

                        if args.use_standard_scaler:
                            output_orig = scaler.inverse_transform(output_norm_np)
                        else:
                            output_orig = qt.inverse_transform(output_norm_np)

                        Y2_Y6 = output_orig[:, 5:30]
                        Y2_Y6_reshaped = Y2_Y6.reshape(mc_batch_size, M, 25)

                        all_samples_orig.append(Y2_Y6_reshaped)

                    samples_orig = np.concatenate(all_samples_orig, axis=0)  # [D, M, 25]
                    print(f"  Generated samples shape: {samples_orig.shape}")

                    print(f"\n  === SHAPE CHECK (Standard Framework) ===")
                    print(f"  true_means.shape == (M={M}, 25): {true_means.shape}")
                    print(f"  samples_orig.shape == (D={D}, M={M}, 25): {samples_orig.shape}")
                    assert true_means.shape == (M, 25), f"Expected true_means shape ({M}, 25), got {true_means.shape}"
                    assert samples_orig.shape == (D, M, 25), f"Expected samples_orig shape ({D}, {M}, 25), got {samples_orig.shape}"
                    print(f"  All shapes correct.\n")

                    est_means = samples_orig.mean(axis=0)
                    mse = np.mean((est_means - true_means)**2)
                    mae = np.mean(np.abs(est_means - true_means))

                    result = {
                        'n_samples_train': n_samples_train,
                        'seed': seed,
                        'pattern': pattern_name,
                        'framework': framework,
                        'mse': mse,
                        'mae': mae,
                        'use_standard_scaler': args.use_standard_scaler,
                    }
                    
                    results.append(result)
                    
                    print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
                    print(f"  Est means[0] (first conditioning vector, first 3): {est_means[0, 0:3]}")
                    print(f"  True means[0] (first conditioning vector, first 3): {true_means[0, 0:3]}")

        # Save results after completing all training sizes for this seed
        print(f"\nSaving results after completing seed {seed}...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        
        if len(results_df) > 0:
            print(f"  Results updated with {len(results_df)} total rows")
            # Show summary for this seed across all training sizes
            seed_data = results_df[results_df['seed'] == seed]
            if len(seed_data) > 0:
                seed_summary = seed_data.groupby(['n_samples_train', 'pattern', 'framework'])['mse'].mean().round(6)
                print(f"  MSE summary for seed {seed}:\n{seed_summary}")

    # Final save and analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    print("\n" + "="*70)
    print("Creating final summary statistics...")
    
    # Overall summary
    overall_summary = results_df.groupby(['n_samples_train', 'pattern', 'framework']).agg({
        'mse': ['mean', 'std', 'count'],
        'mae': ['mean', 'std', 'count']
    }).round(6)
    
    overall_summary.to_csv(f"./results/long_chain_comparison/overall_summary{scaler_suffix}.csv")
    
    print(f"\nFinal results saved to ./results/long_chain_comparison/:")
    print(f"- Detailed results: long_chain_comparison_results{scaler_suffix}.csv")
    print(f"- Summary: overall_summary{scaler_suffix}.csv")
    print(f"Total results: {len(results_df)} rows")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("FRAMEWORK COMPARISON (MSE)")
    print(f"{'='*70}")
    comparison = results_df.groupby(['n_samples_train', 'pattern', 'framework'])['mse'].agg(['mean', 'std']).round(6)
    print(comparison)
    
    print(f"\n{'='*70}")
    print("FRAMEWORK COMPARISON (MAE)")
    print(f"{'='*70}")
    comparison = results_df.groupby(['n_samples_train', 'pattern', 'framework'])['mae'].agg(['mean', 'std']).round(6)
    print(comparison)


if __name__ == "__main__":
    main()
