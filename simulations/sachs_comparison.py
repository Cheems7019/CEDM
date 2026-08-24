"""
sachs_comparison.py

Compares the conditional diffusion framework against the standard bidirectional
framework on the Sachs 11-node protein signaling DAG.

Frameworks compared:
  - Conditional: 11 groups (one variable per group, G0=PKC ... G10=Akt),
    each group's model conditions only on its parent groups in the DAG.
  - Standard: single joint model over all 11 features.

Data: TrueSampler_sachs (11 nodes, 17 edges).
Conditioning on (Plcg, PKC, PKA, Raf); estimating (PIP3, Jnk, P38, PIP2, Mek, Erk, Akt).
Metric: MSE/MAE averaged over M conditioning vectors and 7 target variables.

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
from utils.utils_data import TrueSampler_sachs, compute_conditional_expectation_sachs

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
    parser = argparse.ArgumentParser(description="Sachs DAG: Conditional framework vs Standard framework comparison")
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
    d_in = 11  # 11 nodes in the Sachs DAG

    # Standard bidirectional model.
    d_time_standard = 128
    hidden_dims_standard = [1024, 512, 512, 512, 256]
    lr_standard = 5e-5

    # Conditional model (11 separate per-group models, smaller capacity each).
    hidden_dims_conditional = [512, 256, 256, 256, 128]
    dim_t_conditional = 128
    lr_conditional = 5e-5

    n_steps = 1000
    n_epochs = 3000

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/sachs_comparison", exist_ok=True)

    # Conditional framework: one group per variable (G0=PKC, G1=Plcg, ..., G10=Akt).
    groups_conditional = torch.arange(d_in, dtype=torch.long)

    # Sachs DAG adjacency (11x11).
    # PKC->PKA,Raf,Jnk,P38,Mek; Plcg->PIP3,PIP2; PKA->Raf,Jnk,P38,Mek,Erk,Akt;
    # PIP3->PIP2; Raf->Mek; Mek->Erk; Erk->Akt.
    A_conditional = torch.zeros(11, 11, dtype=torch.float32)
    # PKC (X0) edges
    A_conditional[0, 2] = 1  # PKC → PKA
    A_conditional[0, 4] = 1  # PKC → Raf
    A_conditional[0, 5] = 1  # PKC → Jnk
    A_conditional[0, 6] = 1  # PKC → P38
    A_conditional[0, 8] = 1  # PKC → Mek
    # Plcg (X1) edges
    A_conditional[1, 3] = 1  # Plcg → PIP3
    A_conditional[1, 7] = 1  # Plcg → PIP2
    # PKA (X2) edges
    A_conditional[2, 4] = 1  # PKA → Raf
    A_conditional[2, 5] = 1  # PKA → Jnk
    A_conditional[2, 6] = 1  # PKA → P38
    A_conditional[2, 8] = 1  # PKA → Mek
    A_conditional[2, 9] = 1  # PKA → Erk
    A_conditional[2, 10] = 1  # PKA → Akt
    # PIP3 (X3) edges
    A_conditional[3, 7] = 1  # PIP3 → PIP2
    # Raf (X4) edges
    A_conditional[4, 8] = 1  # Raf → Mek
    # Mek (X8) edges
    A_conditional[8, 9] = 1  # Mek → Erk
    # Erk (X9) edges
    A_conditional[9, 10] = 1  # Erk → Akt

    # Standard framework: all features in a single group.
    groups_standard = torch.zeros(d_in, dtype=torch.long)
    A_standard = torch.ones(1, 1, dtype=torch.float32)

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
    print("FRAMEWORK CONFIGURATIONS (Sachs Protein Signaling DAG):")
    print("="*70)
    
    print("\nNEW CONDITIONAL FRAMEWORK (11-group DAG, one variable per group):")
    print("  Groups:")
    print("    G0: X0 (PKC) - conditioning variable")
    print("    G1: X1 (Plcg) - conditioning variable")
    print("    G2: X2 (PKA) - conditioning variable")
    print("    G3: X3 (PIP3) - target variable")
    print("    G4: X4 (Raf) - conditioning variable")
    print("    G5: X5 (Jnk) - target variable")
    print("    G6: X6 (P38) - target variable")
    print("    G7: X7 (PIP2) - target variable")
    print("    G8: X8 (Mek) - target variable")
    print("    G9: X9 (Erk) - target variable")
    print("    G10: X10 (Akt) - target variable")
    print("  Variable-level Adjacency Matrix (11x11):")
    print("       ", " ".join([f"X{j:2d}" for j in range(11)]))
    for i in range(11):
        row_str = f"  X{i:2d}  "
        for j in range(11):
            row_str += f" {int(A_conditional[i,j])}  "
        print(row_str)
    print("  Training: Separate conditional model per group (11 models)")
    print("  Conditioning: Each group conditions ONLY on parent groups")
    print(f"  Capacity per group: hidden_dims={hidden_dims_conditional}, dim_t={dim_t_conditional}")
    print(f"  Learning rate: {lr_conditional}")
    
    print("\nSTANDARD FRAMEWORK (bidirectional):")
    print("  Groups: All features (X0-X10) in Group 0")
    print("  Structure: Fully connected")
    print("  Adjacency Matrix:")
    print("      G0")
    print(f"  G0    {int(A_standard[0,0])}")
    print("  Training: Single joint model with masked connections")
    print(f"  Capacity: hidden_dims={hidden_dims_standard}, dim_t={d_time_standard}")
    print(f"  Learning rate: {lr_standard}")

    results = []

    print(f"\nConditioning on M={n_samples_test} randomly sampled vectors for (PKC, Plcg, PKA, Raf)")
    print(f"Estimating: PIP3, Jnk, P38, PIP2, Mek, Erk, Akt (7 target variables) for each conditioning vector")
    print(f"Monte Carlo size D={D} per conditioning vector")

    scaler_suffix = "_std" if args.use_standard_scaler else "_qt"
    results_file = f"./results/sachs_comparison/sachs_comparison_results{scaler_suffix}.csv"

    if os.path.exists(results_file):
        os.remove(results_file)

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        
        seed_everything(seed)

        # Sample conditioning vectors once per seed; reused across all training sizes.
        sampler = TrueSampler_sachs(sigma=sigma)
        M = n_samples_test
        cond_data = sampler.sample(M)
        cond_vars_raw = cond_data[:, [1, 0, 2, 4]]  # [Plcg, PKC, PKA, Raf]
        
        print(f"\nSampled M={M} conditioning vectors for this seed")
        print(f"Plcg range: [{cond_vars_raw[:, 0].min():.3f}, {cond_vars_raw[:, 0].max():.3f}]")
        print(f"PKC range:  [{cond_vars_raw[:, 1].min():.3f}, {cond_vars_raw[:, 1].max():.3f}]")
        print(f"PKA range:  [{cond_vars_raw[:, 2].min():.3f}, {cond_vars_raw[:, 2].max():.3f}]")
        print(f"Raf range:  [{cond_vars_raw[:, 3].min():.3f}, {cond_vars_raw[:, 3].max():.3f}]")
        
        # Compute true conditional means once per seed.
        print(f"\nComputing true conditional means for M={M} vectors...")
        true_means = np.zeros((M, 7))  # Shape: (M, 7) for 7 target variables
        
        for m in range(M):
            if m % 50 == 0:
                print(f"  Progress: {m}/{M}")
            
            plcg, pkc, pka, raf = cond_vars_raw[m]
            result = compute_conditional_expectation_sachs(plcg, pkc, pka, raf, sigma=sigma, n_monte_carlo=100000)
            true_means[m] = np.array([
                result['PIP3'], result['Jnk'], result['P38'], result['PIP2'],
                result['Mek'], result['Erk'], result['Akt']
            ])
        
        print(f"True means computed. Shape: {true_means.shape}")
        print(f"True means[0]: PIP3={true_means[0,0]:.3f}, Jnk={true_means[0,1]:.3f}, P38={true_means[0,2]:.3f}")
        
        # ========================================================================
        # LOOP OVER TRAINING SAMPLE SIZES (using same conditioning vectors)
        # ========================================================================
        for n_samples_train in training_samples_list:
            print(f"\n{'='*70}")
            print(f"SEED {seed} | TRAINING SAMPLE SIZE: {n_samples_train}")
            print(f"{'='*70}")

            # Generate training data (sampler already created once per seed)
            train_data = sampler.sample(n_samples_train)
            
            print(f"\nTraining data shape: {train_data.shape}")
            print(f"PKC (X0) mean: {train_data[:, 0].mean():.4f}")
            print(f"Plcg (X1) mean: {train_data[:, 1].mean():.4f}")
            print(f"PKA (X2) mean: {train_data[:, 2].mean():.4f}")
            print(f"Raf (X4) mean: {train_data[:, 4].mean():.4f}")
            print(f"Target nodes - PIP3 (X3) mean: {train_data[:, 3].mean():.4f}")
            print(f"Target nodes - Jnk (X5) mean: {train_data[:, 5].mean():.4f}")
            print(f"Target nodes - Mek (X8) mean: {train_data[:, 8].mean():.4f}")
            
            # Preprocess data (shared for both frameworks)
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

            for pattern_name, pattern_config in patterns.items():
                groups = pattern_config["groups"]
                A = pattern_config["A"]
                framework = pattern_config["framework"]
                
                print(f"\n{'='*60}")
                print(f"TRAINING {pattern_name.upper()} ({framework} framework)")
                print(f"{'='*60}")
                
                seed_everything(seed)

                if framework == "conditional":
                    lr = lr_conditional
                    print(f"Learning rate: {lr}")
                    print("Training separate conditional models for each group...")
                    
                    models = train_all_conditional_ddpms(
                        train_data=train_data_norm_tensor,
                        groups=groups,
                        A=A.cpu().numpy() if isinstance(A, torch.Tensor) else A,
                        n_epochs=n_epochs,
                        lr=lr,
                        hidden_dims=hidden_dims_conditional,
                        dim_t=dim_t_conditional,
                        n_steps=n_steps,
                        device=device,
                        verbose=True,
                    )
                    
                    print(f"\nEstimating conditional means with conditional framework...")
                    print(f"  Using fixed M={M} conditioning vectors from this seed")

                    if args.use_standard_scaler:
                        cond_data_norm = scaler.transform(cond_data)
                    else:
                        cond_data_norm = qt.transform(cond_data)
                    
                    print(f"  Generating D={D} samples for all M={M} conditioning vectors...")

                    # Conditioning variables in normalized space.
                    do_vars = {}
                    do_vars[1] = torch.tensor(cond_data_norm[:, 1], dtype=torch.float32, device=device)  # Plcg
                    do_vars[0] = torch.tensor(cond_data_norm[:, 0], dtype=torch.float32, device=device)  # PKC
                    do_vars[2] = torch.tensor(cond_data_norm[:, 2], dtype=torch.float32, device=device)  # PKA
                    do_vars[4] = torch.tensor(cond_data_norm[:, 4], dtype=torch.float32, device=device)  # Raf

                    sample_vars = [3, 5, 6, 7, 8, 9, 10]  # PIP3, Jnk, P38, PIP2, Mek, Erk, Akt

                    # Returns shape [D, M, 7] in normalized space.
                    samples_norm = generate_conditional_samples(
                        models=models,
                        groups=groups,
                        A=A.cpu().numpy() if isinstance(A, torch.Tensor) else A,
                        do_vars=do_vars,
                        sample_vars=sample_vars,
                        n_samples=D,
                        device=device,
                        verbose=False
                    )
                    
                    samples_norm_np = samples_norm.cpu().numpy()  # [D, M, 7]
                    print(f"  Generated samples shape (normalized): {samples_norm_np.shape}")

                    print(f"  Converting to original scale...")
                    samples_orig = np.empty((D, M, 7))

                    for d in range(D):
                        if d % 10 == 0:
                            print(f"    Draw {d}/{D}")

                        # Reconstruct the full 11-dim vector to use the scaler's inverse_transform.
                        full_norm = np.zeros((M, 11))
                        full_norm[:, 0] = cond_data_norm[:, 0]   # PKC
                        full_norm[:, 1] = cond_data_norm[:, 1]   # Plcg
                        full_norm[:, 2] = cond_data_norm[:, 2]   # PKA
                        full_norm[:, 3] = samples_norm_np[d, :, 0]  # PIP3 (sampled)
                        full_norm[:, 4] = cond_data_norm[:, 4]   # Raf
                        full_norm[:, 5] = samples_norm_np[d, :, 1]  # Jnk (sampled)
                        full_norm[:, 6] = samples_norm_np[d, :, 2]  # P38 (sampled)
                        full_norm[:, 7] = samples_norm_np[d, :, 3]  # PIP2 (sampled)
                        full_norm[:, 8] = samples_norm_np[d, :, 4]  # Mek (sampled)
                        full_norm[:, 9] = samples_norm_np[d, :, 5]  # Erk (sampled)
                        full_norm[:, 10] = samples_norm_np[d, :, 6]  # Akt (sampled)

                        if args.use_standard_scaler:
                            full_orig = scaler.inverse_transform(full_norm)
                        else:
                            full_orig = qt.inverse_transform(full_norm)
                        
                        # Extract target variables: PIP3, Jnk, P38, PIP2, Mek, Erk, Akt
                        samples_orig[d] = full_orig[:, [3, 5, 6, 7, 8, 9, 10]]  # Shape: (M, 7)
                    
                    print(f"  Samples in original space. Shape: {samples_orig.shape}")
                    
                    # ========================================================
                    # Verbose shape check
                    # ========================================================
                    print(f"\n  === SHAPE CHECK (Conditional Framework) ===")
                    print(f"  true_means.shape == (M={M}, 7): {true_means.shape}")
                    print(f"  samples_norm.shape == (D={D}, M={M}, 7): {samples_norm_np.shape}")
                    print(f"  samples_orig.shape == (D={D}, M={M}, 7): {samples_orig.shape}")
                    assert true_means.shape == (M, 7), f"Expected true_means shape ({M}, 7), got {true_means.shape}"
                    assert samples_orig.shape == (D, M, 7), f"Expected samples_orig shape ({D}, {M}, 7), got {samples_orig.shape}"
                    print(f"  ✓ All shapes correct!\n")
                    
                    # ========================================================
                    # Compute mean estimates and errors
                    # ========================================================
                    est_means = samples_orig.mean(axis=0)  # Shape: (M, 7)
                    
                    # Compute MSE/MAE averaged over all (M, 7) entries
                    mse = np.mean((est_means - true_means)**2)
                    mae = np.mean(np.abs(est_means - true_means))
                    
                    # Store results
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
                    print(f"  Est means[0] (first conditioning vector): PIP3={est_means[0,0]:.3f}, Jnk={est_means[0,1]:.3f}, P38={est_means[0,2]:.3f}")
                    print(f"  True means[0] (first conditioning vector): PIP3={true_means[0,0]:.3f}, Jnk={true_means[0,1]:.3f}, P38={true_means[0,2]:.3f}")
                
                else:  # framework == "standard"
                    noise_pred_network = MLPDiffusionContinuous(
                        d_in=d_in,
                        hidden_dims=hidden_dims_standard,
                        dim_t=d_time_standard,
                        groups_0=groups,
                        A=A
                    )
                    tabular_ddpm = MyDDPM(network=noise_pred_network, n_steps=n_steps, device=device)
                    optimizer = optim.Adam(tabular_ddpm.network.parameters(), lr=lr_standard)
                    
                    print(f"Model parameters: {sum(p.numel() for p in noise_pred_network.parameters()):,}")
                    print(f"Learning rate: {lr_standard}")
                    print(f"Training for {n_epochs} epochs...")
                    
                    training_loop(
                        train_data_norm_tensor,
                        tabular_ddpm,
                        n_epochs,
                        optimizer,
                        store_path=None
                    )
                    
                    print(f"\nEstimating conditional means with standard framework...")
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
                        
                        batch_data = np.tile(cond_data, (mc_batch_size, 1))  # (mc_batch_size*M, 11)

                        if args.use_standard_scaler:
                            batch_data_norm = scaler.transform(batch_data)
                        else:
                            batch_data_norm = qt.transform(batch_data)
                        
                        input_norm = torch.tensor(batch_data_norm, dtype=torch.float32)
                        
                        # Observe Plcg, PKC, PKA, Raf (X0, X1, X2, X4); impute the rest.
                        input_mask = torch.ones(input_norm.shape, dtype=torch.float32)
                        input_mask[:, 3] = 0  # Impute X3 (PIP3)
                        input_mask[:, 5] = 0  # Impute X5 (Jnk)
                        input_mask[:, 6] = 0  # Impute X6 (P38)
                        input_mask[:, 7] = 0  # Impute X7 (PIP2)
                        input_mask[:, 8] = 0  # Impute X8 (Mek)
                        input_mask[:, 9] = 0  # Impute X9 (Erk)
                        input_mask[:, 10] = 0  # Impute X10 (Akt)
                        
                        output_norm = generate_imputation(
                            tabular_ddpm, input_norm, input_mask, resampling_steps=20
                        )
                        output_norm_np = output_norm.cpu().detach().numpy()  # Shape: (mc_batch_size*M, 11)
                        
                        if args.use_standard_scaler:
                            output_orig = scaler.inverse_transform(output_norm_np)
                        else:
                            output_orig = qt.inverse_transform(output_norm_np)

                        targets = output_orig[:, [3, 5, 6, 7, 8, 9, 10]]
                        targets_reshaped = targets.reshape(mc_batch_size, M, 7)
                        all_samples_orig.append(targets_reshaped)

                    samples_orig = np.concatenate(all_samples_orig, axis=0)  # (D, M, 7)
                    print(f"  Generated samples shape: {samples_orig.shape}")

                    print(f"\n  === SHAPE CHECK (Standard Framework) ===")
                    print(f"  true_means.shape == (M={M}, 7): {true_means.shape}")
                    print(f"  samples_orig.shape == (D={D}, M={M}, 7): {samples_orig.shape}")
                    assert true_means.shape == (M, 7), f"Expected true_means shape ({M}, 7), got {true_means.shape}"
                    assert samples_orig.shape == (D, M, 7), f"Expected samples_orig shape ({D}, {M}, 7), got {samples_orig.shape}"
                    print(f"  ✓ All shapes correct!\n")
                    
                    # ========================================================
                    # Compute mean estimates and errors
                    # ========================================================
                    est_means = samples_orig.mean(axis=0)  # Shape: (M, 7)
                    
                    # Compute MSE/MAE averaged over all (M, 7) entries
                    mse = np.mean((est_means - true_means)**2)
                    mae = np.mean(np.abs(est_means - true_means))
                    
                    # Store results
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
                    print(f"  Est means[0] (first conditioning vector): PIP3={est_means[0,0]:.3f}, Jnk={est_means[0,1]:.3f}, P38={est_means[0,2]:.3f}")
                    print(f"  True means[0] (first conditioning vector): PIP3={true_means[0,0]:.3f}, Jnk={true_means[0,1]:.3f}, P38={true_means[0,2]:.3f}")

        print(f"\nSaving results after completing seed {seed}...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)

        if len(results_df) > 0:
            print(f"  Results updated with {len(results_df)} total rows")
            seed_data = results_df[results_df['seed'] == seed]
            if len(seed_data) > 0:
                seed_summary = seed_data.groupby(['n_samples_train', 'pattern', 'framework'])['mse'].mean().round(6)
                print(f"  MSE summary for seed {seed}:\n{seed_summary}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

    print("\n" + "="*70)
    print("Creating final summary statistics...")

    overall_summary = results_df.groupby(['n_samples_train', 'pattern', 'framework']).agg({
        'mse': ['mean', 'std', 'count'],
        'mae': ['mean', 'std', 'count']
    }).round(6)
    
    overall_summary.to_csv(f"./results/sachs_comparison/overall_summary{scaler_suffix}.csv")

    print(f"\nFinal results saved to ./results/sachs_comparison/:")
    print(f"- Detailed results: sachs_comparison_results{scaler_suffix}.csv")
    print(f"- Summary: overall_summary{scaler_suffix}.csv")
    print(f"Total results: {len(results_df)} rows")

    print(f"\n{'='*70}")
    print("FRAMEWORK COMPARISON (MSE)")
    print(f"{'='*70}")
    comparison = results_df.groupby(['n_samples_train', 'pattern', 'framework'])['mse'].agg(['mean', 'std']).round(6)
    print(comparison)
    
    print(f"\n{'='*70}")
    print("FRAMEWORK COMPARISON (MAE)")
    print(f"{'='*70}")
    comparison_mae = results_df.groupby(['n_samples_train', 'pattern', 'framework'])['mae'].agg(['mean', 'std']).round(6)
    print(comparison_mae)


if __name__ == "__main__":
    main()
