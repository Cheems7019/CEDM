"""
inference_power_eta.py

Conditional Diffusion Framework power analysis testing Y3 conditionally
independent of Y2 given Y1.

Data: TrueSampler_inference_power with gamma=1, eta!=0, Y3_noise_scale=0.5.
Under H1 (eta!=0), a direct Y2->Y3 dependence exists in the data.

Model structure (fork, intentionally misspecified):
  - G0 (Y1 / X0-X9):   unconditional
  - G1 (Y2 / X10-X19): conditional on G0
  - G2 (Y3 / X20-X29): conditional on G0 only (missing Y2->Y3 edge)

Test procedure (per seed):
  1. Sample Y3 D times from the trained model conditioned on observed Y1, Y2.
  2. Compute tau(d) = MCODEC(Y3(d), Y2, Y1) for d=1,...,D  (null distribution).
  3. Compute tau = MCODEC(Y3_observed, Y2, Y1)              (test statistic).
  4. p-value = (1 + #{tau(d) >= tau}) / (D+1).

Power = rejection rate at alpha=0.05 when H1 is true.
Results include p-value, tau, and all tau(d) values.
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
from utils.conditional_ddpm import (
    ConditionalMLPDiffusion,
    ConditionalDDPM,
    train_all_conditional_ddpms,
    generate_conditional_samples
)
from utils.utils_data import TrueSampler_inference_power
from utils.mcodec import mcodec

def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Conditional Diffusion Framework-based power analysis for conditional independence test (eta)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of seeds (datasets) to test")
    parser.add_argument("--monte_carlo_size", type=int, default=100, help="Number of Y3 samples per training sample (D)")
    parser.add_argument("--monte_carlo_batch_size", type=int, default=50, help="Batch size for sampling Y3")
    parser.add_argument("--use_standard_scaler", action="store_true", help="Use StandardScaler instead of QuantileTransformer")
    parser.add_argument("--eta", type=float, default=None, help="Specific eta value to test (overrides eta_values_list)")
    args = parser.parse_args()

    device = args.device
    n_seeds = args.n_seeds
    D = args.monte_carlo_size
    D_batch = args.monte_carlo_batch_size if args.monte_carlo_batch_size is not None else D

    training_samples_list = [200, 500]
    print(f"Testing training sample sizes: {training_samples_list}")

    if args.eta is not None:
        eta_values_list = [args.eta]
        print(f"Testing single eta value: {eta_values_list}")
    else:
        eta_values_list = [0.5, 0.7, 1.0, 2.0, 3.0]
        print(f"Testing eta values: {eta_values_list}")

    sigma = 1.0
    correlation = 0.5
    gamma = 1.0  # Y1->Y3 connection strength; held fixed across eta sweep
    d_in = 30    # total features: 10 per group (Y1, Y2, Y3)

    hidden_dims_conditional = [640, 320, 320, 320, 160]
    dim_t_conditional = 64

    n_steps = 1000
    n_epochs = 3000
    lr = 5e-5

    os.makedirs("./ckpt/inference_power_eta", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/inference_power_eta", exist_ok=True)
    os.makedirs("./data/inference_power_eta", exist_ok=True)

    # Three-group partition: G0=Y1 (X0-X9), G1=Y2 (X10-X19), G2=Y3 (X20-X29).
    groups = torch.zeros(d_in, dtype=torch.long)
    groups[0:10] = 0
    groups[10:20] = 1
    groups[20:30] = 2

    # Fork structure: G0->G1, G0->G2, no G1->G2 edge.
    # Misspecified when eta!=0 because the true data has a Y2->Y3 dependence.
    A = np.array([
        [0, 1, 1],  # G0->G1, G0->G2
        [0, 0, 0],  # G1 has no children (missing Y2->Y3 edge)
        [0, 0, 0],  # G2 has no children
    ], dtype=np.float32)

    print("\nConditional Diffusion Framework configuration for power test:")
    print("  Groups:")
    print("    G0: Y1 (X0-X9) - Unconditional model")
    print("    G1: Y2 (X10-X19) - Conditional on G0")
    print("    G2: Y3 (X20-X29) - Conditional on G0 (fork structure)")
    print("  Adjacency Matrix:")
    print("      G0 G1 G2")
    for i in range(3):
        row_str = f"  G{i}  "
        for j in range(3):
            row_str += f" {int(A[i,j])} "
        print(row_str)
    print("\n  Note: Model is MISSPECIFIED (missing Y2→Y3) to test MCODEC's power")

    results = []

    for eta in eta_values_list:
        print(f"\n{'='*80}")
        print(f"ETA VALUE: {eta} (testing power under H1: Y3 ⊥̸ Y2 | Y1)")
        print(f"{'='*80}")
        
        scaler_suffix = "_std" if args.use_standard_scaler else "_qt"
        results_file = f"./results/inference_power_eta/inference_power_eta{eta}_results{scaler_suffix}.csv"

        if os.path.exists(results_file):
            os.remove(results_file)

        eta_results = []
        
        for n_samples_train in training_samples_list:
            print(f"\n{'='*70}")
            print(f"TRAINING SAMPLE SIZE: {n_samples_train} (eta={eta})")
            print(f"{'='*70}")
            
            for seed in range(n_seeds):
                print(f"\n{'='*50}")
                print(f"SEED {seed} (eta={eta}, n_train={n_samples_train})")
                print(f"{'='*50}")
                
                seed_everything(seed)

                print(f"\n--- Training Conditional Diffusion Models (eta={eta}, n_train={n_samples_train}) ---")
                print(f"    hidden_dims={hidden_dims_conditional}, dim_t={dim_t_conditional}")
                print(f"    Model is misspecified (missing Y2->Y3) for data with eta={eta}")

                sampler = TrueSampler_inference_power(sigma=sigma, correlation=correlation, gamma=gamma, eta=eta, Y3_noise_scale=0.5)
                train_data = sampler.sample(n_samples_train)
                
                train_data_path = f"./data/inference_power_eta/train_eta{eta}_n{n_samples_train}_seed_{seed}.csv"
                train_data_df = pd.DataFrame(train_data, columns=[f'X{i}' for i in range(30)])
                train_data_df.to_csv(train_data_path, index=False)
                print(f"Training data saved to: {train_data_path}")
                
                print(f"Training data shape: {train_data.shape}")
                print(f"Y1 (X0-X9) stats - Mean: {train_data[:, 0:3].mean(axis=0)}")
                print(f"Y2 (X10-X19) stats - Mean: {train_data[:, 10:13].mean(axis=0)}")
                print(f"Y3 (X20-X29) stats - Mean: {train_data[:, 20:23].mean(axis=0)}")
                
                if args.use_standard_scaler:
                    print("Using StandardScaler for normalization")
                    scaler = StandardScaler()
                    train_data_norm = scaler.fit_transform(train_data)
                else:
                    n_quantiles = min(1000, n_samples_train)
                    print(f"Using QuantileTransformer for normalization with {n_quantiles} quantiles")
                    qt = QuantileTransformer(output_distribution="normal", random_state=seed, 
                                           n_quantiles=n_quantiles,
                                           subsample=min(100000, n_samples_train))
                    train_data_norm = qt.fit_transform(train_data)
                
                train_data_norm = torch.tensor(train_data_norm, dtype=torch.float32)

                models = train_all_conditional_ddpms(
                    train_data=train_data_norm,
                    groups=groups,
                    A=A,
                    hidden_dims=hidden_dims_conditional,
                    dim_t=dim_t_conditional,
                    n_steps=n_steps,
                    n_epochs=n_epochs,
                    lr=lr,
                    device=device,
                    verbose=True
                )
                
                checkpoint_dir = f"./ckpt/inference_power_eta/conditional_eta{eta}_n{n_samples_train}_seed_{seed}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                models_path = os.path.join(checkpoint_dir, "models.pkl")
                with open(models_path, 'wb') as f:
                    pickle.dump(models, f)
                print(f"Models saved to: {models_path}")
                
                print(f"\n--- Computing MCODEC conditional independence test ---")

                Y1_train = train_data[:, 0:10]
                Y2_train = train_data[:, 10:20]
                Y3_train = train_data[:, 20:30]

                print(f"  Computing tau = MCODEC(Y3_observed, Y2, Y1)...")
                tau_observed = mcodec(Y3_train, Y2_train, Y1_train)
                print(f"  tau (observed) = {tau_observed:.6f}")

                print(f"  Sampling Y3 {D} times from trained conditional models (given Y1, Y2)...")

                tau_samples = []

                mc_batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])
                
                for batch_idx, mc_batch_size in enumerate(mc_batch_sizes):
                    print(f"    Batch {batch_idx+1}/{len(mc_batch_sizes)}: Generating {mc_batch_size} Y3 samples...")
                    
                    # Condition on observed Y1 and Y2 (normalized); sample Y3 only.
                    do_vars = {}
                    for i in range(10):
                        do_vars[i] = train_data_norm[:, i].cpu().numpy()
                    for i in range(10, 20):
                        do_vars[i] = train_data_norm[:, i].cpu().numpy()

                    sample_vars = list(range(20, 30))

                    # Returns shape: [mc_batch_size, n_samples_train, 10] in normalized space.
                    Y3_sampled_batch_norm = generate_conditional_samples(
                        models=models,
                        groups=groups,
                        A=A,
                        do_vars=do_vars,
                        sample_vars=sample_vars,
                        n_samples=mc_batch_size,
                        device=device
                    )
                    
                    Y3_sampled_batch_norm = Y3_sampled_batch_norm.cpu().numpy()

                    for i in range(mc_batch_size):
                        # Reconstruct the full 30-dim vector to apply inverse_transform.
                        full_data_norm = np.zeros((n_samples_train, 30))
                        full_data_norm[:, 0:10] = train_data_norm[:, 0:10].cpu().numpy()
                        full_data_norm[:, 10:20] = train_data_norm[:, 10:20].cpu().numpy()
                        full_data_norm[:, 20:30] = Y3_sampled_batch_norm[i]

                        if args.use_standard_scaler:
                            full_data = scaler.inverse_transform(full_data_norm)
                        else:
                            full_data = qt.inverse_transform(full_data_norm)

                        Y3_sampled = full_data[:, 20:30]
                        tau_d = mcodec(Y3_sampled, Y2_train, Y1_train)
                        tau_samples.append(tau_d)
                
                print(f"  Generated {len(tau_samples)} tau(d) values")
                print(f"  tau(d) range: [{min(tau_samples):.6f}, {max(tau_samples):.6f}]")
                print(f"  tau(d) mean: {np.mean(tau_samples):.6f}")

                num_larger_or_equal = sum(1 for tau_d in tau_samples if tau_d >= tau_observed)
                p_value = (1 + num_larger_or_equal) / (D + 1)

                print(f"\n  Results:")
                print(f"    tau (observed):              {tau_observed:.6f}")
                print(f"    # of tau(d) >= tau:          {num_larger_or_equal}")
                print(f"    p-value:                     {p_value:.6f}")

                result = {
                    'eta': eta,
                    'n_samples_train': n_samples_train,
                    'seed': seed,
                    'framework': 'conditional',
                    'tau_observed': tau_observed,
                    'p_value': p_value,
                    'num_larger_or_equal': num_larger_or_equal,
                    'D': D,
                    'use_standard_scaler': args.use_standard_scaler
                }
                
                for d, tau_d in enumerate(tau_samples):
                    result[f'tau_d_{d+1}'] = tau_d
                
                eta_results.append(result)

                print(f"\n  Saving results after seed {seed}...")
                eta_results_df = pd.DataFrame(eta_results)
                eta_results_df.to_csv(results_file, index=False)

                if len(eta_results_df) > 0:
                    print(f"    Results updated with {len(eta_results_df)} total rows")

                    current_data = eta_results_df[eta_results_df['n_samples_train'] == n_samples_train]
                    if len(current_data) > 0:
                        mean_p_value = current_data['p_value'].mean()
                        mean_tau = current_data['tau_observed'].mean()
                        rejection_rate = (current_data['p_value'] <= 0.05).mean()
                        print(f"    Current summary for eta={eta}, n_train={n_samples_train} (up to seed {seed}):")
                        print(f"      Mean p-value:   {mean_p_value:.6f}")
                        print(f"      Mean tau:       {mean_tau:.6f}")
                        print(f"      Rejection rate: {rejection_rate:.4f}")

        eta_results_df = pd.DataFrame(eta_results)
        eta_results_df.to_csv(results_file, index=False)

        print(f"\n{'='*70}")
        print(f"Creating summary statistics for eta={eta}...")

        summary_cols = ['tau_observed', 'p_value']
        summary = eta_results_df.groupby(['n_samples_train'])[summary_cols].agg(['mean', 'std', 'min', 'max', 'count']).round(6)

        summary.to_csv(f"./results/inference_power_eta/summary_eta{eta}{scaler_suffix}.csv")
        
        print(f"\nResults for eta={eta} saved to ./results/inference_power_eta/:")
        print(f"- Detailed results: inference_power_eta{eta}_results{scaler_suffix}.csv")
        print(f"- Summary: summary_eta{eta}{scaler_suffix}.csv")
        print(f"Total results for eta={eta}: {len(eta_results_df)} rows")
        
        # Print quick comparison by training sample size
        print(f"\n{'='*70}")
        print(f"SUMMARY FOR ETA={eta}")
        print(f"{'='*70}")
        print("\nP-value statistics:")
        print(eta_results_df.groupby('n_samples_train')['p_value'].agg(['count', 'mean', 'std', 'min', 'max']).round(6))
        print("\nTau (observed) statistics:")
        print(eta_results_df.groupby('n_samples_train')['tau_observed'].agg(['mean', 'std', 'min', 'max']).round(6))

        eta_results_df['reject_005'] = eta_results_df['p_value'] <= 0.05
        power_by_n = eta_results_df.groupby('n_samples_train')['reject_005'].mean()
        print(f"\nPower (rejection rate at alpha=0.05 under H1, eta={eta}):")
        print(power_by_n)

        results.extend(eta_results)

    if len(eta_values_list) > 1:
        print(f"\n{'='*70}")
        print("Creating combined results across all eta values...")
        combined_results_df = pd.DataFrame(results)
        combined_file = f"./results/inference_power_eta/inference_power_eta_combined_results{scaler_suffix}.csv"
        combined_results_df.to_csv(combined_file, index=False)

        combined_summary = combined_results_df.groupby(['eta', 'n_samples_train'])['p_value'].agg(['count', 'mean', 'std']).round(6)
        combined_summary.to_csv(f"./results/inference_power_eta/combined_summary{scaler_suffix}.csv")

        print(f"\nPower curves (rejection rate at alpha=0.05):")
        combined_results_df['reject_005'] = combined_results_df['p_value'] <= 0.05
        power_table = combined_results_df.groupby(['eta', 'n_samples_train'])['reject_005'].mean().unstack(fill_value=0)
        print(power_table)

        print(f"\nCombined results saved to: {combined_file}")

if __name__ == "__main__":
    main()
