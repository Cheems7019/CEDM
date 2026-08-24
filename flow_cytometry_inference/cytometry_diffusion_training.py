"""
cytometry_diffusion_training.py

Train conditional diffusion models on Flow Cytometry data for 3 independent clusters.

Clusters (no inter-cluster connections):
- Cluster 0: PKC (single feature, unconditional)
- Cluster 1: PIP3 (G0) → PKA, p44/42 (G1) → pakts473 (G2)
- Cluster 2: PIP3, PKA (G0) → p44/42 (G1), pakts473 (G2)

Process:
1. Train each cluster sequentially (0 → 1 → 2) using conditional diffusion framework
2. For each cluster, try seeds 0-10
3. Validate with Energy two-sample test (α=0.2)
4. If good model found, save checkpoint, delete failed ones, move to next cluster
5. If all seeds fail for any cluster, stop and notify

Energy tests per cluster:
- Cluster 0: (PKC)
- Cluster 1: (PKA, p44/42, pakts473)
- Cluster 2: (PIP3, PKA, pakts473)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer
from pathlib import Path
import shutil
import pickle
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from utils.conditional_ddpm import (
    ConditionalMLPDiffusion,
    ConditionalDDPM,
    train_all_conditional_ddpms,
    generate_conditional_samples
)

try:
    from hyppo.ksample import Energy
except ImportError:
    print("ERROR: hyppo package not found. Please install it: pip install hyppo")
    sys.exit(1)


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


def run_energy_test(X, Y, alpha=0.2):
    """
    Run Energy two-sample test.
    
    Returns:
        passed: bool - True if p-value > alpha (fail to reject null = same distribution)
        p_value: float - the p-value from the test
    """
    result = Energy().test(X, Y)
    p_value = result.pvalue
    passed = p_value > alpha
    return passed, p_value


def train_cluster(cluster_name, data, groups, A, test_columns, 
                  device, checkpoint_dir, alpha=0.2, max_seeds=11):
    """
    Train conditional diffusion models for a single cluster.
    
    Args:
        cluster_name: name of the cluster
        data: numpy array of data for this cluster
        groups: torch tensor of group assignments
        A: numpy array adjacency matrix
        test_columns: dict mapping test name to column indices for Energy test
        device: torch device
        checkpoint_dir: directory to save checkpoints
        alpha: significance level for Energy test
        max_seeds: maximum seeds to try
    
    Returns:
        success: bool - True if good model found
        best_seed: int or None - seed of best model
    """
    n_samples, d_in = data.shape
    n_groups = A.shape[0]
    
    hidden_dims = [640, 320, 320, 320, 160]
    dim_t = 64
    n_steps = 1000
    n_epochs = 3000
    lr = 5e-5
    
    print(f"\n{'='*80}")
    print(f"TRAINING {cluster_name}")
    print(f"{'='*80}")
    print(f"Features: {d_in}, Groups: {n_groups}, Samples: {n_samples}")
    print(f"Hyperparameters: hidden_dims={hidden_dims}, dim_t={dim_t}")
    print(f"                 n_epochs={n_epochs}, lr={lr}")
    print(f"Energy test α = {alpha}")
    
    failed_checkpoints = []

    for seed in range(max_seeds):
        print(f"\n{'-'*60}")
        print(f"{cluster_name} - Seed {seed}")
        print(f"{'-'*60}")

        seed_everything(seed)

        scaler = QuantileTransformer(output_distribution='normal', random_state=seed)
        data_scaled = scaler.fit_transform(data)
        X = torch.tensor(data_scaled, dtype=torch.float32)

        print(f"Training conditional diffusion models...")
        models = train_all_conditional_ddpms(
            train_data=X,
            groups=groups,
            A=A,
            hidden_dims=hidden_dims,
            dim_t=dim_t,
            n_steps=n_steps,
            n_epochs=n_epochs,
            lr=lr,
            device=device
        )
        
        checkpoint_path = checkpoint_dir / f"{cluster_name}_seed{seed}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(models, f)

        print(f"Models saved to {checkpoint_path}")

        print(f"\nValidating with Energy tests...")
        all_passed = True
        
        for sample_idx in range(2):
            print(f"  Sample {sample_idx + 1}/2:")
            
            generated_norm = generate_conditional_samples(
                models=models,
                groups=groups,
                A=A,
                do_vars=None,
                sample_vars=None,
                n_samples=n_samples,
                device=device
            )
            
            # Shape: [n_samples, 1, d_in] -> [n_samples, d_in]
            generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()
            generated = scaler.inverse_transform(generated_norm)
            
            sample_passed = True
            for test_name, cols in test_columns.items():
                X_orig = data[:, cols] if len(cols) > 1 else data[:, cols].reshape(-1, 1)
                Y_gen = generated[:, cols] if len(cols) > 1 else generated[:, cols].reshape(-1, 1)
                
                passed, p_value = run_energy_test(X_orig, Y_gen, alpha)
                status = "PASS" if passed else "FAIL"
                print(f"    {test_name}: p-value = {p_value:.4f} [{status}]")
                
                if not passed:
                    sample_passed = False
                    all_passed = False
                    break  # Early stop for this sample
            
            if not sample_passed:
                print(f"  Sample {sample_idx + 1} FAILED - stopping validation")
                break  # Early stop, don't test sample 2
        
        if all_passed:
            print(f"\n{'='*60}")
            print(f"SUCCESS! {cluster_name} found good model at seed {seed}")
            print(f"{'='*60}")
            
            best_path = checkpoint_dir / f"{cluster_name}_best.pkl"
            shutil.copy(checkpoint_path, best_path)

            metadata_path = checkpoint_dir / f"{cluster_name}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'seed': seed,
                    'scaler': scaler,
                    'groups': groups,
                    'A': A,
                    'hidden_dims': hidden_dims,
                    'dim_t': dim_t,
                    'n_steps': n_steps
                }, f)
            
            if checkpoint_path.exists():
                os.remove(checkpoint_path)

            for failed_path in failed_checkpoints:
                if failed_path.exists():
                    os.remove(failed_path)
                    print(f"Deleted failed checkpoint: {failed_path.name}")
            
            print(f"Saved: {best_path.name}, {metadata_path.name}")
            return True, seed
        else:
            failed_checkpoints.append(checkpoint_path)
            print(f"Seed {seed} failed, trying next...")

    print(f"\n{'='*60}")
    print(f"FAILED! {cluster_name} - no good model found after {max_seeds} seeds")
    print(f"{'='*60}")
    
    for failed_path in failed_checkpoints:
        if failed_path.exists():
            os.remove(failed_path)
    
    return False, None


def main():
    parser = argparse.ArgumentParser(description="Train conditional diffusion models on Flow Cytometry data")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use (default: cuda:2)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    max_seeds = 11
    alpha = 0.2

    output_dir = Path(__file__).parent
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Flow Cytometry Conditional Diffusion Model Training (3 Clusters)")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Maximum seeds per cluster: {max_seeds}")
    print(f"Significance level (α): {alpha}")
    
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)

    data_path = output_dir / "Flow_Cytometry.csv"
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Original columns:
    # praf(0), pmek(1), plcg(2), PIP2(3), PIP3(4), p44/42(5), pakts473(6), PKA(7), PKC(8), P38(9), pjnk(10)
    
    # ========================================
    # CLUSTER 0: PKC (single feature, unconditional)
    # ========================================
    cluster0_columns = ['PKC']
    cluster0_data = df[cluster0_columns].values.astype(np.float32)

    cluster0_groups = torch.zeros(1, dtype=torch.long)
    cluster0_A = np.array([[0]], dtype=np.float32)
    cluster0_tests = {'(PKC)': [0]}
    
    print(f"\nCluster 0: {cluster0_columns}")
    print(f"  Groups: G0=[PKC]")
    print(f"  Connections: unconditional")
    
    # ========================================
    # CLUSTER 1: PIP3 (G0) → PKA, p44/42 (G1) → pakts473 (G2)
    # ========================================
    cluster1_columns = ['PIP3', 'PKA', 'p44/42', 'pakts473']
    cluster1_data = df[cluster1_columns].values.astype(np.float32)

    # G0=PIP3, G1=PKA+p44/42, G2=pakts473
    cluster1_groups = torch.zeros(4, dtype=torch.long)
    cluster1_groups[0] = 0  # PIP3
    cluster1_groups[1] = 1  # PKA
    cluster1_groups[2] = 1  # p44/42
    cluster1_groups[3] = 2  # pakts473

    cluster1_A = np.array([
        [0, 1, 0],  # G0 → G1
        [0, 0, 1],  # G1 → G2
        [0, 0, 0]
    ], dtype=np.float32)

    cluster1_tests = {'(PKA, p44/42, pakts473)': [1, 2, 3]}
    
    print(f"\nCluster 1: {cluster1_columns}")
    print(f"  Groups: G0=[PIP3], G1=[PKA, p44/42], G2=[pakts473]")
    print(f"  Connections: G0→G1, G1→G2")
    print(f"  Test: (PKA, p44/42, pakts473)")
    
    # ========================================
    # CLUSTER 2: PIP3, PKA (G0) → p44/42 (G1), pakts473 (G2)
    # ========================================
    cluster2_columns = ['PIP3', 'PKA', 'p44/42', 'pakts473']
    cluster2_data = df[cluster2_columns].values.astype(np.float32)

    # G0=PIP3+PKA, G1=p44/42, G2=pakts473
    cluster2_groups = torch.zeros(4, dtype=torch.long)
    cluster2_groups[0] = 0  # PIP3
    cluster2_groups[1] = 0  # PKA
    cluster2_groups[2] = 1  # p44/42
    cluster2_groups[3] = 2  # pakts473

    cluster2_A = np.array([
        [0, 1, 1],  # G0 → G1, G0 → G2
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.float32)

    cluster2_tests = {'(PIP3, PKA, pakts473)': [0, 1, 3]}
    
    print(f"\nCluster 2: {cluster2_columns}")
    print(f"  Groups: G0=[PIP3, PKA], G1=[p44/42], G2=[pakts473]")
    print(f"  Connections: G0→G1, G0→G2")
    print(f"  Test: (PIP3, PKA, pakts473)")
    
    clusters = [
        ('cluster0', cluster0_data, cluster0_groups, cluster0_A, cluster0_tests),
        ('cluster1', cluster1_data, cluster1_groups, cluster1_A, cluster1_tests),
        ('cluster2', cluster2_data, cluster2_groups, cluster2_A, cluster2_tests),
    ]
    
    results = {}

    for cluster_name, data, groups, A, tests in clusters:
        success, best_seed = train_cluster(
            cluster_name, data, groups, A, tests,
            device, checkpoint_dir, alpha, max_seeds
        )
        
        results[cluster_name] = {'success': success, 'seed': best_seed}
        
        if not success:
            print(f"\n{'='*80}")
            print(f"STOPPING: {cluster_name} failed to train a good model")
            print(f"{'='*80}")
            print("\nPlease consider:")
            print("  1. Increasing n_epochs")
            print("  2. Adjusting learning rate")
            print("  3. Increasing model capacity (hidden_dims, dim_t)")
            print("  4. Relaxing α (currently 0.2)")
            print("  5. Checking data quality for this cluster")
            break
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    all_success = True
    for cluster_name, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        seed_info = f"seed={result['seed']}" if result['seed'] is not None else "no good seed"
        print(f"  {cluster_name}: {status} ({seed_info})")
        if not result['success']:
            all_success = False
    
    if all_success and len(results) == 3:
        print("\nAll 3 clusters trained successfully!")
        print(f"Checkpoints saved in: {checkpoint_dir}")
        
        # ========================================
        # GENERATE AND SAVE SAMPLES
        # ========================================
        print("\n" + "="*80)
        print("GENERATING SAMPLES")
        print("="*80)
        
        M = 500
        batch_size = 250
        n_samples = cluster0_data.shape[0]
        
        simulated_check_dir = output_dir / "simulated_data_check"
        simulated_edge_dir = output_dir / "simulated_data_edge"
        simulated_check_dir.mkdir(parents=True, exist_ok=True)
        simulated_edge_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directories:")
        print(f"  - simulated_data_check: {simulated_check_dir}")
        print(f"  - simulated_data_edge: {simulated_edge_dir}")
        print(f"M = {M}, batch_size = {batch_size}, n = {n_samples}")
        
        n_batches = M // batch_size + (1 if M % batch_size != 0 else 0)
        batch_sizes_list = [batch_size] * (M // batch_size) + ([M % batch_size] if M % batch_size != 0 else [])
        
        # ------------------------------------------
        # CLUSTER 0: PKC unconditional sampling
        # ------------------------------------------
        print("\n" + "-"*60)
        print("CLUSTER 0: Generating PKC unconditional samples")
        print("-"*60)
        
        cluster0_model_path = checkpoint_dir / "cluster0_best.pkl"
        cluster0_meta_path = checkpoint_dir / "cluster0_metadata.pkl"
        
        with open(cluster0_model_path, 'rb') as f:
            models_c0 = pickle.load(f)
        with open(cluster0_meta_path, 'rb') as f:
            cluster0_meta = pickle.load(f)
        
        scaler_c0 = cluster0_meta['scaler']
        
        all_pkc = []

        for batch_idx, curr_batch_size in enumerate(batch_sizes_list):
            print(f"  Batch {batch_idx+1}/{len(batch_sizes_list)}: generating {curr_batch_size} × {n_samples} samples...")

            # Shape: [n_samples * curr_batch_size, 1, 1]
            generated_norm = generate_conditional_samples(
                models=models_c0,
                groups=cluster0_meta['groups'],
                A=cluster0_meta['A'],
                do_vars=None,
                sample_vars=None,
                n_samples=n_samples * curr_batch_size,
                device=device
            )
            
            # Shape: [n_samples * curr_batch_size, 1]
            generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()
            generated = scaler_c0.inverse_transform(generated_norm)
            
            # Reshape to (curr_batch_size, n_samples) then transpose to (n_samples, curr_batch_size)
            pkc_batch = generated.reshape(curr_batch_size, n_samples).T
            all_pkc.append(pkc_batch)
        
        pkc_unconditional = np.hstack(all_pkc)  # (n_samples, M)

        pd.DataFrame(pkc_unconditional).to_csv(simulated_check_dir / "PKC.csv", index=False, header=False)
        print(f"  Saved: simulated_data_check/PKC.csv, shape={pkc_unconditional.shape}")

        pd.DataFrame(pkc_unconditional).to_csv(simulated_edge_dir / "PIP2_plcg_to_PKC.csv", index=False, header=False)
        print(f"  Saved: simulated_data_edge/PIP2_plcg_to_PKC.csv, shape={pkc_unconditional.shape}")
        
        # ------------------------------------------
        # CLUSTER 1: Unconditional sampling (PKA, p44/42, pakts473)
        # ------------------------------------------
        print("\n" + "-"*60)
        print("CLUSTER 1: Generating unconditional samples (PKA, p44/42, pakts473)")
        print("-"*60)
        
        cluster1_model_path = checkpoint_dir / "cluster1_best.pkl"
        cluster1_meta_path = checkpoint_dir / "cluster1_metadata.pkl"
        
        with open(cluster1_model_path, 'rb') as f:
            models_c1 = pickle.load(f)
        with open(cluster1_meta_path, 'rb') as f:
            cluster1_meta = pickle.load(f)
        
        scaler_c1 = cluster1_meta['scaler']
        
        # Each generation produces paired triplets (PKA, p44/42, pakts473).
        all_triplets = []

        for batch_idx, curr_batch_size in enumerate(batch_sizes_list):
            print(f"  Batch {batch_idx+1}/{len(batch_sizes_list)}: generating {curr_batch_size} × {n_samples} samples...")

            # Shape: [n_samples * curr_batch_size, 1, 4]
            generated_norm = generate_conditional_samples(
                models=models_c1,
                groups=cluster1_meta['groups'],
                A=cluster1_meta['A'],
                do_vars=None,
                sample_vars=None,
                n_samples=n_samples * curr_batch_size,
                device=device
            )
            
            # Shape: [n_samples * curr_batch_size, 4]
            generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()
            generated = scaler_c1.inverse_transform(generated_norm)
            
            # Reshape to (curr_batch_size, n_samples, 4)
            generated_reshaped = generated.reshape(curr_batch_size, n_samples, 4)
            
            # Columns: PIP3=0, PKA=1, p44/42=2, pakts473=3; shape per feature: (n_samples, curr_batch_size)
            pka_batch = generated_reshaped[:, :, 1].T
            p44_42_batch = generated_reshaped[:, :, 2].T
            pakts473_batch = generated_reshaped[:, :, 3].T

            for i in range(curr_batch_size):
                all_triplets.extend([pka_batch[:, i], p44_42_batch[:, i], pakts473_batch[:, i]])

        pka_p44_42_pakts473 = np.column_stack(all_triplets)  # (n_samples, 3*M)
        
        pd.DataFrame(pka_p44_42_pakts473).to_csv(simulated_check_dir / "PKA_p44_42_pakts473.csv", index=False, header=False)
        print(f"  Saved: simulated_data_check/PKA_p44_42_pakts473.csv, shape={pka_p44_42_pakts473.shape}")
        
        # ------------------------------------------
        # CLUSTER 1: Conditional sampling - pakts473 | (PIP3, PKA, p44/42)
        # ------------------------------------------
        print("\n" + "-"*60)
        print("CLUSTER 1: Generating conditional pakts473 | (PIP3, PKA, p44/42)")
        print("-"*60)
        
        all_pakts473_cond = []

        cluster1_data_norm = scaler_c1.transform(cluster1_data)

        for batch_idx, curr_batch_size in enumerate(batch_sizes_list):
            print(f"  Batch {batch_idx+1}/{len(batch_sizes_list)}: generating {curr_batch_size} conditional samples...")

            # Observe PIP3 (col 0), PKA (col 1), p44/42 (col 2); sample pakts473 (col 3).
            do_vars = {
                0: cluster1_data_norm[:, 0],  # PIP3
                1: cluster1_data_norm[:, 1],  # PKA
                2: cluster1_data_norm[:, 2],  # p44/42
            }

            sample_vars = [3]  # pakts473

            # Shape: [curr_batch_size, n_samples, 1]
            output_norm = generate_conditional_samples(
                models=models_c1,
                groups=cluster1_meta['groups'],
                A=cluster1_meta['A'],
                do_vars=do_vars,
                sample_vars=sample_vars,
                n_samples=curr_batch_size,
                device=device
            )
            
            output_norm = output_norm.cpu().detach().numpy()  # (curr_batch_size, n_samples, 1)

            for i in range(curr_batch_size):
                pakts473_norm = output_norm[i, :, 0]

                # Reconstruct the full 4-dim vector to use the scaler's inverse_transform.
                full_vector_norm = np.zeros((n_samples, 4))
                full_vector_norm[:, 0:3] = cluster1_data_norm[:, 0:3]
                full_vector_norm[:, 3] = pakts473_norm

                full_vector = scaler_c1.inverse_transform(full_vector_norm)
                pakts473_sampled = full_vector[:, 3]

                all_pakts473_cond.append(pakts473_sampled)
        
        pakts473_conditional = np.column_stack(all_pakts473_cond)  # (n_samples, M)
        pd.DataFrame(pakts473_conditional).to_csv(simulated_edge_dir / "PIP3_to_pakts473.csv", index=False, header=False)
        print(f"  Saved: simulated_data_edge/PIP3_to_pakts473.csv, shape={pakts473_conditional.shape}")
        
        # ------------------------------------------
        # CLUSTER 2: Unconditional sampling (PIP3, PKA, pakts473)
        # ------------------------------------------
        print("\n" + "-"*60)
        print("CLUSTER 2: Generating unconditional samples (PIP3, PKA, pakts473)")
        print("-"*60)
        
        cluster2_model_path = checkpoint_dir / "cluster2_best.pkl"
        cluster2_meta_path = checkpoint_dir / "cluster2_metadata.pkl"
        
        with open(cluster2_model_path, 'rb') as f:
            models_c2 = pickle.load(f)
        with open(cluster2_meta_path, 'rb') as f:
            cluster2_meta = pickle.load(f)
        
        scaler_c2 = cluster2_meta['scaler']
        
        # Each generation produces paired triplets (PIP3, PKA, pakts473).
        all_triplets_c2 = []
        all_pka_c2 = []

        for batch_idx, curr_batch_size in enumerate(batch_sizes_list):
            print(f"  Batch {batch_idx+1}/{len(batch_sizes_list)}: generating {curr_batch_size} × {n_samples} samples...")

            # Shape: [n_samples * curr_batch_size, 1, 4]
            generated_norm = generate_conditional_samples(
                models=models_c2,
                groups=cluster2_meta['groups'],
                A=cluster2_meta['A'],
                do_vars=None,
                sample_vars=None,
                n_samples=n_samples * curr_batch_size,
                device=device
            )
            
            # Shape: [n_samples * curr_batch_size, 4]
            generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()
            generated = scaler_c2.inverse_transform(generated_norm)
            
            # Reshape to (curr_batch_size, n_samples, 4)
            generated_reshaped = generated.reshape(curr_batch_size, n_samples, 4)
            
            # Columns: PIP3=0, PKA=1, p44/42=2, pakts473=3; shape per feature: (n_samples, curr_batch_size)
            pip3_batch = generated_reshaped[:, :, 0].T
            pka_batch = generated_reshaped[:, :, 1].T
            pakts473_batch = generated_reshaped[:, :, 3].T

            for i in range(curr_batch_size):
                all_triplets_c2.extend([pip3_batch[:, i], pka_batch[:, i], pakts473_batch[:, i]])
                all_pka_c2.append(pka_batch[:, i])

        pip3_pka_pakts473 = np.column_stack(all_triplets_c2)  # (n_samples, 3*M)
        pka_only = np.column_stack(all_pka_c2)
        
        pd.DataFrame(pip3_pka_pakts473).to_csv(simulated_check_dir / "PIP3_PKA_pakts473.csv", index=False, header=False)
        print(f"  Saved: simulated_data_check/PIP3_PKA_pakts473.csv, shape={pip3_pka_pakts473.shape}")
        
        pd.DataFrame(pka_only).to_csv(simulated_check_dir / "PKA.csv", index=False, header=False)
        print(f"  Saved: simulated_data_check/PKA.csv, shape={pka_only.shape}")
        
        pd.DataFrame(pka_only).to_csv(simulated_edge_dir / "PKC_to_PKA.csv", index=False, header=False)
        print(f"  Saved: simulated_data_edge/PKC_to_PKA.csv, shape={pka_only.shape}")
        
        # ------------------------------------------
        # CLUSTER 2: Conditional sampling - pakts473 | (PIP3, PKA, p44/42)
        # ------------------------------------------
        print("\n" + "-"*60)
        print("CLUSTER 2: Generating conditional pakts473 | (PIP3, PKA, p44/42)")
        print("-"*60)
        
        all_pakts473_cond_c2 = []

        cluster2_data_norm = scaler_c2.transform(cluster2_data)

        for batch_idx, curr_batch_size in enumerate(batch_sizes_list):
            print(f"  Batch {batch_idx+1}/{len(batch_sizes_list)}: generating {curr_batch_size} conditional samples...")

            # Observe PIP3 (col 0), PKA (col 1), p44/42 (col 2); sample pakts473 (col 3).
            do_vars = {
                0: cluster2_data_norm[:, 0],  # PIP3
                1: cluster2_data_norm[:, 1],  # PKA
                2: cluster2_data_norm[:, 2],  # p44/42
            }

            sample_vars = [3]  # pakts473

            # Shape: [curr_batch_size, n_samples, 1]
            output_norm = generate_conditional_samples(
                models=models_c2,
                groups=cluster2_meta['groups'],
                A=cluster2_meta['A'],
                do_vars=do_vars,
                sample_vars=sample_vars,
                n_samples=curr_batch_size,
                device=device
            )
            
            output_norm = output_norm.cpu().detach().numpy()  # (curr_batch_size, n_samples, 1)

            for i in range(curr_batch_size):
                pakts473_norm = output_norm[i, :, 0]

                # Reconstruct the full 4-dim vector to use the scaler's inverse_transform.
                full_vector_norm = np.zeros((n_samples, 4))
                full_vector_norm[:, 0:3] = cluster2_data_norm[:, 0:3]
                full_vector_norm[:, 3] = pakts473_norm

                full_vector = scaler_c2.inverse_transform(full_vector_norm)
                pakts473_sampled = full_vector[:, 3]

                all_pakts473_cond_c2.append(pakts473_sampled)
        
        pakts473_conditional_c2 = np.column_stack(all_pakts473_cond_c2)  # (n_samples, M)
        pd.DataFrame(pakts473_conditional_c2).to_csv(simulated_edge_dir / "P44_42_to_pakts473.csv", index=False, header=False)
        print(f"  Saved: simulated_data_edge/P44_42_to_pakts473.csv, shape={pakts473_conditional_c2.shape}")
        
        print("\n" + "="*80)
        print("SAMPLE GENERATION COMPLETE")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"\nsimulated_data_check/ (4 files):")
        print(f"  - PKC.csv: {pkc_unconditional.shape}")
        print(f"  - PKA.csv: {pka_only.shape}")
        print(f"  - PKA_p44_42_pakts473.csv: {pka_p44_42_pakts473.shape}")
        print(f"  - PIP3_PKA_pakts473.csv: {pip3_pka_pakts473.shape}")
        print(f"\nsimulated_data_edge/ (4 files):")
        print(f"  - PIP2_plcg_to_PKC.csv: {pkc_unconditional.shape}")
        print(f"  - PIP3_to_pakts473.csv: {pakts473_conditional.shape}")
        print(f"  - PKC_to_PKA.csv: {pka_only.shape}")
        print(f"  - P44_42_to_pakts473.csv: {pakts473_conditional_c2.shape}")
        
    else:
        print("\nTraining incomplete - see above for details")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

