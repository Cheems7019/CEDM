"""
model_check.py

Validate diffusion models by computing Energy test p-values for generated samples.

Tests performed on simulated_data_check:
1. PKC: Compare each of M generated samples with original PKC data
2. PKA: Compare each of M generated samples with original PKA data
3. (PKA, p44/42, pakts473): Compare each of M generated triplets with original data
4. (PIP3, PKA, pakts473): Compare each of M generated triplets with original data

Outputs p-values to CSV for visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from hyppo.ksample import Energy
except ImportError:
    print("ERROR: hyppo package not found. Please install it: pip install hyppo")
    exit(1)


def main():
    # Paths
    output_dir = Path(__file__).parent
    simulated_check_dir = output_dir / "simulated_data_check"
    data_path = output_dir / "Flow_Cytometry.csv"
    
    print("="*80)
    print("Model Validation: Energy Test P-values")
    print("="*80)
    
    # Load original data
    print("\nLoading original data...")
    df_original = pd.read_csv(data_path)
    PKC_original = df_original['PKC'].values.reshape(-1, 1)
    PKA_original = df_original['PKA'].values.reshape(-1, 1)
    PIP3_original = df_original['PIP3'].values
    p44_42_original = df_original['p44/42'].values
    pakts473_original = df_original['pakts473'].values
    
    # Prepare multivariate originals
    PKA_p44_42_pakts473_original = np.column_stack([
        df_original['PKA'].values,
        df_original['p44/42'].values,
        df_original['pakts473'].values
    ])
    PIP3_PKA_pakts473_original = np.column_stack([
        df_original['PIP3'].values,
        df_original['PKA'].values,
        df_original['pakts473'].values
    ])
    
    n_samples = PKC_original.shape[0]
    print(f"Original data: n = {n_samples}")
    
    # Load generated data
    print("\nLoading generated data from simulated_data_check/...")
    PKC_gen = pd.read_csv(simulated_check_dir / "PKC.csv", header=None).values
    PKA_gen = pd.read_csv(simulated_check_dir / "PKA.csv", header=None).values
    PKA_p44_42_pakts473_gen = pd.read_csv(simulated_check_dir / "PKA_p44_42_pakts473.csv", header=None).values
    PIP3_PKA_pakts473_gen = pd.read_csv(simulated_check_dir / "PIP3_PKA_pakts473.csv", header=None).values
    
    M_pkc = PKC_gen.shape[1]
    M_pka = PKA_gen.shape[1]
    M_triplet1 = PKA_p44_42_pakts473_gen.shape[1] // 3
    M_triplet2 = PIP3_PKA_pakts473_gen.shape[1] // 3
    
    print(f"\nGenerated samples:")
    print(f"  PKC.csv shape: {PKC_gen.shape} -> M = {M_pkc}")
    print(f"  PKA.csv shape: {PKA_gen.shape} -> M = {M_pka}")
    print(f"  PKA_p44_42_pakts473.csv shape: {PKA_p44_42_pakts473_gen.shape} -> M = {M_triplet1} triplets")
    print(f"  PIP3_PKA_pakts473.csv shape: {PIP3_PKA_pakts473_gen.shape} -> M = {M_triplet2} triplets")
    
    # ========================================
    # (1) PKC Energy test p-values
    # ========================================
    print("\n" + "-"*60)
    print("(1) Computing Energy test p-values for PKC...")
    print("-"*60)
    
    p_values_pkc = []
    for j in tqdm(range(M_pkc), desc="PKC tests"):
        PKC_gen_j = PKC_gen[:, j].reshape(-1, 1)
        result = Energy().test(PKC_original, PKC_gen_j)
        p_values_pkc.append(result.pvalue)
    
    p_values_pkc = np.array(p_values_pkc)
    print(f"PKC p-values: mean={p_values_pkc.mean():.4f}, std={p_values_pkc.std():.4f}")
    print(f"PKC p-values > 0.05: {(p_values_pkc > 0.05).sum()}/{M_pkc} ({100*(p_values_pkc > 0.05).mean():.1f}%)")
    print(f"PKC p-values > 0.2: {(p_values_pkc > 0.2).sum()}/{M_pkc} ({100*(p_values_pkc > 0.2).mean():.1f}%)")
    
    # ========================================
    # (2) PKA Energy test p-values
    # ========================================
    print("\n" + "-"*60)
    print("(2) Computing Energy test p-values for PKA...")
    print("-"*60)
    
    p_values_pka = []
    for j in tqdm(range(M_pka), desc="PKA tests"):
        PKA_gen_j = PKA_gen[:, j].reshape(-1, 1)
        result = Energy().test(PKA_original, PKA_gen_j)
        p_values_pka.append(result.pvalue)
    
    p_values_pka = np.array(p_values_pka)
    print(f"PKA p-values: mean={p_values_pka.mean():.4f}, std={p_values_pka.std():.4f}")
    print(f"PKA p-values > 0.05: {(p_values_pka > 0.05).sum()}/{M_pka} ({100*(p_values_pka > 0.05).mean():.1f}%)")
    print(f"PKA p-values > 0.2: {(p_values_pka > 0.2).sum()}/{M_pka} ({100*(p_values_pka > 0.2).mean():.1f}%)")
    
    # ========================================
    # (3) (PKA, p44/42, pakts473) Energy test p-values
    # ========================================
    print("\n" + "-"*60)
    print("(3) Computing Energy test p-values for (PKA, p44/42, pakts473)...")
    print("-"*60)
    
    p_values_pka_p44_pakts = []
    for j in tqdm(range(M_triplet1), desc="(PKA, p44/42, pakts473) tests"):
        # Extract triplet j: columns [3*j, 3*j+1, 3*j+2]
        triplet_gen_j = PKA_p44_42_pakts473_gen[:, 3*j:3*j+3]
        result = Energy().test(PKA_p44_42_pakts473_original, triplet_gen_j)
        p_values_pka_p44_pakts.append(result.pvalue)
    
    p_values_pka_p44_pakts = np.array(p_values_pka_p44_pakts)
    print(f"(PKA, p44/42, pakts473) p-values: mean={p_values_pka_p44_pakts.mean():.4f}, std={p_values_pka_p44_pakts.std():.4f}")
    print(f"(PKA, p44/42, pakts473) p-values > 0.05: {(p_values_pka_p44_pakts > 0.05).sum()}/{M_triplet1} ({100*(p_values_pka_p44_pakts > 0.05).mean():.1f}%)")
    print(f"(PKA, p44/42, pakts473) p-values > 0.2: {(p_values_pka_p44_pakts > 0.2).sum()}/{M_triplet1} ({100*(p_values_pka_p44_pakts > 0.2).mean():.1f}%)")
    
    # ========================================
    # (4) (PIP3, PKA, pakts473) Energy test p-values
    # ========================================
    print("\n" + "-"*60)
    print("(4) Computing Energy test p-values for (PIP3, PKA, pakts473)...")
    print("-"*60)
    
    p_values_pip3_pka_pakts = []
    for j in tqdm(range(M_triplet2), desc="(PIP3, PKA, pakts473) tests"):
        # Extract triplet j: columns [3*j, 3*j+1, 3*j+2]
        triplet_gen_j = PIP3_PKA_pakts473_gen[:, 3*j:3*j+3]
        result = Energy().test(PIP3_PKA_pakts473_original, triplet_gen_j)
        p_values_pip3_pka_pakts.append(result.pvalue)
    
    p_values_pip3_pka_pakts = np.array(p_values_pip3_pka_pakts)
    print(f"(PIP3, PKA, pakts473) p-values: mean={p_values_pip3_pka_pakts.mean():.4f}, std={p_values_pip3_pka_pakts.std():.4f}")
    print(f"(PIP3, PKA, pakts473) p-values > 0.05: {(p_values_pip3_pka_pakts > 0.05).sum()}/{M_triplet2} ({100*(p_values_pip3_pka_pakts > 0.05).mean():.1f}%)")
    print(f"(PIP3, PKA, pakts473) p-values > 0.2: {(p_values_pip3_pka_pakts > 0.2).sum()}/{M_triplet2} ({100*(p_values_pip3_pka_pakts > 0.2).mean():.1f}%)")
    
    # ========================================
    # Save p-values to CSV
    # ========================================
    print("\n" + "-"*60)
    print("Saving p-values to CSV...")
    print("-"*60)
    
    # Create results directory
    results_dir = output_dir / "results_check"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine max length for dataframe (pad with NaN if lengths differ)
    max_len = max(len(p_values_pkc), len(p_values_pka), len(p_values_pka_p44_pakts), len(p_values_pip3_pka_pakts))
    
    def pad_array(arr, target_len):
        if len(arr) < target_len:
            return np.concatenate([arr, np.full(target_len - len(arr), np.nan)])
        return arr
    
    pvalues_df = pd.DataFrame({
        'PKC_pvalue': pad_array(p_values_pkc, max_len),
        'PKA_pvalue': pad_array(p_values_pka, max_len),
        'PKA_p44_42_pakts473_pvalue': pad_array(p_values_pka_p44_pakts, max_len),
        'PIP3_PKA_pakts473_pvalue': pad_array(p_values_pip3_pka_pakts, max_len)
    })
    
    pvalues_path = results_dir / "model_check_pvalues.csv"
    pvalues_df.to_csv(pvalues_path, index=False)
    print(f"P-values saved to: {pvalues_path}")
    
    print("\n" + "="*80)
    print("ENERGY TEST COMPLETE")
    print("="*80)
    print(f"\nTo visualize results, run: python plot_histograms.py")
    print("\nDone!")


if __name__ == "__main__":
    main()

