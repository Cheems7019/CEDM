"""
plot_histograms.py

Visualize Energy test p-values from model_check.py results.

Creates histograms for:
1. PKC
2. PKA
3. (PKA, p44/42, pakts473)
4. (PIP3, PKA, pakts473)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Paths
    output_dir = Path(__file__).parent
    results_dir = output_dir / "results_check"
    pvalues_path = results_dir / "model_check_pvalues.csv"
    
    print("="*80)
    print("Plotting Energy Test P-value Histograms")
    print("="*80)
    
    # Load p-values
    print(f"\nLoading p-values from: {pvalues_path}")
    pvalues_df = pd.read_csv(pvalues_path)
    
    # Extract arrays (remove NaN values)
    p_values_pkc = pvalues_df['PKC_pvalue'].dropna().values
    p_values_pka = pvalues_df['PKA_pvalue'].dropna().values
    p_values_pka_p44_pakts = pvalues_df['PKA_p44_42_pakts473_pvalue'].dropna().values
    p_values_pip3_pka_pakts = pvalues_df['PIP3_PKA_pakts473_pvalue'].dropna().values
    
    M_pkc = len(p_values_pkc)
    M_pka = len(p_values_pka)
    M_triplet1 = len(p_values_pka_p44_pakts)
    M_triplet2 = len(p_values_pip3_pka_pakts)
    
    print(f"\nLoaded p-values:")
    print(f"  PKC: {M_pkc} samples")
    print(f"  PKA: {M_pka} samples")
    print(f"  (PKA, p44/42, pakts473): {M_triplet1} samples")
    print(f"  (PIP3, PKA, pakts473): {M_triplet2} samples")
    
    # ========================================
    # Create figure with 2x2 subplots
    # ========================================
    print("\nGenerating histograms...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Histogram 1: PKC
    axes[0].hist(p_values_pkc, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    axes[0].axvline(x=0.2, color='orange', linestyle='--', linewidth=2, label='α=0.2')
    axes[0].set_xlabel('P-value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'PKC Energy Test P-values (M={M_pkc})\nmean={p_values_pkc.mean():.3f}', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Histogram 2: PKA
    axes[1].hist(p_values_pka, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
    axes[1].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    axes[1].axvline(x=0.2, color='orange', linestyle='--', linewidth=2, label='α=0.2')
    axes[1].set_xlabel('P-value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'PKA Energy Test P-values (M={M_pka})\nmean={p_values_pka.mean():.3f}', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Histogram 3: (PKA, p44/42, pakts473)
    axes[2].hist(p_values_pka_p44_pakts, bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[2].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    axes[2].axvline(x=0.2, color='orange', linestyle='--', linewidth=2, label='α=0.2')
    axes[2].set_xlabel('P-value', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title(f'(PKA, p44/42, pakts473) Energy Test P-values (M={M_triplet1})\nmean={p_values_pka_p44_pakts.mean():.3f}', 
                      fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Histogram 4: (PIP3, PKA, pakts473)
    axes[3].hist(p_values_pip3_pka_pakts, bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    axes[3].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    axes[3].axvline(x=0.2, color='orange', linestyle='--', linewidth=2, label='α=0.2')
    axes[3].set_xlabel('P-value', fontsize=12)
    axes[3].set_ylabel('Frequency', fontsize=12)
    axes[3].set_title(f'(PIP3, PKA, pakts473) Energy Test P-values (M={M_triplet2})\nmean={p_values_pip3_pka_pakts.mean():.3f}', 
                      fontsize=14, fontweight='bold')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = results_dir / "model_check_histograms.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nHistograms saved to: {fig_path}")
    
    plt.show()
    
    # ========================================
    # Print summary statistics
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    tests = [
        ('PKC', p_values_pkc, M_pkc),
        ('PKA', p_values_pka, M_pka),
        ('(PKA, p44/42, pakts473)', p_values_pka_p44_pakts, M_triplet1),
        ('(PIP3, PKA, pakts473)', p_values_pip3_pka_pakts, M_triplet2)
    ]
    
    for test_name, p_vals, M in tests:
        print(f"\n{test_name}:")
        print(f"  Mean: {p_vals.mean():.4f}")
        print(f"  Std:  {p_vals.std():.4f}")
        print(f"  Min:  {p_vals.min():.4f}")
        print(f"  Max:  {p_vals.max():.4f}")
        print(f"  p > 0.05: {(p_vals > 0.05).sum()}/{M} ({100*(p_vals > 0.05).mean():.1f}%)")
        print(f"  p > 0.20: {(p_vals > 0.2).sum()}/{M} ({100*(p_vals > 0.2).mean():.1f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

