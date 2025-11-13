#!/usr/bin/env python3
"""
Phase 2b: Raw p-value vs FDR Comparison
Aging Clock Project - Addon Script

This script addresses advisor feedback by comparing raw p-values vs FDR q-values
to demonstrate the impact of multiple testing correction on feature selection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


def load_correlation_results(result_dir="result"):
    """
    Load correlation results from Phase 2
    
    Args:
        result_dir (str): Directory where Phase 2 results are stored
    
    Returns:
        DataFrame: Correlation results with p-values and adjusted p-values
    """
    print("Loading correlation results from Phase 2...")
    
    corr_path = os.path.join(result_dir, "2_correlation_results.csv")
    corr_df = pd.read_csv(corr_path)
    
    print(f"Loaded results for {len(corr_df)} CpG sites")
    print(f"p-value range: {corr_df['p_value'].min():.2e} to {corr_df['p_value'].max():.2f}")
    print(f"FDR q-value range: {corr_df['adj_p_value'].min():.2e} to {corr_df['adj_p_value'].max():.2f}")
    
    return corr_df


def compare_thresholds(corr_df, alpha=0.05):
    """
    Compare raw p-value vs FDR q-value thresholds
    
    Args:
        corr_df (DataFrame): Correlation results
        alpha (float): Significance threshold
    
    Returns:
        dict: Comparison statistics
    """
    print(f"\nComparing thresholds at α = {alpha}...")
    
    # Apply raw p-value threshold
    raw_sig = corr_df['p_value'] < alpha
    n_raw_sig = raw_sig.sum()
    
    # Apply FDR q-value threshold
    fdr_sig = corr_df['adj_p_value'] < alpha
    n_fdr_sig = fdr_sig.sum()
    
    # Verify subset relationship
    fdr_are_raw_subset = all(corr_df.loc[fdr_sig, 'p_value'] < alpha)
    
    # Calculate overlap
    both_sig = raw_sig & fdr_sig
    n_both_sig = both_sig.sum()
    
    # Raw only (lost due to FDR correction)
    raw_only = raw_sig & ~fdr_sig
    n_raw_only = raw_only.sum()
    
    # Percentages
    pct_raw = (n_raw_sig / len(corr_df)) * 100
    pct_fdr = (n_fdr_sig / len(corr_df)) * 100
    pct_lost = (n_raw_only / n_raw_sig) * 100 if n_raw_sig > 0 else 0
    
    print(f"Raw p-value < {alpha}: {n_raw_sig} sites ({pct_raw:.1f}%)")
    print(f"FDR q-value < {alpha}: {n_fdr_sig} sites ({pct_fdr:.1f}%)")
    print(f"Sites lost due to FDR correction: {n_raw_only} ({pct_lost:.1f}% of raw significant)")
    print(f"All FDR-significant sites are raw-significant: {fdr_are_raw_subset}")
    
    return {
        'n_total': len(corr_df),
        'n_raw_sig': n_raw_sig,
        'n_fdr_sig': n_fdr_sig,
        'n_both_sig': n_both_sig,
        'n_raw_only': n_raw_only,
        'pct_raw': pct_raw,
        'pct_fdr': pct_fdr,
        'pct_lost': pct_lost,
        'is_subset': fdr_are_raw_subset
    }


def create_comparison_plot(corr_df, comparison_stats, figure_dir="figure"):
    """
    Create visualization comparing p-value vs FDR thresholds
    
    Args:
        corr_df (DataFrame): Correlation results
        comparison_stats (dict): Comparison statistics
        figure_dir (str): Directory to save the plot
    """
    print("\nCreating comparison visualization...")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: p-value distribution
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(corr_df['p_value'], bins=50, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
                label='p = 0.05 threshold')
    ax1.set_xlabel('Raw p-value', fontsize=12)
    ax1.set_ylabel('Number of CpG sites', fontsize=12)
    ax1.set_title('Distribution of Raw p-values', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: FDR q-value distribution
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(corr_df['adj_p_value'], bins=50, alpha=0.7, color='lightcoral', 
             edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
                label='FDR = 0.05 threshold')
    ax2.set_xlabel('FDR q-value', fontsize=12)
    ax2.set_ylabel('Number of CpG sites', fontsize=12)
    ax2.set_title('Distribution of FDR q-values', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Scatter plot of p-value vs FDR
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(corr_df['p_value'], corr_df['adj_p_value'], 
               alpha=0.6, s=20, c='gray', edgecolors='none')
    
    # Highlight significant points
    raw_sig = corr_df['p_value'] < 0.05
    fdr_sig = corr_df['adj_p_value'] < 0.05
    both_sig = raw_sig & fdr_sig
    
    ax3.scatter(corr_df.loc[both_sig, 'p_value'], 
               corr_df.loc[both_sig, 'adj_p_value'],
               alpha=0.7, s=30, c='green', label='Both significant', edgecolors='black')
    
    ax3.scatter(corr_df.loc[raw_sig & ~fdr_sig, 'p_value'], 
               corr_df.loc[raw_sig & ~fdr_sig, 'adj_p_value'],
               alpha=0.7, s=30, c='orange', label='Raw only (lost)', edgecolors='black')
    
    ax3.plot([0, 0.05], [0, 0.05], 'r--', linewidth=2, alpha=0.7, label='y=x line')
    ax3.axvline(x=0.05, color='red', linestyle=':', alpha=0.5)
    ax3.axhline(y=0.05, color='red', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Raw p-value', fontsize=12)
    ax3.set_ylabel('FDR q-value', fontsize=12)
    ax3.set_title('p-value vs FDR q-value', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Venn diagram-style visualization
    ax4 = plt.subplot(2, 2, 4)
    
    # Create a simple Venn-like representation
    total = comparison_stats['n_total']
    raw_sig = comparison_stats['n_raw_sig']
    fdr_sig = comparison_stats['n_fdr_sig']
    both_sig = comparison_stats['n_both_sig']
    raw_only = comparison_stats['n_raw_only']
    
    # Draw circles
    circle1 = plt.Circle((0.3, 0.5), 0.25, color='skyblue', alpha=0.5, 
                         label=f'Raw p<0.05\n({raw_sig} sites)')
    circle2 = plt.Circle((0.7, 0.5), 0.25, color='lightcoral', alpha=0.5,
                         label=f'FDR q<0.05\n({fdr_sig} sites)')
    
    ax4.add_patch(circle1)
    ax4.add_patch(circle2)
    
    # Add text labels
    ax4.text(0.3, 0.5, f'{raw_sig}', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    ax4.text(0.7, 0.5, f'{fdr_sig}', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.5, f'{both_sig}', ha='center', va='center', 
             fontsize=16, fontweight='bold', color='green')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Overlap of Significant Sites', fontsize=14)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(figure_dir, "2b_pvalue_fdr_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    return fig


def save_comparison_results(comparison_stats, corr_df, result_dir="result"):
    """
    Save comparison results to files
    
    Args:
        comparison_stats (dict): Comparison statistics
        corr_df (DataFrame): Correlation results
        result_dir (str): Directory to save results
    """
    print("\nSaving comparison results...")
    
    # Save summary statistics
    summary_df = pd.DataFrame([comparison_stats])
    summary_path = os.path.join(result_dir, "2b_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save sites that are raw-significant but not FDR-significant
    raw_sig = corr_df['p_value'] < 0.05
    fdr_sig = corr_df['adj_p_value'] < 0.05
    raw_only = raw_sig & ~fdr_sig
    
    lost_sites = corr_df[raw_only].copy()
    lost_sites = lost_sites.sort_values('p_value')
    
    lost_path = os.path.join(result_dir, "2b_sites_lost_to_fdr.csv")
    lost_sites.to_csv(lost_path, index=False)
    
    print(f"Summary saved: {summary_path}")
    print(f"Lost sites saved: {lost_path}")
    print(f"Sites lost to FDR correction: {len(lost_sites)}")
    
    # Print top lost sites
    if len(lost_sites) > 0:
        print("\nTop 10 sites lost due to FDR correction:")
        print(lost_sites[['CpG_site', 'correlation', 'p_value', 'adj_p_value']].head(10).to_string())


def main():
    """
    Main execution function
    """
    print("Starting Phase 2b: Raw p-value vs FDR Comparison\n")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load data
    corr_df = load_correlation_results()
    
    # Compare thresholds
    comparison_stats = compare_thresholds(corr_df)
    
    # Create visualization
    create_comparison_plot(corr_df, comparison_stats)
    
    # Save results
    save_comparison_results(comparison_stats, corr_df)
    
    print(f"\nPhase 2b completed successfully!")
    print(f"- Compared raw p-values vs FDR q-values")
    print(f"- Generated comparison visualization")
    print(f"- Identified sites lost due to FDR correction")
    print(f"- Results saved to 'result' directory with '2b_' prefix")


if __name__ == "__main__":
    main()
