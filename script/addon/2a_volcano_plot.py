#!/usr/bin/env python3
"""
Phase 2a: Volcano Plot for Feature Selection Results
Aging Clock Project - Addon Script

This script creates a volcano plot to visualize the correlation analysis results
from Phase 2, highlighting the selected age-associated CpG sites.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    print(f"Loaded correlation results for {len(corr_df)} CpG sites")
    
    return corr_df


def create_volcano_plot(corr_df, figure_dir="figure"):
    """
    Create a volcano plot of correlation results with log2FoldChange on x-axis
    
    Args:
        corr_df (DataFrame): Correlation results
        figure_dir (str): Directory to save the plot
    """
    print("Creating volcano plot...")
    
    # Calculate -log10(adjusted p-value) for y-axis
    corr_df['neg_log10_adj_pval'] = -np.log10(corr_df['adj_p_value'])
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color points based on significance and effect size
    colors = []
    sizes = []
    for _, row in corr_df.iterrows():
        if row['significant'] and abs(row['correlation']) > 0.2:
            colors.append('red')  # Selected features
            sizes.append(40)
        elif row['significant']:
            colors.append('orange')  # Significant but small effect
            sizes.append(20)
        else:
            colors.append('lightgray')  # Not significant
            sizes.append(10)
    
    # Create scatter plot with Pearson correlation coefficient on x-axis
    scatter = ax.scatter(corr_df['correlation'], corr_df['neg_log10_adj_pval'], 
                        c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add threshold lines
    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', alpha=0.7, 
               label='FDR = 0.05')
    ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.7, 
               label='|r| = 0.2')
    ax.axvline(x=-0.2, color='green', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_ylabel('-log₁₀(FDR-adjusted p-value)', fontsize=12)
    ax.set_title('Volcano Plot: Age-Associated CpG Sites\n' + 
                f'({len(corr_df[(corr_df["significant"]) & (abs(corr_df["correlation"]) > 0.2)])} sites selected from {len(corr_df)} total)', 
                fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with summary statistics
    selected_features = len(corr_df[(corr_df['significant']) & (abs(corr_df['correlation']) > 0.2)])
    summary_text = f"""Total sites: {len(corr_df)}
Significant (FDR<0.05): {corr_df['significant'].sum()}
Selected (|r|>0.2): {selected_features}
Positive correlation: {(corr_df[(corr_df['significant']) & (abs(corr_df['correlation']) > 0.2)]['correlation'] > 0).sum()}
Negative correlation: {(corr_df[(corr_df['significant']) & (abs(corr_df['correlation']) > 0.2)]['correlation'] < 0).sum()}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "2a_volcano_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Volcano plot saved to: {output_path}")
    
    # Print summary statistics
    selected_count = len(corr_df[(corr_df['significant']) & 
                                 (abs(corr_df['correlation']) > 0.2)])
    significant_count = corr_df['significant'].sum()
    total_count = len(corr_df)
    
    print(f"Summary:")
    print(f"  - Total CpG sites analyzed: {total_count}")
    print(f"  - Significant sites (FDR < 0.05): {significant_count}")
    print(f"  - Selected sites (FDR < 0.05 & |r| > 0.2): {selected_count}")
    
    return fig





def main():
    """
    Main execution function
    """
    print("Starting Phase 2a: Volcano Plot for Feature Selection Results\n")
    
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Load correlation results
    corr_df = load_correlation_results()
    
    # Create volcano plot
    create_volcano_plot(corr_df)
    
    print(f"\nPhase 2a completed successfully!")
    print(f"- Generated volcano plots showing correlation results")
    print(f"- Highlighted {len(corr_df[(corr_df['significant']) & (abs(corr_df['correlation']) > 0.2)])} selected features")
    print(f"- Plots saved to 'figure' directory with '2a_' prefix")


if __name__ == "__main__":
    main()
