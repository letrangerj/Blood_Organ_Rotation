#!/usr/bin/env python3
"""
Phase 2: Feature Selection via Correlation Analysis
Aging Clock Project

This script performs:
1. Calculate Pearson correlation coefficients between methylation and age
2. Apply multiple testing correction
3. Select age-associated CpG sites based on significance and effect size
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os


def load_data(result_dir="result"):
    """
    Load the preprocessed data from Phase 1
    
    Args:
        result_dir (str): Directory where Phase 1 results are stored
    
    Returns:
        DataFrame: Merged dataset with methylation and age data
    """
    print("Loading preprocessed data from Phase 1...")
    data_path = os.path.join(result_dir, "1_preprocessed_data.csv")
    data = pd.read_csv(data_path, index_col=0)
    print(f"Loaded data with shape: {data.shape}")
    
    # Check that Age column exists
    if 'Age' not in data.columns:
        raise ValueError("Age column not found in the data")
    
    return data


def calculate_correlations(data):
    """
    Calculate Pearson correlation coefficients between methylation levels and age
    
    Args:
        data: DataFrame with samples as rows, CpG sites and Age as columns
    
    Returns:
        DataFrame: Results with CpG site IDs, correlation coefficients, and p-values
    """
    print("Calculating Pearson correlation coefficients...")
    
    # Identify methylation columns (exclude Age and other non-methylation columns if needed)
    methylation_cols = [col for col in data.columns if col != 'Age' and col not in ['Gender', 'Annotation']]
    
    print(f"Analyzing {len(methylation_cols)} CpG sites for age correlation...")
    
    # Calculate correlations
    correlations = []
    for cpg_site in methylation_cols:
        age_values = data['Age'].dropna()
        methylation_values = data[cpg_site].dropna()
        
        # Ensure both series have the same indices after dropping NaN
        common_indices = age_values.index.intersection(methylation_values.index)
        if len(common_indices) < 10:  # Minimum sample requirement
            continue
        
        age_aligned = age_values.loc[common_indices]
        meth_aligned = methylation_values.loc[common_indices]
        
        # Calculate Pearson correlation
        corr, p_value = stats.pearsonr(age_aligned, meth_aligned)
        
        correlations.append({
            'CpG_site': cpg_site,
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(common_indices)
        })
    
    corr_df = pd.DataFrame(correlations)
    print(f"Calculated correlations for {len(corr_df)} CpG sites")
    
    return corr_df


def apply_multiple_testing_correction(corr_df, method='fdr_bh'):
    """
    Apply multiple testing correction to p-values
    
    Args:
        corr_df: DataFrame with correlation results
        method: Method for multiple testing correction ('fdr_bh', 'bonferroni', etc.)
    
    Returns:
        DataFrame: Results with corrected p-values
    """
    print(f"Applying {method} multiple testing correction...")
    
    # Perform multiple testing correction
    corrected_pvals = multipletests(corr_df['p_value'], method=method)
    corr_df['adj_p_value'] = corrected_pvals[1]
    corr_df['significant'] = corrected_pvals[0]
    
    num_significant = corrected_pvals[0].sum()
    print(f"Number of significant sites after correction: {num_significant}")
    
    return corr_df


def select_features(corr_df, pval_threshold=0.05, effect_threshold=0.2):
    """
    Select age-associated CpG sites based on significance and effect size
    
    Args:
        corr_df: DataFrame with correlation results and corrected p-values
        pval_threshold: Threshold for adjusted p-value
        effect_threshold: Threshold for absolute correlation coefficient
    
    Returns:
        DataFrame: Selected CpG sites meeting criteria
    """
    print(f"Selecting features with p-value < {pval_threshold} and |correlation| > {effect_threshold}...")
    
    # Filter based on adjusted p-value and effect size
    selected = corr_df[
        (corr_df['adj_p_value'] < pval_threshold) & 
        (abs(corr_df['correlation']) > effect_threshold)
    ]
    
    print(f"Selected {len(selected)} CpG sites meeting criteria")
    print(f"Positive correlations: {(selected['correlation'] > 0).sum()}")
    print(f"Negative correlations: {(selected['correlation'] < 0).sum()}")
    
    return selected


def save_outputs(corr_df, selected_features, result_dir="result"):
    """
    Save correlation results and selected features
    
    Args:
        corr_df: DataFrame with all correlation results
        selected_features: DataFrame with selected features
        result_dir: Directory to save results
    """
    print(f"Saving outputs to {result_dir}...")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Save full correlation results
    corr_df.to_csv(os.path.join(result_dir, "2_correlation_results.csv"), index=False)
    
    # Save only the selected features
    selected_features.to_csv(os.path.join(result_dir, "2_selected_features.csv"), index=False)
    
    # Save summary statistics
    summary_data = {
        'total_cpg_sites': len(corr_df),
        'significant_sites': (corr_df['significant'] == True).sum(),
        'selected_sites': len(selected_features),
        'pval_threshold': 0.05,
        'effect_threshold': 0.2,
        'positive_correlations': (selected_features['correlation'] > 0).sum(),
        'negative_correlations': (selected_features['correlation'] < 0).sum()
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join(result_dir, "2_summary.txt"), sep='\t', index=False)
    
    print(f"Saved {len(selected_features)} selected features to 2_selected_features.csv")


def main():
    """
    Main execution function
    """
    print("Starting Phase 2: Feature Selection via Correlation Analysis\n")
    
    # Load data from Phase 1
    data = load_data()
    
    # Calculate correlations
    corr_results = calculate_correlations(data)
    
    # Apply multiple testing correction
    corr_results = apply_multiple_testing_correction(corr_results)
    
    # Select features based on significance and effect size
    selected_features = select_features(corr_results)
    
    # Save outputs
    save_outputs(corr_results, selected_features)
    
    print(f"\nPhase 2 completed successfully!")
    print(f"- Analyzed {len(corr_results)} CpG sites")
    print(f"- Selected {len(selected_features)} age-associated sites")
    print(f"- Results saved to 'result' directory with '2_' prefix")


if __name__ == "__main__":
    main()