#!/usr/bin/env python3
"""
Phase 1: Data Preparation & Exploration
Aging Clock Project

This script performs:
1. Load and validate methylation matrix and metadata
2. Quality control checks
3. Data transformation (transpose methylation matrix)
4. Generate summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def load_data(data_dir="data"):
    """
    Load methylation matrix and metadata files
    
    Args:
        data_dir (str): Path to data directory
    
    Returns:
        tuple: (methylation_df, metadata_df)
    """
    print("Loading data...")
    
    # Load methylation matrix
    methylation_path = os.path.join(data_dir, "overlap_cfDNA.tsv")
    methylation_df = pd.read_csv(methylation_path, sep='\t')
    print(f"Methylation matrix loaded: {methylation_df.shape[0]} CpG sites x {methylation_df.shape[1]} samples")
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "Metadata_PY_104.csv")
    metadata_df = pd.read_csv(metadata_path)
    print(f"Metadata loaded: {metadata_df.shape[0]} samples")
    
    return methylation_df, metadata_df


def validate_data(methylation_df, metadata_df):
    """
    Validate data integrity and correspondence between files
    
    Args:
        methylation_df: Methylation matrix DataFrame
        metadata_df: Metadata DataFrame
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("\nValidating data...")
    
    # Check basic dimensions and structure
    print(f"Methylation matrix shape: {methylation_df.shape}")
    print(f"Metadata shape: {metadata_df.shape}")
    
    # Check that methylation data has expected chromosome, start, end, MD columns
    expected_cols = ['chr', 'start', 'end', 'MD']
    if not all(col in methylation_df.columns[:4] for col in expected_cols):
        print("Warning: Expected columns (chr, start, end, MD) not found in first 4 columns of methylation data")
        print(f"Actual first few columns: {list(methylation_df.columns[:6])}")
    else:
        print("✓ Methylation matrix has expected genomic coordinate columns")
    
    # Extract sample IDs from methylation matrix (columns 5 onwards)
    sample_ids_meth = [col for col in methylation_df.columns if col not in ['chr', 'start', 'end', 'MD']]
    print(f"Number of samples in methylation data: {len(sample_ids_meth)}")
    
    # Check sample IDs correspondence
    sample_ids_meta = metadata_df['Sample'].tolist()
    print(f"Number of samples in metadata: {len(sample_ids_meta)}")
    
    # Find intersection of samples
    common_samples = set(sample_ids_meth).intersection(set(sample_ids_meta))
    print(f"Common samples between both files: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("✗ Error: No common samples found between methylation data and metadata!")
        return False
    
    if len(common_samples) < min(len(sample_ids_meth), len(sample_ids_meta)):
        print(f"⚠️  Warning: Not all samples are matched between datasets")
        print(f"Missing from methylation data: {len(set(sample_ids_meta) - set(sample_ids_meth))} samples")
        print(f"Missing from metadata: {len(set(sample_ids_meth) - set(sample_ids_meta))} samples")
    
    # Check age range in metadata
    age_range = (metadata_df['Age'].min(), metadata_df['Age'].max())
    print(f"Age range in cohort: {age_range[0]} - {age_range[1]} years")
    
    return True


def quality_control(methylation_df, metadata_df):
    """
    Perform quality control checks on the data
    
    Args:
        methylation_df: Methylation matrix DataFrame
        metadata_df: Metadata DataFrame
    
    Returns:
        dict: QC results
    """
    print("\nPerforming quality control...")
    qc_results = {}
    
    # Extract sample data (excluding coordinate columns)
    sample_cols = [col for col in methylation_df.columns if col not in ['chr', 'start', 'end', 'MD']]
    methylation_samples = methylation_df[sample_cols]
    
    # Check for missing values in methylation data
    missing_values = methylation_samples.isnull().sum().sum()
    qc_results['total_missing_values'] = missing_values
    qc_results['missing_percentage'] = (missing_values / (methylation_samples.shape[0] * methylation_samples.shape[1])) * 100
    print(f"Total missing values in methylation data: {missing_values} ({qc_results['missing_percentage']:.2f}%)")
    
    # Check methylation value ranges (should be 0-1 for beta values)
    all_values = methylation_samples.values.flatten()
    valid_beta = all_values[~pd.isna(all_values)]  # Exclude NaN values
    out_of_range = np.sum((valid_beta < 0) | (valid_beta > 1))
    qc_results['out_of_range_values'] = out_of_range
    print(f"Values outside 0-1 range: {out_of_range}")
    
    if out_of_range > 0:
        print(f"Values < 0: {np.sum(valid_beta < 0)}")
        print(f"Values > 1: {np.sum(valid_beta > 1)}")
    
    # Check for duplicate samples in metadata
    duplicates_meta = metadata_df.duplicated(subset=['Sample']).sum()
    qc_results['duplicate_samples_metadata'] = duplicates_meta
    print(f"Duplicate samples in metadata: {duplicates_meta}")
    
    # Check for duplicate CpG sites in methylation data
    if 'chr' in methylation_df.columns and 'start' in methylation_df.columns:
        duplicate_cpgs = methylation_df.duplicated(subset=['chr', 'start']).sum()
        qc_results['duplicate_cpg_sites'] = duplicate_cpgs
        print(f"Duplicate CpG sites (by chr:start): {duplicate_cpgs}")
    
    # Examine age distribution
    age_stats = {
        'mean': metadata_df['Age'].mean(),
        'median': metadata_df['Age'].median(),
        'std': metadata_df['Age'].std(),
        'min': metadata_df['Age'].min(),
        'max': metadata_df['Age'].max(),
        'q25': metadata_df['Age'].quantile(0.25),
        'q75': metadata_df['Age'].quantile(0.75)
    }
    qc_results['age_distribution'] = age_stats
    print(f"Age statistics - Mean: {age_stats['mean']:.2f}, Median: {age_stats['median']:.2f}, Std: {age_stats['std']:.2f}")
    
    # Check gender distribution
    if 'Gender' in metadata_df.columns:
        gender_dist = metadata_df['Gender'].value_counts()
        qc_results['gender_distribution'] = gender_dist.to_dict()
        print(f"Gender distribution: {dict(gender_dist)}")
    
    return qc_results


def transform_data(methylation_df, metadata_df):
    """
    Transform methylation matrix and merge with metadata
    
    Args:
        methylation_df: Methylation matrix DataFrame
        metadata_df: Metadata DataFrame
    
    Returns:
        tuple: (transformed_meth_df, merged_df)
    """
    print("\nTransforming data...")
    
    # Extract sample columns (excluding coordinate columns: chr, start, end, MD)
    coord_cols = ['chr', 'start', 'end', 'MD']
    sample_cols = [col for col in methylation_df.columns if col not in coord_cols]
    
    # Transpose methylation matrix: rows = samples, columns = CpG sites
    # First set CpG sites as index (using chr:start as unique identifier)
    methylation_df = methylation_df.set_index(['chr', 'start', 'end', 'MD'])
    
    # Transpose so samples are rows and CpG sites are columns
    methylation_t = methylation_df.T
    print(f"Transposed methylation matrix: {methylation_t.shape[0]} samples x {methylation_t.shape[1]} CpG sites")
    
    # Merge with metadata based on sample IDs
    # Ensure sample IDs in methylation data match those in metadata
    sample_ids_meth = set(methylation_t.index)
    sample_ids_meta = set(metadata_df['Sample'])
    common_samples = sample_ids_meth.intersection(sample_ids_meta)
    
    if len(common_samples) == 0:
        raise ValueError("No common samples between methylation data and metadata!")
    
    # Keep only common samples
    methylation_t = methylation_t.loc[list(common_samples)]
    metadata_filtered = metadata_df[metadata_df['Sample'].isin(common_samples)]
    metadata_filtered = metadata_filtered.set_index('Sample')
    
    # Reorder metadata to match methylation sample order
    metadata_filtered = metadata_filtered.reindex(methylation_t.index)
    
    # Merge methylation data with metadata
    merged_df = pd.concat([metadata_filtered, methylation_t], axis=1)
    print(f"Merged dataset: {merged_df.shape[0]} samples x {merged_df.shape[1]} features")
    
    return methylation_t, merged_df


def generate_summary_statistics(merged_df, methylation_t):
    """
    Generate summary statistics for the dataset
    
    Args:
        merged_df: Merged DataFrame with metadata and methylation data
        methylation_t: Transposed methylation matrix
    
    Returns:
        dict: Summary statistics
    """
    print("\nGenerating summary statistics...")
    summary_stats = {}
    
    # Basic dataset info
    summary_stats['n_samples'] = merged_df.shape[0]
    summary_stats['n_cpg_sites'] = methylation_t.shape[1]
    summary_stats['n_features_total'] = merged_df.shape[1]
    
    print(f"Number of CpG sites: {summary_stats['n_cpg_sites']}")
    print(f"Number of samples: {summary_stats['n_samples']}")
    
    # Age distribution
    age_col = merged_df['Age']
    summary_stats['age_mean'] = age_col.mean()
    summary_stats['age_median'] = age_col.median()
    summary_stats['age_std'] = age_col.std()
    summary_stats['age_min'] = age_col.min()
    summary_stats['age_max'] = age_col.max()
    
    # Methylation distribution per site
    # Calculate mean and variance for each CpG site across samples
    methylation_means = methylation_t.mean(axis=0)
    methylation_vars = methylation_t.var(axis=0)
    
    summary_stats['methylation_mean_overall'] = methylation_means.mean()
    summary_stats['methylation_var_overall'] = methylation_vars.mean()
    summary_stats['methylation_mean_range'] = (methylation_means.min(), methylation_means.max())
    summary_stats['methylation_var_range'] = (methylation_vars.min(), methylation_vars.max())
    
    print(f"Age statistics - Mean: {summary_stats['age_mean']:.2f}, Range: {summary_stats['age_min']:.0f}-{summary_stats['age_max']:.0f}")
    print(f"Avg methylation level: {summary_stats['methylation_mean_overall']:.3f}")
    print(f"Avg methylation variance: {summary_stats['methylation_var_overall']:.3f}")
    
    return summary_stats


def save_outputs(transformed_meth, merged_df, qc_results, summary_stats, result_dir="result"):
    """
    Save processed data and results
    
    Args:
        transformed_meth: Transposed methylation matrix
        merged_df: Merged dataset
        qc_results: Quality control results
        summary_stats: Summary statistics
        result_dir: Directory to save results
    """
    print(f"\nSaving outputs to {result_dir}...")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Save only the essential transposed methylation matrix with age data
    # Include only samples with age information
    samples_with_age = merged_df[['Age']].dropna()
    final_data = merged_df.loc[samples_with_age.index]
    final_data.to_csv(os.path.join(result_dir, "1_preprocessed_data.csv"))
    
    # Save only key summary statistics to a minimal text file
    with open(os.path.join(result_dir, "1_summary.txt"), 'w') as f:
        f.write("Phase 1 - Data Preparation Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Number of samples: {summary_stats['n_samples']}\n")
        f.write(f"Number of CpG sites: {summary_stats['n_cpg_sites']}\n")
        f.write(f"Age range: {summary_stats['age_min']:.0f} - {summary_stats['age_max']:.0f} years\n")
        f.write(f"Age mean: {summary_stats['age_mean']:.2f} years\n")
        f.write(f"Missing values in methylation data: {qc_results['total_missing_values']} ({qc_results['missing_percentage']:.2f}%)\n")
        f.write(f"Out of range methylation values: {qc_results['out_of_range_values']}\n")


def main():
    """
    Main execution function
    """
    print("Starting Phase 1: Data Preparation & Exploration\n")
    
    # Load data
    methylation_df, metadata_df = load_data()
    
    # Validate data
    validation_passed = validate_data(methylation_df, metadata_df)
    if not validation_passed:
        print("Data validation failed. Please check the data files.")
        return
    
    # Quality control
    qc_results = quality_control(methylation_df, metadata_df)
    
    # Transform data
    transformed_meth, merged_df = transform_data(methylation_df, metadata_df)
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(merged_df, transformed_meth)
    
    # Save outputs
    save_outputs(transformed_meth, merged_df, qc_results, summary_stats)
    
    print(f"\nPhase 1 completed successfully!")
    print(f"- Processed {summary_stats['n_samples']} samples")
    print(f"- Processed {summary_stats['n_cpg_sites']} CpG sites")
    print(f"- Saved outputs to 'result' directory")


if __name__ == "__main__":
    main()