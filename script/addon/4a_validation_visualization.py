#!/usr/bin/env python3
"""
Phase 4a: Validation Visualization
Aging Clock Project - Addon Script

Creates validation visualizations:
1. Coefficient Stability Analysis for all selected sites
2. Subgroup performance plots (age and gender) using same style as Phase 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_validation_data(result_dir="result"):
    """Load Phase 4 validation results"""
    print("Loading Phase 4 validation results...")
    
    # Load coefficient stability data
    stability_path = os.path.join(result_dir, "4_coefficient_stability.csv")
    stability_df = pd.read_csv(stability_path)
    
    # Load CV predictions from Phase 3
    predictions_path = os.path.join(result_dir, "3_cv_predictions.csv")
    predictions_df = pd.read_csv(predictions_path)
    
    # Load metadata for gender information
    try:
        metadata_path = os.path.join("data", "Metadata_PY_104.csv")
        metadata_df = pd.read_csv(metadata_path)
        predictions_df = predictions_df.merge(metadata_df[['Sample', 'Gender']], 
                                            left_on='Sample_ID', right_on='Sample', how='left')
    except FileNotFoundError:
        print("Gender metadata not found, will create age-only plots")
        metadata_df = None
    
    print(f"Loaded stability data for {len(stability_df)} CpG sites")
    print(f"Loaded predictions for {len(predictions_df)} samples")
    
    return {
        'stability': stability_df,
        'predictions': predictions_df
    }


def plot_coefficient_stability(results, figure_dir="figure"):
    """
    Create coefficient stability visualization - bar plot only (selection frequency)
    
    Args:
        results (dict): Validation results
        figure_dir (str): Directory to save the plot
    """
    print("Creating coefficient stability plot (selection frequency only)...")
    
    stability_df = results['stability'].copy()
    
    # Filter to only non-zero coefficients (actually selected features)
    selected_features = stability_df[stability_df['original_coeff'] != 0].copy()
    
    # Sort by absolute coefficient value
    selected_features = selected_features.sort_values('original_coeff', key=abs, ascending=True)
    
    # Create single figure for bar plot only
    fig, ax = plt.subplots(figsize=(6, 10))
    
    y_pos = np.arange(len(selected_features))
    
    # Color code by stability level
    colors = ['red' if freq < 0.7 else 'orange' if freq < 0.9 else 'green' 
              for freq in selected_features['selection_frequency']]
    
    # Create horizontal bar plot
    bars = ax.barh(y_pos, selected_features['selection_frequency'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Selection Frequency (Bootstrap)', fontsize=12)
    ax.set_ylabel('CpG Sites (sorted by importance)', fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-2, 52.5)
    ax.set_title('Feature Selection Stability Across Bootstrap Samples\n' +
                f'{len(selected_features)} CpG Sites Selected by LASSO', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add stability threshold lines
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Low stability (<70%)')
    ax.axvline(x=0.9, color='orange', linestyle='--', alpha=0.7, label='Good stability (90%+)')
    ax.legend()
       
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(figure_dir, "4a_coefficient_stability.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Coefficient stability plot saved to: {output_path}")
    
    return fig


def plot_subgroup_performance(results, figure_dir="figure"):
    """
    Create subgroup performance plots - only first row (Performance by X)
    
    Args:
        results (dict): Validation results
        figure_dir (str): Directory to save the plot
    """
    print("Creating subgroup performance plots...")
    
    predictions_df = results['predictions'].copy()
    
    # Create age groups
    predictions_df['age_group'] = pd.cut(predictions_df['Actual_Age'], 
                                       bins=[0, 40, 60, 100], 
                                       labels=['Young (≤40)', 'Middle (40-60)', 'Elderly (≥60)'])
    
    # Set up the plotting style to match Phase 3a
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with only the first row subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Age Group Performance (Young, Middle, Elderly)
    ax1 = plt.subplot(121)
    
    for age_group in ['Young (≤40)', 'Middle (40-60)', 'Elderly (≥60)']:
        group_data = predictions_df[predictions_df['age_group'] == age_group]
        
        scatter = ax1.scatter(group_data['Actual_Age'], group_data['Predicted_Age'], 
                             alpha=0.7, s=60, label=f'{age_group} (n={len(group_data)})',
                             edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    min_age = predictions_df['Actual_Age'].min()
    max_age = predictions_df['Actual_Age'].max()
    ax1.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, 
            label='Perfect Prediction', alpha=0.7)
    
    # Calculate MAE for each group and add to plot
    for age_group in ['Young (≤40)', 'Middle (40-60)', 'Elderly (≥60)']:
        group_data = predictions_df[predictions_df['age_group'] == age_group]
        mae = np.mean(np.abs(group_data['Predicted_Age'] - group_data['Actual_Age']))
        
        # Reposition text boxes for better visibility in 2-panel layout
        if age_group == 'Young (≤40)':
            # Position in upper left for young group
            ax1.text(25, 75, f'MAE: {mae:.1f}y', 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                             edgecolor='black', linewidth=0.5))
        elif age_group == 'Middle (40-60)':
            # Position in center for middle group
            ax1.text(50, 35, f'MAE: {mae:.1f}y', 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                             edgecolor='black', linewidth=0.5))
        else:  # Elderly
            # Position in lower right for elderly group
            ax1.text(75, 25, f'MAE: {mae:.1f}y', 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                             edgecolor='black', linewidth=0.5))
    
    ax1.set_xlabel('Actual Age (years)', fontsize=12)
    ax1.set_ylabel('Predicted Age (years)', fontsize=12)
    ax1.set_title('Performance by Age Group', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gender Performance
    if 'Gender' in predictions_df.columns:
        ax2 = plt.subplot(122)
        
        for gender in ['M', 'F']:
            gender_data = predictions_df[predictions_df['Gender'] == gender]
            
            scatter = ax2.scatter(gender_data['Actual_Age'], gender_data['Predicted_Age'], 
                                 alpha=0.7, s=60, label=f'{gender} (n={len(gender_data)})',
                                 edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line
        ax2.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, 
                label='Perfect Prediction', alpha=0.7)
        
        # Add MAE for each gender - reposition for better 2-panel layout
        for gender in ['M', 'F']:
            gender_data = predictions_df[predictions_df['Gender'] == gender]
            mae = np.mean(np.abs(gender_data['Predicted_Age'] - gender_data['Actual_Age']))
            
            # Reposition text boxes for better balance in 2-panel layout
            if gender == 'M':
                # Position in upper right for males
                ax2.text(70, 70, f'Male MAE: {mae:.1f}y', 
                        fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                                 edgecolor='black', linewidth=0.5))
            else:  # Female
                # Position in lower left for females
                ax2.text(30, 25, f'Female MAE: {mae:.1f}y', 
                        fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                                 edgecolor='black', linewidth=0.5))
        
        ax2.set_xlabel('Actual Age (years)', fontsize=12)
        ax2.set_ylabel('Predicted Age (years)', fontsize=12)
        ax2.set_title('Performance by Gender', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2 = plt.subplot(122)
        ax2.text(0.5, 0.5, 'Gender data not available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Performance by Gender', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(figure_dir, "4a_subgroup_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Subgroup performance plot saved to: {output_path}")
    
    return fig


def main():
    """Main execution function"""
    print("Starting Phase 4a: Validation Visualization\n")
    
    # Set style to match Phase 3a
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Load validation data
    results = load_validation_data()
    
    # Create visualizations
    plot_coefficient_stability(results)
    plot_subgroup_performance(results)
    
    print(f"\nPhase 4a completed successfully!")
    print(f"- Generated 2 validation visualization plots")
    print(f"- Plots saved to 'figure' directory with '4a_' prefix")


if __name__ == "__main__":
    main()
