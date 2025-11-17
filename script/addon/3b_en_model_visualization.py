#!/usr/bin/env python3
"""
Phase 3b: Elastic Net Model Visualization
Aging Clock Project - Addon Script for EN Results

This script creates visualizations for the Phase 3a Elastic Net model results:
1. Predicted vs Actual age plot
2. Residual plot
3. Feature importance (coefficients)
4. Age acceleration distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_model_results(result_dir="result"):
    """
    Load model results from Phase 3
    
    Args:
        result_dir (str): Directory where Phase 3 results are stored
    
    Returns:
        dict: Loaded results including predictions, metrics, and coefficients
    """
    print("Loading Elastic Net model results from Phase 3a...")
    
    # Load CV predictions
    predictions_path = os.path.join(result_dir, "5a_cv_predictions.csv")
    predictions_df = pd.read_csv(predictions_path)
    
    # Load metrics
    metrics_path = os.path.join(result_dir, "5a_cv_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    
    # Load coefficients
    coefficients_path = os.path.join(result_dir, "5a_model_coefficients.csv")
    coefficients_df = pd.read_csv(coefficients_path)
    
    print(f"Loaded predictions for {len(predictions_df)} samples")
    print(f"Elastic Net model uses {len(coefficients_df)} CpG sites")
    
    return {
        'predictions': predictions_df,
        'metrics': metrics_df,
        'coefficients': coefficients_df
    }


def plot_predicted_vs_actual(results, figure_dir="figure"):
    """
    Create predicted vs actual age plot
    
    Args:
        results (dict): Model results
        figure_dir (str): Directory to save the plot
    """
    print("Creating Elastic Net predicted vs actual plot...")
    
    df = results['predictions']
    metrics = results['metrics'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(df['Actual_Age'], df['Predicted_Age'], 
                        alpha=0.7, s=60, c=df['Age_Acceleration'], 
                        cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age Acceleration (years)', fontsize=12)
    
    # Add perfect prediction line
    min_age = min(df['Actual_Age'].min(), df['Predicted_Age'].min())
    max_age = max(df['Actual_Age'].max(), df['Predicted_Age'].max())
    ax.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, 
            label='Perfect Prediction', alpha=0.7)
    
    # Add metrics text box
    metrics_text = f"""MAE: {metrics['MAE']:.2f} years
RMSE: {metrics['RMSE']:.2f} years
R²: {metrics['R2']:.3f}
Correlation: {metrics['Correlation']:.3f}"""
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Actual Age (years)', fontsize=12)
    ax.set_ylabel('Predicted Age (years)', fontsize=12)
    ax.set_title('Aging Clock: Predicted vs Actual Age\n' + 
                f'({len(df)} samples, {metrics["n_features"]} CpG sites)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(figure_dir, "3b_predicted_vs_actual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Elastic Net predicted vs actual plot saved to: {output_path}")
    
    return fig


def plot_residuals(results, figure_dir="figure"):
    """
    Create residual plot (errors vs actual age)
    
    Args:
        results (dict): Model results
        figure_dir (str): Directory to save the plot
    """
    print("Creating Elastic Net residual plot...")
    
    df = results['predictions']
    
    # Calculate residuals
    residuals = df['Predicted_Age'] - df['Actual_Age']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Create main axes for the two subplots
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    # Plot 1: Residuals vs Actual Age
    scatter1 = ax1.scatter(df['Actual_Age'], residuals, 
                          alpha=0.7, s=60, c=df['Age_Acceleration'], 
                          cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Actual Age (years)', fontsize=12)
    ax1.set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    ax1.set_title('Elastic Net: Residuals vs Actual Age', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Predicted Age
    scatter2 = ax2.scatter(df['Predicted_Age'], residuals, 
                          alpha=0.7, s=60, c=df['Age_Acceleration'], 
                          cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Predicted Age (years)', fontsize=12)
    ax2.set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    ax2.set_title('Elastic Net: Residuals vs Predicted Age', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Use make_axes_locatable to add colorbar on the right
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter2, cax=cax)
    cbar.set_label('Age Acceleration (years)', fontsize=12)
    
    # Ensure both subplots are same size
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')
    
    # Save plot
    output_path = os.path.join(figure_dir, "3b_residuals.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Elastic Net residual plot saved to: {output_path}")
    
    return fig

def plot_coefficients(results, figure_dir="figure"):
    """
    Create feature importance plot (model coefficients)
    
    Args:
        results (dict): Model results
        figure_dir (str): Directory to save the plot
    """
    print("Creating Elastic Net coefficients plot...")
    
    coef_df = results['coefficients'].copy()
    
    # Extract chromosome information for coloring
    def extract_chrom(cpg_str):
        try:
            return cpg_str.split("'")[1]
        except:
            return 'Unknown'
    
    coef_df['chromosome'] = coef_df['CpG_site'].apply(extract_chrom)
    
    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('coefficient', key=abs, ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create color palette for chromosomes
    unique_chroms = coef_df['chromosome'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_chroms)))
    chrom_color_map = dict(zip(unique_chroms, colors))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(coef_df))
    bar_colors = [chrom_color_map[chrom] for chrom in coef_df['chromosome']]
    
    bars = ax.barh(y_pos, coef_df['coefficient'], color=bar_colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Color positive and negative bars differently with edge
    for i, (bar, coef) in enumerate(zip(bars, coef_df['coefficient'])):
        if coef > 0:
            bar.set_edgecolor('darkred')
            bar.set_linewidth(1.5)
        else:
            bar.set_edgecolor('darkblue')
            bar.set_linewidth(1.5)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([cpg[:50] + '...' if len(cpg) > 50 else cpg 
                       for cpg in coef_df['CpG_site']], fontsize=8)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title('Elastic Net Aging Clock Model Coefficients\n' + 
                f'{len(coef_df)} CpG Sites Selected by Elastic Net', fontsize=14)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add legend for positive/negative
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='darkred', 
              label=f'Positive ({(coef_df["coefficient"] > 0).sum()} sites)'),
        Patch(facecolor='lightgray', edgecolor='darkblue', 
              label=f'Negative ({(coef_df["coefficient"] < 0).sum()} sites)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(figure_dir, "3b_coefficients.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Elastic Net coefficients plot saved to: {output_path}")
    
    return fig


def plot_age_acceleration(results, figure_dir="figure"):
    """
    Create age acceleration distribution plot (histogram only)
    
    Args:
        results (dict): Model results
        figure_dir (str): Directory to save the plot
    """
    print("Creating Elastic Net age acceleration plot...")
    
    df = results['predictions']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot: Distribution of age acceleration
    ax.hist(df['Age_Acceleration'], bins=20, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                label='No Acceleration')
    ax.axvline(x=df['Age_Acceleration'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {df["Age_Acceleration"].mean():.2f} years')
    
    ax.set_xlabel('Age Acceleration (years)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Elastic Net: Distribution of Age Acceleration', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with statistics
    accel_mean = df['Age_Acceleration'].mean()
    accel_std = df['Age_Acceleration'].std()
    accel_range = df['Age_Acceleration'].max() - df['Age_Acceleration'].min()
    
    stats_text = f"""Mean: {accel_mean:.2f} years
Std: {accel_std:.2f} years
Range: {accel_range:.2f} years
Max positive: {df['Age_Acceleration'].max():.2f}
Max negative: {df['Age_Acceleration'].min():.2f}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, 
             fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(figure_dir, "3b_en_age_acceleration.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Elastic Net age acceleration plot saved to: {output_path}")
    
    return fig


def main():
    """
    Main execution function
    """
    print("Starting Phase 3b: Elastic Net Model Visualization\n")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Load results
    results = load_model_results()
    
    # Create visualizations
    plot_predicted_vs_actual(results)
    plot_residuals(results)
    plot_coefficients(results)
    plot_age_acceleration(results)
    
    print(f"\nPhase 3b completed successfully!")
    print(f"- Generated 4 visualization plots for Elastic Net results")
    print(f"- Plots saved to 'figure' directory with '3b_' prefix")


if __name__ == "__main__":
    main()
