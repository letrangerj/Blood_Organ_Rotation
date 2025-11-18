#!/usr/bin/env python3
"""
Phase 5: Age Correlation Analysis Based on Residuals

This script performs residual analysis for all aging clock models to:
1. Test age-residual correlation for each model
2. Apply age-based correction to all models (regardless of significance)
3. Generate corrected age acceleration values (primary result)
4. Compare model performance before and after correction
5. Create diagnostic visualizations

Author: Aging Clock Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define model information
MODELS = {
    '3_': 'LASSO',
    '5a_': 'Elastic Net',
    '3b_': 'Ridge',
    '3c_': 'PLSR',
    '3d_': 'SVR',
    '3e_': 'Random Forest',
    '3f_': 'XGBoost',
    '3g_': 'LightGBM'
}

def load_model_predictions(model_prefix):
    """Load CV predictions for a specific model"""
    filepath = f'result/{model_prefix}cv_predictions.csv'
    try:
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} predictions from {filepath}")
        return df
    except FileNotFoundError:
        print(f"  WARNING: File not found: {filepath}")
        return None

def calculate_residual_correlation(residuals, actual_ages):
    """
    Perform linear regression to test age-residual correlation
    Returns: beta_0, beta_1, r_squared, p_value, correlation_coef
    """
    # Reshape for sklearn
    X = actual_ages.values.reshape(-1, 1)
    y = residuals.values

    # Fit linear regression: residual = beta_0 + beta_1 * actual_age
    model = LinearRegression()
    model.fit(X, y)

    beta_0 = model.intercept_
    beta_1 = model.coef_[0]

    # Calculate R-squared
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)

    # Calculate correlation and p-value
    correlation_coef, p_value = stats.pearsonr(actual_ages, residuals)

    return beta_0, beta_1, r_squared, p_value, correlation_coef

def apply_age_correction(predictions, actual_ages, beta_0, beta_1):
    """
    Apply age-based correction to predictions
    Corrected_prediction = Original_prediction - (beta_0 + beta_1 * actual_age)
    """
    correction = beta_0 + beta_1 * actual_ages.values
    corrected_predictions = predictions.values - correction
    return corrected_predictions

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    corr, _ = stats.pearsonr(actual, predicted)
    return mae, rmse, r2, corr

def plot_residual_diagnostics(results_df, correction_params_df, output_dir='figure'):
    """Generate residual diagnostic plots"""

    # Create figure directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(MODELS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (prefix, model_name) in enumerate(MODELS.items()):
        ax = axes[idx]

        # Get data for this model
        actual_col = 'Actual_Age'
        original_residual_col = f'{model_name}_original_residual'
        corrected_diff_col = f'{model_name}_corrected_age_diff'

        if original_residual_col not in results_df.columns:
            ax.text(0.5, 0.5, f'{model_name}\nNo data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        actual_age = results_df[actual_col]
        original_residual = results_df[original_residual_col]
        corrected_residual = results_df[corrected_diff_col]

        # Plot original residuals
        ax.scatter(actual_age, original_residual, alpha=0.5, s=30,
                  color='#d62728', label='Before correction', edgecolors='black', linewidth=0.5)

        # Plot corrected residuals
        ax.scatter(actual_age, corrected_residual, alpha=0.5, s=30,
                  color='#2ca02c', label='After correction', edgecolors='black', linewidth=0.5)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Get correction parameters
        params = correction_params_df[correction_params_df['model'] == model_name]
        if not params.empty:
            beta_1 = params['beta_1'].values[0]
            p_value = params['p_value'].values[0]
            r_squared = params['r_squared'].values[0]

            # Add regression line for original residuals
            z = np.polyfit(actual_age, original_residual, 1)
            p = np.poly1d(z)
            x_line = np.array([actual_age.min(), actual_age.max()])
            ax.plot(x_line, p(x_line), color='#d62728', linestyle='-',
                   linewidth=2, alpha=0.7)

            # Add statistics text
            stats_text = f'β₁={beta_1:.4f}\np={p_value:.4f}\nR²={r_squared:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Actual Age (years)', fontsize=9)
        ax.set_ylabel('Residual (years)', fontsize=9)
        ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'{output_dir}/5_residual_vs_age_all_models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved residual diagnostic plot: {output_path}")
    plt.close()
    return

def main():
    print("="*80)
    print("Phase 5: Age Correlation Analysis Based on Residuals")
    print("="*80)

    # Step 1: Load all model predictions
    print("\n[Step 1] Loading CV predictions from all models...")
    all_predictions = {}
    for prefix, model_name in MODELS.items():
        print(f"  Loading {model_name}...")
        df = load_model_predictions(prefix)
        if df is not None:
            all_predictions[model_name] = df

    print(f"\n  Successfully loaded {len(all_predictions)} models")

    if len(all_predictions) == 0:
        print("ERROR: No prediction files found!")
        return

    # Step 2: Analyze residuals and test age correlation for each model
    print("\n[Step 2] Analyzing residuals and testing age-residual correlation...")
    correction_params = []

    for model_name, df in all_predictions.items():
        print(f"\n  {model_name}:")

        # Extract data
        actual_age = df['Actual_Age']
        predicted_age = df['Predicted_Age']
        residuals = df['Age_Acceleration']  # This is predicted - actual

        # Calculate original metrics
        mae_before, rmse_before, r2_before, corr_before = calculate_metrics(actual_age, predicted_age)
        print(f"    Before correction: MAE={mae_before:.3f}, RMSE={rmse_before:.3f}, R²={r2_before:.3f}, Corr={corr_before:.3f}")

        # Test age-residual correlation
        beta_0, beta_1, r_squared, p_value, correlation_coef = calculate_residual_correlation(residuals, actual_age)
        print(f"    Residual-Age correlation: β₀={beta_0:.4f}, β₁={beta_1:.4f}, R²={r_squared:.4f}, p={p_value:.4f}, r={correlation_coef:.4f}")

        if p_value < 0.05:
            print(f"    *** Significant age-dependent bias detected (p < 0.05) ***")
        else:
            print(f"    No significant age-dependent bias detected (p >= 0.05)")

        # Apply age-based correction (REGARDLESS of significance)
        print(f"    Applying age-based correction to all samples...")
        corrected_predictions = apply_age_correction(predicted_age, actual_age, beta_0, beta_1)

        # Calculate corrected metrics
        mae_after, rmse_after, r2_after, corr_after = calculate_metrics(actual_age, corrected_predictions)
        print(f"    After correction:  MAE={mae_after:.3f}, RMSE={rmse_after:.3f}, R²={r2_after:.3f}, Corr={corr_after:.3f}")

        # Calculate improvement
        mae_improvement = ((mae_before - mae_after) / mae_before) * 100
        rmse_improvement = ((rmse_before - rmse_after) / rmse_before) * 100
        print(f"    Improvement: MAE={mae_improvement:.1f}%, RMSE={rmse_improvement:.1f}%")

        # Store correction parameters
        correction_params.append({
            'model': model_name,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'r_squared': r_squared,
            'p_value': p_value,
            'correlation_coef': correlation_coef,
            'correlation_significant': p_value < 0.05,
            'mae_before': mae_before,
            'rmse_before': rmse_before,
            'r2_before': r2_before,
            'corr_before': corr_before,
            'mae_after': mae_after,
            'rmse_after': rmse_after,
            'r2_after': r2_after,
            'corr_after': corr_after,
            'mae_improvement_pct': mae_improvement,
            'rmse_improvement_pct': rmse_improvement
        })

        # Store corrected predictions in the dataframe
        df['corrected_prediction'] = corrected_predictions
        df['corrected_age_diff'] = corrected_predictions - actual_age.values

    # Step 3: Compile results into comprehensive dataframes
    print("\n[Step 3] Compiling results...")

    # Get sample IDs from first available model
    first_model = list(all_predictions.values())[0]
    sample_ids = first_model['Sample_ID'].values
    actual_ages = first_model['Actual_Age'].values

    # Create comprehensive results dataframe
    results_data = {
        'Sample_ID': sample_ids,
        'Actual_Age': actual_ages
    }

    for model_name, df in all_predictions.items():
        results_data[f'{model_name}_original_prediction'] = df['Predicted_Age'].values
        results_data[f'{model_name}_original_residual'] = df['Age_Acceleration'].values
        results_data[f'{model_name}_corrected_prediction'] = df['corrected_prediction'].values
        results_data[f'{model_name}_corrected_age_diff'] = df['corrected_age_diff'].values

    results_df = pd.DataFrame(results_data)

    # Create correction parameters dataframe
    correction_params_df = pd.DataFrame(correction_params)

    # Create summary dataframe
    summary_data = []
    for model_name in all_predictions.keys():
        params = correction_params_df[correction_params_df['model'] == model_name].iloc[0]
        summary_data.append({
            'model': model_name,
            'mae_before': params['mae_before'],
            'mae_after': params['mae_after'],
            'mae_improvement_pct': params['mae_improvement_pct'],
            'rmse_before': params['rmse_before'],
            'rmse_after': params['rmse_after'],
            'rmse_improvement_pct': params['rmse_improvement_pct'],
            'r2_before': params['r2_before'],
            'r2_after': params['r2_after'],
            'age_bias_detected': params['correlation_significant'],
            'correction_beta_1': params['beta_1'],
            'correction_p_value': params['p_value']
        })

    summary_df = pd.DataFrame(summary_data)

    # Step 4: Save results
    print("\n[Step 4] Saving results...")

    # Save corrected age diffs (PRIMARY RESULT)
    output_file = 'result/phase5_corrected_age_diffs.csv'
    results_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save correction parameters
    output_file = 'result/phase5_correction_parameters.csv'
    correction_params_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save summary
    output_file = 'result/phase5_residual_analysis_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Step 5: Generate visualizations
    print("\n[Step 5] Generating diagnostic visualizations...")
    plot_residual_diagnostics(results_df, correction_params_df)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nSummary of corrections:")
    print(summary_df.to_string(index=False))

    print("\nKey outputs:")
    print("  1. result/phase5_corrected_age_diffs.csv - Corrected age acceleration for all samples")
    print("  2. result/phase5_correction_parameters.csv - Correction parameters for each model")
    print("  3. result/phase5_residual_analysis_summary.csv - Performance comparison before/after")
    print("  4. figure/5_residual_vs_age_all_models.png - Residual diagnostic plots")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
