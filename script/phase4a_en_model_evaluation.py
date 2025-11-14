#!/usr/bin/env python3
"""
Phase 4a: Elastic Net Model Evaluation & Validation
Aging Clock Project - Elastic Net Model Validation

Evaluates Elastic Net model performance, stability, and biological validity.
This script performs comprehensive validation of the trained Elastic Net aging clock model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
import os
from scipy import stats


def load_model_and_data(result_dir="result"):
    """Load trained Elastic Net model and data"""
    print("Loading Elastic Net model and data...")
    
    # Load model
    with open(os.path.join(result_dir, "5a_trained_model.pkl"), 'rb') as f:
        model_package = pickle.load(f)
    
    # Load predictions from Phase 3a
    cv_predictions = pd.read_csv(os.path.join(result_dir, "5a_cv_predictions.csv"))
    coefficients = pd.read_csv(os.path.join(result_dir, "5a_model_coefficients.csv"))
    
    print(f"Loaded {len(coefficients)} CpG sites, {len(cv_predictions)} samples")
    return model_package, cv_predictions, coefficients


def residual_analysis(cv_predictions):
    """Analyze residuals for systematic patterns"""
    print("\nResidual analysis...")
    
    residuals = cv_predictions['Predicted_Age'] - cv_predictions['Actual_Age']
    
    # Basic residual statistics
    residual_stats = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'min': residuals.min(),
        'max': residuals.max(),
        'shapiro_pvalue': stats.shapiro(residuals)[1]  # Normality test
    }
    
    # Correlation with actual age (check for age-dependent bias)
    age_corr = np.corrcoef(cv_predictions['Actual_Age'], residuals)[0, 1]
    
    # Age group analysis
    cv_predictions['age_group'] = pd.cut(cv_predictions['Actual_Age'], 
                                       bins=[0, 40, 60, 100], 
                                       labels=['young', 'middle', 'elderly'])
    
    group_performance = cv_predictions.groupby('age_group').apply(
        lambda x: mean_absolute_error(x['Actual_Age'], x['Predicted_Age'])
    ).round(2)
    
    print(f"Residual mean: {residual_stats['mean']:.2f}, std: {residual_stats['std']:.2f}")
    print(f"Age-residual correlation: {age_corr:.3f}")
    print(f"MAE by age group:")
    for group, mae in group_performance.items():
        print(f"  {group}: {mae:.2f} years")
    
    return residual_stats, age_corr, group_performance


def subgroup_analysis(cv_predictions):
    """Analyze performance by subgroups"""
    print("\nSubgroup analysis...")
    
    # Load metadata for gender information
    try:
        metadata = pd.read_csv("data/Metadata_PY_104.csv")
        cv_predictions = cv_predictions.merge(metadata[['Sample', 'Gender']], 
                                            left_on='Sample_ID', right_on='Sample', how='left')
        
        gender_performance = cv_predictions.groupby('Gender').apply(
            lambda x: mean_absolute_error(x['Actual_Age'], x['Predicted_Age'])
        ).round(2)
        print("MAE by gender:")
        for gender, mae in gender_performance.items():
            print(f"  {gender}: {mae:.2f} years")
        
    except FileNotFoundError:
        print("Gender metadata not found, skipping gender analysis")
        gender_performance = None
    
    return gender_performance


def coefficient_stability(model_package, cv_predictions, n_bootstrap=100):
    """Assess coefficient stability via bootstrapping for Elastic Net"""
    print(f"\nCoefficient stability analysis ({n_bootstrap} bootstrap samples)...")
    
    # Load original data
    data = pd.read_csv("result/1_preprocessed_data.csv", index_col=0)
    cpg_sites = model_package['cpg_sites']
    
    X = data[cpg_sites].values
    y = data['Age'].values
    
    # Handle missing values
    if np.isnan(X).any():
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    bootstrap_coeffs = []
    bootstrap_alphas = []
    bootstrap_l1_ratios = []
    
    for i in range(n_bootstrap):
        if i % 20 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")
        
        # Resample with replacement
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Standardize
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        
        # Train Elastic Net with same hyperparameters as final model
        alpha = model_package['alpha']
        l1_ratio = model_package['l1_ratio']
        
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
        en.fit(X_boot_scaled, y_boot)
        
        bootstrap_coeffs.append(en.coef_)
        bootstrap_alphas.append(alpha)
        bootstrap_l1_ratios.append(l1_ratio)
    
    bootstrap_coeffs = np.array(bootstrap_coeffs)
    
    # Calculate stability metrics
    coeff_stability = pd.DataFrame({
        'CpG_site': cpg_sites,
        'original_coeff': model_package['model'].coef_,
        'mean_bootstrap': np.mean(bootstrap_coeffs, axis=0),
        'std_bootstrap': np.std(bootstrap_coeffs, axis=0),
        'selection_frequency': np.mean(bootstrap_coeffs != 0, axis=0)
    })
    
    # Focus on top 10 most important features
    top_features = coeff_stability.reindex(
        coeff_stability['original_coeff'].abs().sort_values(ascending=False).index
    ).head(10)
    
    print("Top feature stability:")
    for _, row in top_features.iterrows():
        if row['original_coeff'] != 0:
            print(f"  {row['CpG_site'][:50]}...: coeff={row['original_coeff']:.2f}, "
                  f"selected={row['selection_frequency']:.1%}, "
                  f"std={row['std_bootstrap']:.2f}")
    
    return coeff_stability


def hyperparameter_stability_analysis(model_package, cv_predictions):
    """Analyze hyperparameter stability across CV folds"""
    print("\nHyperparameter stability analysis...")
    
    # Load CV predictions to get hyperparameter info
    try:
        # Try to load hyperparameters from CV results if available
        hyperparams_df = pd.read_csv("result/5a_model_hyperparameters.csv")
        optimal_alpha = hyperparams_df['alpha'].iloc[0]
        optimal_l1_ratio = hyperparams_df['l1_ratio'].iloc[0]
        
        print(f"Final model hyperparameters:")
        print(f"  Alpha (regularization strength): {optimal_alpha:.6f}")
        print(f"  L1 ratio (L1 vs L2 balance): {optimal_l1_ratio:.2f}")
        
        # Analyze the distribution of hyperparameters from nested CV if available
        # This would require storing the per-fold hyperparameters during CV
        
    except FileNotFoundError:
        print("Hyperparameter file not found, using model package values")
        optimal_alpha = model_package['alpha']
        optimal_l1_ratio = model_package['l1_ratio']
        
        print(f"Model hyperparameters:")
        print(f"  Alpha (regularization strength): {optimal_alpha:.6f}")
        print(f"  L1 ratio (L1 vs L2 balance): {optimal_l1_ratio:.2f}")
    
    return optimal_alpha, optimal_l1_ratio


def biological_validation(coefficients):
    """Basic biological validation of selected features"""
    print("\nBiological validation...")
    
    # Simple genomic distribution analysis
    chromosomes = []
    for cpg in coefficients['CpG_site']:
        try:
            # Extract chromosome from string format
            chrom = str(cpg).split(',')[0].strip("('") 
            chromosomes.append(chrom)
        except:
            chromosomes.append('unknown')
    
    chrom_dist = pd.Series(chromosomes).value_counts()
    print("Chromosome distribution of selected features:")
    for chrom, count in chrom_dist.head(10).items():
        print(f"  {chrom}: {count} features")
    
    # Coefficient sign analysis
    pos_coeffs = (coefficients['coefficient'] > 0).sum()
    neg_coeffs = (coefficients['coefficient'] < 0).sum()
    print(f"Coefficient signs: {pos_coeffs} positive, {neg_coeffs} negative")
    
    # Model sparsity analysis
    total_features = len(coefficients)
    non_zero_features = (coefficients['coefficient'] != 0).sum()
    sparsity = non_zero_features / total_features * 100
    print(f"Model sparsity: {non_zero_features}/{total_features} features ({sparsity:.1f}%)")
    
    return chrom_dist


def model_comparison(cv_predictions):
    """Compare Elastic Net performance with LASSO if available"""
    print("\nModel comparison analysis...")
    
    # Try to load LASSO results for comparison
    try:
        lasso_predictions = pd.read_csv("result/3_cv_predictions.csv")
        lasso_mae = mean_absolute_error(lasso_predictions['Actual_Age'], lasso_predictions['Predicted_Age'])
        en_mae = mean_absolute_error(cv_predictions['Actual_Age'], cv_predictions['Predicted_Age'])
        
        print(f"LASSO MAE: {lasso_mae:.2f} years")
        print(f"Elastic Net MAE: {en_mae:.2f} years")
        print(f"Improvement: {((lasso_mae - en_mae) / lasso_mae * 100):+.1f}%")
        
        return {'lasso_mae': lasso_mae, 'en_mae': en_mae}
        
    except FileNotFoundError:
        print("LASSO results not found for comparison")
        return None


def main():
    """Main evaluation pipeline for Elastic Net model"""
    print("=" * 70)
    print("Phase 4a: Elastic Net Model Evaluation & Validation")
    print("=" * 70)
    
    # Load data
    model_package, cv_predictions, coefficients = load_model_and_data()
    
    # 1. Residual analysis
    residual_stats, age_corr, group_performance = residual_analysis(cv_predictions)
    
    # 2. Subgroup analysis
    gender_performance = subgroup_analysis(cv_predictions)
    
    # 3. Hyperparameter stability analysis
    optimal_alpha, optimal_l1_ratio = hyperparameter_stability_analysis(model_package, cv_predictions)
    
    # 4. Coefficient stability
    coeff_stability = coefficient_stability(model_package, cv_predictions)
    
    # 5. Biological validation
    chrom_dist = biological_validation(coefficients)
    
    # 6. Model comparison
    comparison_results = model_comparison(cv_predictions)
    
    # Summary metrics
    print(f"\n{'='*70}")
    print("ELASTIC NET EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    en_mae = mean_absolute_error(cv_predictions['Actual_Age'], cv_predictions['Predicted_Age'])
    en_rmse = np.sqrt(mean_squared_error(cv_predictions['Actual_Age'], cv_predictions['Predicted_Age']))
    en_r2 = np.corrcoef(cv_predictions['Actual_Age'], cv_predictions['Predicted_Age'])[0, 1]**2
    
    print(f"Overall MAE: {en_mae:.2f} years")
    print(f"Overall RMSE: {en_rmse:.2f} years")
    print(f"R²: {en_r2:.3f}")
    print(f"Residual normality: p={residual_stats['shapiro_pvalue']:.3f} ({'normal' if residual_stats['shapiro_pvalue'] > 0.05 else 'non-normal'})")
    print(f"Age-residual correlation: {age_corr:.3f} ({'biased' if abs(age_corr) > 0.1 else 'unbiased'})")
    print(f"Top feature selection frequency: {coeff_stability['selection_frequency'].max():.1%}")
    print(f"Chromosome diversity: {len(chrom_dist)} chromosomes represented")
    print(f"Model hyperparameters: alpha={optimal_alpha:.6f}, l1_ratio={optimal_l1_ratio:.2f}")
    
    if comparison_results:
        improvement = (comparison_results['lasso_mae'] - comparison_results['en_mae']) / comparison_results['lasso_mae'] * 100
        print(f"Performance vs LASSO: {improvement:+.1f}% improvement")
    
    # Save results
    residual_stats_df = pd.DataFrame([residual_stats])
    residual_stats_df.to_csv("result/5a_residual_stats.csv", index=False)
    
    group_performance.to_csv("result/5a_age_group_performance.csv")
    if gender_performance is not None:
        gender_performance.to_csv("result/5a_gender_performance.csv")
    
    coeff_stability.to_csv("result/5a_coefficient_stability.csv", index=False)
    chrom_dist.to_csv("result/5a_chromosome_distribution.csv", header=['count'])
    
    # Save summary metrics
    summary_metrics = {
        'MAE': en_mae,
        'RMSE': en_rmse,
        'R2': en_r2,
        'Correlation': np.sqrt(en_r2),
        'Residual_Mean': residual_stats['mean'],
        'Residual_Std': residual_stats['std'],
        'Age_Residual_Correlation': age_corr,
        'Shapiro_PValue': residual_stats['shapiro_pvalue'],
        'Alpha': optimal_alpha,
        'L1_Ratio': optimal_l1_ratio,
        'N_Features': len(coefficients),
        'N_NonZero_Features': (coefficients['coefficient'] != 0).sum(),
        'Sparsity_Percent': (coefficients['coefficient'] != 0).sum() / len(coefficients) * 100
    }
    
    if comparison_results:
        summary_metrics['LASSO_MAE'] = comparison_results['lasso_mae']
        summary_metrics['Improvement_vs_LASSO'] = improvement
    
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv("result/5a_evaluation_summary.csv", index=False)
    
    print(f"\nPhase 4a completed - results saved with 5a_ prefix")


if __name__ == "__main__":
    main()