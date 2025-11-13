#!/usr/bin/env python3
"""
Phase 4: Model Evaluation & Validation
Aging Clock Project - Minimal Output Version

Evaluates model performance, stability, and biological validity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
import os
from scipy import stats


def load_model_and_data(result_dir="result"):
    """Load trained model and data"""
    print("Loading model and data...")
    
    # Load model
    with open(os.path.join(result_dir, "3_trained_model.pkl"), 'rb') as f:
        model_package = pickle.load(f)
    
    # Load predictions from Phase 3
    cv_predictions = pd.read_csv(os.path.join(result_dir, "3_cv_predictions.csv"))
    coefficients = pd.read_csv(os.path.join(result_dir, "3_model_coefficients.csv"))
    
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
    print(f"MAE by age group:\n{group_performance}")
    
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
        print(f"MAE by gender:\n{gender_performance}")
        
    except FileNotFoundError:
        print("Gender metadata not found, skipping gender analysis")
        gender_performance = None
    
    return gender_performance


def coefficient_stability(model_package, cv_predictions, n_bootstrap=100):
    """Assess coefficient stability via bootstrapping"""
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
    
    for i in range(n_bootstrap):
        if i % 20 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")
        
        # Resample with replacement
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Standardize
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        
        # Train LASSO with same alpha as final model
        alpha = 0.1  # Approximate optimal alpha from nested CV
        lasso = Lasso(alpha=alpha, max_iter=1000, random_state=42)
        lasso.fit(X_boot_scaled, y_boot)
        
        bootstrap_coeffs.append(lasso.coef_)
    
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
    
    print(f"Top feature stability:")
    for _, row in top_features.iterrows():
        if row['original_coeff'] != 0:
            print(f"  {row['CpG_site'][:50]}...: coeff={row['original_coeff']:.2f}, "
                  f"selected={row['selection_frequency']:.1%}, "
                  f"std={row['std_bootstrap']:.2f}")
    
    return coeff_stability


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
    print(f"Chromosome distribution of selected features:")
    for chrom, count in chrom_dist.head(10).items():
        print(f"  {chrom}: {count} features")
    
    # Coefficient sign analysis
    pos_coeffs = (coefficients['coefficient'] > 0).sum()
    neg_coeffs = (coefficients['coefficient'] < 0).sum()
    print(f"Coefficient signs: {pos_coeffs} positive, {neg_coeffs} negative")
    
    return chrom_dist


def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("Phase 4: Model Evaluation & Validation")
    print("=" * 60)
    
    # Load data
    model_package, cv_predictions, coefficients = load_model_and_data()
    
    # 1. Residual analysis
    residual_stats, age_corr, group_performance = residual_analysis(cv_predictions)
    
    # 2. Subgroup analysis
    gender_performance = subgroup_analysis(cv_predictions)
    
    # 3. Coefficient stability
    coeff_stability = coefficient_stability(model_package, cv_predictions)
    
    # 4. Biological validation
    chrom_dist = biological_validation(coefficients)
    
    # Summary metrics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Overall MAE: {mean_absolute_error(cv_predictions['Actual_Age'], cv_predictions['Predicted_Age']):.2f} years")
    print(f"Residual normality: p={residual_stats['shapiro_pvalue']:.3f} ({'normal' if residual_stats['shapiro_pvalue'] > 0.05 else 'non-normal'})")
    print(f"Age-residual correlation: {age_corr:.3f} ({'biased' if abs(age_corr) > 0.1 else 'unbiased'})")
    print(f"Top feature selection frequency: {coeff_stability['selection_frequency'].max():.1%}")
    print(f"Chromosome diversity: {len(chrom_dist)} chromosomes represented")
    
    # Save results
    residual_stats_df = pd.DataFrame([residual_stats])
    residual_stats_df.to_csv("result/4_residual_stats.csv", index=False)
    
    group_performance.to_csv("result/4_age_group_performance.csv")
    if gender_performance is not None:
        gender_performance.to_csv("result/4_gender_performance.csv")
    
    coeff_stability.to_csv("result/4_coefficient_stability.csv", index=False)
    chrom_dist.to_csv("result/4_chromosome_distribution.csv", header=['count'])
    
    print(f"\nPhase 4 completed - results saved with 4_ prefix")


if __name__ == "__main__":
    main()