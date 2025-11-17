#!/usr/bin/env python3
"""
Phase 3g: Aging Clock Model Building with LightGBM
Aging Clock Project - LightGBM Implementation

This script builds the aging clock model using LightGBM with nested cross-validation.
LightGBM uses leaf-wise tree growth and gradient-based sampling for efficient learning.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_data(result_dir="result"):
    """
    Load the selected features and preprocessed data
    
    Args:
        result_dir (str): Directory where previous phase results are stored
    
    Returns:
        tuple: Feature matrix X, target vector y, and feature names
    """
    print("Loading data for LightGBM model building...")
    
    # Load selected features from Phase 2
    selected_features_path = os.path.join(result_dir, "2_selected_features.csv")
    selected_features = pd.read_csv(selected_features_path)
    cpg_sites = selected_features['CpG_site'].tolist()
    print(f"Loaded {len(cpg_sites)} selected CpG sites")
    
    # Load preprocessed data
    data_path = os.path.join(result_dir, "1_preprocessed_data.csv")
    data = pd.read_csv(data_path, index_col=0)
    
    # Extract feature matrix X and target vector y
    X = data[cpg_sites].copy()
    y = data['Age'].values
    
    # Check for and handle missing values
    print(f"Missing values per feature: {X.isnull().sum().sum()}")
    if X.isnull().sum().max() > 0:
        print("Imputing missing values with median...")
        X = X.fillna(X.median())
    
    X = X.values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, cpg_sites


def nested_cross_validation(X, y, cpg_sites, data, result_dir="result"):
    """
    Perform nested cross-validation with LOOCV outer loop and 5-fold inner loop
    for LightGBM hyperparameter tuning
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation with LightGBM...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    print("Conservative parameter grid suitable for small dataset")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store model information from each outer fold
    feature_importance_list = []
    
    # === CONSERVATIVE LIGHTGBM PARAMETER GRID ===
    # Conservative settings suitable for small dataset (n=105) with overfitting prevention
    param_grid = [
        # === ULTRA-CONSERVATIVE (minimal complexity) ===
        {'n_estimators': 100, 'num_leaves': 10, 'learning_rate': 0.1, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 15, 'lambda_l1': 1.0, 'lambda_l2': 2.0},
        {'n_estimators': 200, 'num_leaves': 15, 'learning_rate': 0.1, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 20, 'lambda_l1': 2.0, 'lambda_l2': 3.0},
        {'n_estimators': 300, 'num_leaves': 20, 'learning_rate': 0.05, 'feature_fraction': 0.4, 'bagging_fraction': 0.4, 'min_data_in_leaf': 25, 'lambda_l1': 3.0, 'lambda_l2': 5.0},
        
        # === CONSERVATIVE (controlled complexity) ===
        {'n_estimators': 200, 'num_leaves': 25, 'learning_rate': 0.1, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 300, 'num_leaves': 30, 'learning_rate': 0.1, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 15, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 500, 'num_leaves': 35, 'learning_rate': 0.05, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 20, 'lambda_l1': 2.0, 'lambda_l2': 5.0},
        
        # === MODERATE (carefully regularized) ===
        {'n_estimators': 300, 'num_leaves': 40, 'learning_rate': 0.1, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'min_data_in_leaf': 8, 'lambda_l1': 0.2, 'lambda_l2': 1.0},
        {'n_estimators': 500, 'num_leaves': 45, 'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 700, 'num_leaves': 50, 'learning_rate': 0.03, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 12, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        
        # === SLOW LEARNING (very stable) ===
        {'n_estimators': 500, 'num_leaves': 25, 'learning_rate': 0.02, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 1000, 'num_leaves': 30, 'learning_rate': 0.01, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 12, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 1500, 'num_leaves': 35, 'learning_rate': 0.005, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 15, 'lambda_l1': 2.0, 'lambda_l2': 5.0},
        
        # === MAXIMUM REGULARIZATION (extreme prevention) ===
        {'n_estimators': 1000, 'num_leaves': 20, 'learning_rate': 0.01, 'feature_fraction': 0.4, 'bagging_fraction': 0.4, 'min_data_in_leaf': 20, 'lambda_l1': 3.0, 'lambda_l2': 7.0},
        {'n_estimators': 1500, 'num_leaves': 25, 'learning_rate': 0.005, 'feature_fraction': 0.3, 'bagging_fraction': 0.3, 'min_data_in_leaf': 25, 'lambda_l1': 5.0, 'lambda_l2': 10.0},
    ]
    
    print(f"Testing {len(param_grid)} conservative parameter combinations")
    print("Grid covers: tree complexity, regularization, learning rates, sampling")
    print("Overfitting prevention: strong L1/L2, sufficient samples, feature/row sampling")
    
    # Store optimal parameters for analysis
    optimal_params = []
    
    # Outer LOOCV loop with progress bar
    print("\nOuter CV progress:")
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(loo.split(X), total=n_samples, desc="Outer CV Folds")):
        if fold_idx % 20 == 0 and fold_idx > 0:
            print(f"  Completed {fold_idx}/105 outer folds...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Handle missing values in training and test sets
        train_median = np.nanmedian(X_train, axis=0)
        X_train = np.where(np.isnan(X_train), train_median, X_train)
        X_test = np.where(np.isnan(X_test), train_median, X_test)
        
        # Inner 5-fold CV for hyperparameter selection with progress bar
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        best_params = None
        best_score = float('inf')
        
        # Test parameter combinations with progress bar
        param_progress = tqdm(param_grid, desc=f"Fold {fold_idx+1} Param Tuning", leave=False)
        
        for params in param_progress:
            # Update progress description
            param_desc = f"leaves={params['num_leaves']}, lr={params['learning_rate']}, trees={params['n_estimators']}"
            param_progress.set_description(param_desc)
            
            # Perform inner CV for these parameters
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
                # Create LightGBM model with current parameters
                lgb_model = lgb.LGBMRegressor(
                    **params,
                    random_state=42,
                    n_jobs=1,  # Limit cores to prevent memory issues
                    verbosity=-1  # Suppress warnings
                )
                
                # Fit model
                lgb_model.fit(X_inner_train, y_inner_train)
                
                # Predict on validation set
                y_pred_val = lgb_model.predict(X_inner_val)
                val_mae = mean_absolute_error(y_inner_val, y_pred_val)
                inner_scores.append(val_mae)
            
            # Average performance across inner folds
            avg_val_mae = np.mean(inner_scores)
            
            # Select best parameters (prioritize validation performance)
            if avg_val_mae < best_score:
                best_score = avg_val_mae
                best_params = params.copy()
        
        optimal_params.append(best_params)
        
        # Train final LightGBM model with optimal parameters on full training set
        final_lgb = lgb.LGBMRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1  # Use all cores for final model
        )
        final_lgb.fit(X_train, y_train)
        
        # Predict the held-out sample
        pred = final_lgb.predict(X_test)[0]
        predictions[test_idx[0]] = pred
        
        # Store feature importance
        if hasattr(final_lgb, 'feature_importances_'):
            feature_importance_list.append(final_lgb.feature_importances_)
        
        fold_idx += 1
    
    print("Nested cross-validation completed!")
    
    # Calculate performance metrics
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    correlation = np.corrcoef(actual, predictions)[0, 1]
    
    print(f"\nCross-validation performance:")
    print(f"  MAE: {mae:.2f} years")
    print(f"  RMSE: {rmse:.2f} years")
    print(f"  R²: {r2:.3f}")
    print(f"  Correlation: {correlation:.3f}")
    
    # Hyperparameter statistics
    print(f"\nHyperparameter selection summary:")
    n_estimators = [params['n_estimators'] for params in optimal_params]
    num_leaves = [params['num_leaves'] for params in optimal_params]
    learning_rates = [params['learning_rate'] for params in optimal_params]
    feature_fractions = [params['feature_fraction'] for params in optimal_params]
    
    print(f"  Trees range: {min(n_estimators)} - {max(n_estimators)}")
    print(f"  Most common trees: {max(set(n_estimators), key=n_estimators.count)}")
    print(f"  Leaves range: {min(num_leaves)} - {max(num_leaves)}")
    print(f"  Most common leaves: {max(set(num_leaves), key=num_leaves.count)}")
    print(f"  Learning rate range: {min(learning_rates)} - {max(learning_rates)}")
    print(f"  Feature fraction range: {min(feature_fractions)} - {max(feature_fractions)}")
    
    # Feature importance statistics
    if feature_importance_list:
        mean_importance = np.mean(feature_importance_list, axis=0)
        std_importance = np.std(feature_importance_list, axis=0)
        
        # Find most consistently important features
        top_features_idx = np.argsort(mean_importance)[-10:][::-1]
        print(f"\nTop 10 most important CpG sites (by mean importance):")
        for i, idx in enumerate(top_features_idx):
            print(f"  {i+1}. {cpg_sites[idx]}: {mean_importance[idx]:.4f} ± {std_importance[idx]:.4f}")
    
    # Save CV predictions
    cv_results = pd.DataFrame({
        'Sample_ID': data.index,
        'Actual_Age': actual,
        'Predicted_Age': predictions,
        'Age_Acceleration': predictions - actual
    })
    cv_results.to_csv(os.path.join(result_dir, "3g_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'mean_n_estimators': np.mean(n_estimators),
        'mean_num_leaves': np.mean(num_leaves),
        'mean_learning_rate': np.mean(learning_rates),
        'mean_feature_fraction': np.mean(feature_fractions)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "3g_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'optimal_params': optimal_params,
        'feature_importance': feature_importance_list
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final LightGBM model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model and selected features (all features for LightGBM)
    """
    print("\nTraining final LightGBM model on all data...")
    
    # Conservative parameter grid for final model selection
    param_grid = [
        # Conservative options first
        {'n_estimators': 200, 'num_leaves': 25, 'learning_rate': 0.1, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 300, 'num_leaves': 30, 'learning_rate': 0.1, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 15, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 500, 'num_leaves': 35, 'learning_rate': 0.05, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 20, 'lambda_l1': 2.0, 'lambda_l2': 5.0},
        {'n_estimators': 500, 'num_leaves': 40, 'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 700, 'num_leaves': 45, 'learning_rate': 0.03, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 12, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 1000, 'num_leaves': 50, 'learning_rate': 0.02, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 15, 'lambda_l1': 2.0, 'lambda_l2': 5.0},
    ]
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Perform 5-fold CV to select optimal parameters with progress bar
    best_params = None
    best_score = float('inf')
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Final model parameter selection:")
    for params in tqdm(param_grid, desc="Final CV Tuning"):
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create LightGBM model with current parameters
            xgb_model = lgb.LGBMRegressor(
                **params,
                random_state=42,
                n_jobs=1  # Limit cores
            )
            
            # Fit model
            xgb_model.fit(X_train, y_train)
            
            # Predict and calculate MAE
            y_pred = xgb_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
        
        avg_mae = np.mean(cv_scores)
        
        # Select best parameters
        if avg_mae < best_score:
            best_score = avg_mae
            best_params = params.copy()
    
    print(f"Optimal parameters selected: {best_params}")
    print(f"Final model CV MAE: {best_score:.3f} years")
    
    # Train final LightGBM model with optimal parameters
    final_lgb = lgb.LGBMRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1  # Use all cores for final model
    )
    final_lgb.fit(X, y)
    
    # All features are used in LightGBM (no feature selection)
    selected_cpg_sites = cpg_sites.copy()
    
    print(f"Final LightGBM model uses all {len(selected_cpg_sites)} CpG sites")
    print(f"Number of trees: {best_params['n_estimators']}")
    print(f"Number of leaves: {best_params['num_leaves']}")
    print(f"Learning rate: {best_params['learning_rate']}")
    print(f"L1 regularization: {best_params['lambda_l1']}")
    print(f"L2 regularization: {best_params['lambda_l2']}")
    
    # Extract and save feature importance
    if hasattr(final_lgb, 'feature_importances_'):
        importance_scores = final_lgb.feature_importances_
        importance_df = pd.DataFrame({
            'CpG_site': selected_cpg_sites,
            'importance': importance_scores
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(os.path.join(result_dir, "3g_feature_importance.csv"), index=False)
        
        # Save top 10 most important features
        top_10 = importance_df.head(10)
        print(f"\nTop 10 most important CpG sites:")
        for i, (_, row) in enumerate(top_10.iterrows()):
            print(f"  {i+1}. {row['CpG_site']}: {row['importance']:.4f}")
    
    # Save hyperparameters
    hyperparams_df = pd.DataFrame([best_params])
    hyperparams_df.to_csv(os.path.join(result_dir, "3g_model_hyperparameters.csv"), index=False)
    
    # Save the trained model
    model_package = {
        'model': final_lgb,
        'cpg_sites': cpg_sites,
        'optimal_params': best_params,
        'cv_mae': best_score
    }
    
    with open(os.path.join(result_dir, "3g_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"LightGBM model saved with {best_params['n_estimators']} trees")
    print(f"Feature importance rankings saved")
    
    return final_lgb, selected_cpg_sites


def plot_predicted_vs_actual(result_dir="result", figure_dir="figure"):
    """
    Create predicted vs actual age plot using LightGBM model results
    
    Args:
        result_dir (str): Directory where LightGBM model results are stored
        figure_dir (str): Directory to save the plot
    """
    print("Creating LightGBM predicted vs actual plot...")
    
    # Load CV predictions
    predictions_path = os.path.join(result_dir, "3g_cv_predictions.csv")
    df = pd.read_csv(predictions_path)
    
    # Load metrics
    metrics_path = os.path.join(result_dir, "3g_cv_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    metrics = metrics_df.iloc[0]
    
    # Load hyperparameters to get model details
    hyperparams_path = os.path.join(result_dir, "3g_model_hyperparameters.csv")
    hyperparams_df = pd.read_csv(hyperparams_path)
    n_estimators = hyperparams_df['n_estimators'].iloc[0]
    num_leaves = hyperparams_df['num_leaves'].iloc[0]
    
    # Set style (same as visualization scripts)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with exact same settings as visualization scripts
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
    
    # Add metrics text box (exact same format as visualization scripts)
    metrics_text = f"""MAE: {metrics['MAE']:.2f} years
RMSE: {metrics['RMSE']:.2f} years
R²: {metrics['R2']:.3f}
Correlation: {metrics['Correlation']:.3f}"""
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Actual Age (years)', fontsize=12)
    ax.set_ylabel('Predicted Age (years)', fontsize=12)
    ax.set_title(f'LightGBM Aging Clock: Predicted vs Actual Age\n' + 
                f'({len(df)} samples, {metrics["n_features"]} CpG sites, {n_estimators} trees, {num_leaves} leaves)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)
    
    # Save plot with LightGBM-specific filename
    output_path = os.path.join(figure_dir, "3g_predicted_vs_actual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"LightGBM predicted vs actual plot saved to: {output_path}")
    
    return fig


def estimate_runtime():
    """
    Estimate runtime for the LightGBM script
    """
    print("\n" + "="*60)
    print("RUNTIME ESTIMATION FOR LIGHTGBM SCRIPT")
    print("="*60)
    
    # Dataset characteristics
    n_samples = 105
    n_features = 86
    n_param_combinations = 15  # Our conservative grid
    n_inner_cv_folds = 5
    n_outer_loocv_folds = 105
    
    # Estimated times per operation (LightGBM is typically faster than XGBoost)
    time_per_lgb_fit = 0.02  # seconds for one conservative LightGBM fit (very fast)
    time_per_outer_fold = n_param_combinations * n_inner_cv_folds * time_per_lgb_fit
    total_nested_cv_time = n_outer_loocv_folds * time_per_outer_fold
    
    # Final model training (much faster)
    final_model_time = n_param_combinations * n_inner_cv_folds * time_per_lgb_fit * 0.2  # ~80% faster
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Parameter grid: {n_param_combinations} combinations (conservative)")
    print(f"Cross-validation: {n_outer_loocv_folds} outer folds × {n_inner_cv_folds} inner folds")
    print()
    print(f"Estimated time per outer fold: {time_per_outer_fold:.1f} seconds")
    print(f"Total nested CV time: {total_nested_cv_time/60:.1f} minutes")
    print(f"Final model selection: {final_model_time/60:.1f} minutes")
    print(f"Total estimated runtime: {(total_nested_cv_time + final_model_time)/60:.1f} minutes")
    print(f"                           ~{((total_nested_cv_time + final_model_time)/60/60):.1f} hours")
    print()
    print("LightGBM advantages:")
    print("- Leaf-wise growth often achieves better accuracy with fewer trees")
    print("- GOSS sampling reduces computation while maintaining accuracy")
    print("- EFB bundling reduces dimensionality for high-dimensional data")
    print("- Progress bars will show real-time status")
    print("="*60)


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("Phase 3g: Aging Clock Model Building with LightGBM")
    print("=" * 80)
    print("Using LightGBM with conservative parameter grid to prevent overfitting")
    print("Features: Leaf-wise growth, GOSS sampling, EFB bundling, missing value handling")
    print("Optimizations: Conservative complexity, strong regularization, leaf-wise learning")
    print()
    
    # Estimate runtime
    estimate_runtime()
    
    # Record start time
    start_time = time.time()
    
    # Load data
    print("\nStep 1: Loading data...")
    X, y, cpg_sites = load_data()
    data = pd.read_csv("result/1_preprocessed_data.csv", index_col=0)
    
    # Perform nested cross-validation
    print("\nStep 2: Starting nested cross-validation (this will take time)...")
    cv_results = nested_cross_validation(X, y, cpg_sites, data)
    
    # Train final model
    print("\nStep 3: Training final model...")
    final_model, selected_features = train_final_model(X, y, cpg_sites)
    
    # Generate predicted vs actual plot
    print("\nStep 4: Generating visualization...")
    plot_predicted_vs_actual()
    
    # Calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n" + "="*60)
    print(f"Phase 3g completed successfully!")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final LightGBM model uses {final_model.n_estimators} trees")
    print(f"- Model saved to 'result' directory with '3g_' prefix")
    print(f"- Feature importance rankings saved")
    print(f"- Predicted vs actual plot saved to 'figure' directory")
    print(f"="*60)


if __name__ == "__main__":
    main()