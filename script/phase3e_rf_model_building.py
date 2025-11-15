#!/usr/bin/env python3
"""
Phase 3e: Aging Clock Model Building with Random Forest
Aging Clock Project - Conservative RF Implementation

This script builds the aging clock model using Random Forest with nested cross-validation.
Uses conservative parameters to prevent overfitting on small dataset (n=105).
Includes tqdm progress tracking for long-running operations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    print("Loading data for Random Forest model building...")
    
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
    if X.isnull().sum().sum() > 0:
        print("Imputing missing values with median...")
        X = X.fillna(X.median())
    
    X = X.values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, cpg_sites


def nested_cross_validation(X, y, cpg_sites, data, result_dir="result"):
    """
    Perform nested cross-validation with LOOCV outer loop and 5-fold inner loop
    for Random Forest hyperparameter tuning
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation with Random Forest...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    print("Conservative parameters to prevent overfitting on small dataset")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store model information from each outer fold
    feature_importance_list = []
    oob_scores_list = []
    
    # === CONSERVATIVE PARAMETER GRID ===
    # Designed for small dataset (n=105) to prevent overfitting
    param_grid = [
        # Conservative settings first
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 'sqrt'},
        {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 0.5},
        {'n_estimators': 300, 'max_depth': 7, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 0.5},
        
        # Moderate complexity
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 0.5},
        
        # Slightly more complex (but still conservative)
        {'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 700, 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 0.7},
    ]
    
    print(f"Testing {len(param_grid)} conservative parameter combinations")
    print("Parameters focus on: limited depth, sufficient samples per split/leaf")
    
    # Store optimal parameters for analysis
    optimal_params = []
    
    # Outer LOOCV loop with progress bar
    print("\nOuter CV progress:")
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(loo.split(X), total=n_samples, desc="Outer CV Folds")):
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
        best_oob_score = -float('inf')
        
        # Test parameter combinations with progress bar
        param_progress = tqdm(param_grid, desc=f"Fold {fold_idx+1} Param Tuning", leave=False)
        
        for params in param_progress:
            # Update progress description
            param_progress.set_description(f"Testing: depth={params['max_depth']}, trees={params['n_estimators']}")
            
            # Perform inner CV for these parameters
            inner_scores = []
            inner_oob_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
                # Fit Random Forest with OOB if available
                rf = RandomForestRegressor(
                    **params,
                    oob_score=True,  # Enable OOB for additional validation
                    random_state=42,
                    n_jobs=2  # Limit cores to prevent memory issues
                )
                rf.fit(X_inner_train, y_inner_train)
                
                # Predict and calculate MAE
                y_pred = rf.predict(X_inner_val)
                mae = mean_absolute_error(y_inner_val, y_pred)
                inner_scores.append(mae)
                
                # Store OOB score if available
                if hasattr(rf, 'oob_score_'):
                    inner_oob_scores.append(rf.oob_score_)
            
            # Average performance across inner folds
            avg_mae = np.mean(inner_scores)
            avg_oob = np.mean(inner_oob_scores) if inner_oob_scores else 0
            
            # Select best parameters (prioritize CV performance, use OOB as tiebreaker)
            if avg_mae < best_score:
                best_score = avg_mae
                best_params = params.copy()
                best_oob_score = avg_oob
            elif avg_mae == best_score and avg_oob > best_oob_score:
                # If CV performance is equal, prefer better OOB score
                best_params = params.copy()
                best_oob_score = avg_oob
        
        optimal_params.append(best_params)
        
        # Train final RF model with optimal parameters on full training set
        final_rf = RandomForestRegressor(
            **best_params,
            oob_score=True,
            random_state=42,
            n_jobs=2  # Limit cores
        )
        final_rf.fit(X_train, y_train)
        
        # Predict the held-out sample
        pred = final_rf.predict(X_test)[0]
        predictions[test_idx[0]] = pred
        
        # Store feature importance and OOB score
        feature_importance_list.append(final_rf.feature_importances_)
        if hasattr(final_rf, 'oob_score_'):
            oob_scores_list.append(final_rf.oob_score_)
    
    print("Nested cross-validation completed!")
    
    # Calculate performance metrics
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    correlation = np.corrcoef(actual, predictions)[0, 1]

    print("\nCross-validation performance:")
    print(f"  MAE: {mae:.2f} years")
    print(f"  RMSE: {rmse:.2f} years")
    print(f"  R²: {r2:.3f}")
    print(f"  Correlation: {correlation:.3f}")
    
    # Hyperparameter statistics
    print("\nHyperparameter selection summary:")
    n_estimators = [params['n_estimators'] for params in optimal_params]
    max_depths = [params['max_depth'] for params in optimal_params]
    min_samples = [params['min_samples_split'] for params in optimal_params]
    
    print(f"  Trees range: {min(n_estimators)} - {max(n_estimators)}")
    print(f"  Most common trees: {max(set(n_estimators), key=n_estimators.count)}")
    print(f"  Depth range: {min(max_depths)} - {max([d for d in max_depths if d is not None], default=0)}")
    print(f"  Most common depth: {max(set(max_depths), key=max_depths.count)}")
    print(f"  Min samples split range: {min(min_samples)} - {max(min_samples)}")
    
    # OOB statistics
    if oob_scores_list:
        print(f"  OOB R² range: {min(oob_scores_list):.3f} - {max(oob_scores_list):.3f}")
        print(f"  Mean OOB R²: {np.mean(oob_scores_list):.3f}")
    
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
    cv_results.to_csv(os.path.join(result_dir, "3e_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'mean_n_estimators': np.mean(n_estimators),
        'mean_max_depth': np.mean([d if d is not None else 0 for d in max_depths]),
        'mean_min_samples_split': np.mean(min_samples),
        'mean_oob_r2': np.mean(oob_scores_list) if oob_scores_list else 0
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "3e_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'optimal_params': optimal_params,
        'feature_importance': feature_importance_list,
        'oob_scores': oob_scores_list
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final Random Forest model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model and selected features (all features for RF)
    """
    print("\nTraining final Random Forest model on all data...")
    
    # Conservative parameter grid for final model selection
    param_grid = [
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 300, 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 0.5},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 700, 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 0.7},
    ]
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Perform 5-fold CV to select optimal parameters with progress bar
    best_params = None
    best_score = float('inf')
    best_oob_score = -float('inf')
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Final model parameter selection:")
    for params in tqdm(param_grid, desc="Final CV Tuning"):
        cv_scores = []
        cv_oob_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit Random Forest
            rf = RandomForestRegressor(
                **params,
                oob_score=True,
                random_state=42,
                n_jobs=2  # Limit cores
            )
            rf.fit(X_train, y_train)
            
            # Predict and calculate MAE
            y_pred = rf.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
            
            # Store OOB score
            if hasattr(rf, 'oob_score_'):
                cv_oob_scores.append(rf.oob_score_)
        
        avg_mae = np.mean(cv_scores)
        avg_oob = np.mean(cv_oob_scores) if cv_oob_scores else 0
        
        # Select best parameters
        if avg_mae < best_score:
            best_score = avg_mae
            best_params = params.copy()
            best_oob_score = avg_oob
        elif avg_mae == best_score and avg_oob > best_oob_score:
            best_params = params.copy()
            best_oob_score = avg_oob
    
    print(f"Optimal parameters selected: {best_params}")
    print(f"Final model CV MAE: {best_score:.3f} years")
    print(f"Final model OOB R²: {best_oob_score:.3f}")
    
    # Train final Random Forest model with optimal parameters
    final_rf = RandomForestRegressor(
        **best_params,
        oob_score=True,
        random_state=42,
        n_jobs=-1  # Use all cores for final model
    )
    final_rf.fit(X, y)
    
    # All features are used in Random Forest (no feature selection)
    selected_cpg_sites = cpg_sites.copy()
    
    print(f"Final RF model uses all {len(selected_cpg_sites)} CpG sites")
    print(f"Number of trees: {best_params['n_estimators']}")
    print(f"Tree depth: {best_params['max_depth']}")
    print(f"Min samples per split: {best_params['min_samples_split']}")
    print(f"Min samples per leaf: {best_params['min_samples_leaf']}")
    print(f"OOB R² score: {final_rf.oob_score_:.3f}")
    
    # Extract and save feature importance
    importance_scores = final_rf.feature_importances_
    importance_df = pd.DataFrame({
        'CpG_site': selected_cpg_sites,
        'importance': importance_scores
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(result_dir, "3e_feature_importance.csv"), index=False)
    
    # Save top 10 most important features
    top_10 = importance_df.head(10)
    print(f"\nTop 10 most important CpG sites:")
    for i, (_, row) in enumerate(top_10.iterrows()):
        print(f"  {i+1}. {row['CpG_site']}: {row['importance']:.4f}")
    
    # Save hyperparameters
    hyperparams_df = pd.DataFrame([best_params])
    hyperparams_df.to_csv(os.path.join(result_dir, "3e_model_hyperparameters.csv"), index=False)
    
    # Save OOB score
    oob_df = pd.DataFrame({
        'oob_score': [final_rf.oob_score_],
        'oob_score_r2': [final_rf.oob_score_]
    })
    oob_df.to_csv(os.path.join(result_dir, "3e_oob_score.csv"), index=False)
    
    # Save the trained model
    model_package = {
        'model': final_rf,
        'cpg_sites': cpg_sites,
        'optimal_params': best_params,
        'oob_score': final_rf.oob_score_,
        'feature_importance': importance_scores
    }
    
    with open(os.path.join(result_dir, "3e_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Random Forest model saved with {best_params['n_estimators']} trees")
    print(f"Feature importance rankings saved")
    
    return final_rf, selected_cpg_sites


def plot_predicted_vs_actual(result_dir="result", figure_dir="figure"):
    """
    Create predicted vs actual age plot using Random Forest model results
    
    Args:
        result_dir (str): Directory where RF model results are stored
        figure_dir (str): Directory to save the plot
    """
    print("Creating Random Forest predicted vs actual plot...")
    
    # Load CV predictions
    predictions_path = os.path.join(result_dir, "3e_cv_predictions.csv")
    df = pd.read_csv(predictions_path)
    
    # Load metrics
    metrics_path = os.path.join(result_dir, "3e_cv_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    metrics = metrics_df.iloc[0]
    
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
    ax.set_title('Random Forest Aging Clock: Predicted vs Actual Age\n' + 
                f'({len(df)} samples, {metrics["n_features"]} CpG sites, {int(metrics["mean_n_estimators"])} trees)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)
    
    # Save plot with RF-specific filename
    output_path = os.path.join(figure_dir, "3e_predicted_vs_actual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Random Forest predicted vs actual plot saved to: {output_path}")
    
    return fig


def estimate_runtime():
    """
    Estimate runtime for the Random Forest script
    """
    print("\n" + "="*60)
    print("RUNTIME ESTIMATION FOR RANDOM FOREST SCRIPT")
    print("="*60)
    
    # Dataset characteristics
    n_samples = 105
    n_features = 86
    n_param_combinations = 9  # Our conservative grid
    n_inner_cv_folds = 5
    n_outer_loocv_folds = 105
    
    # Estimated times per operation (based on typical performance)
    time_per_rf_fit = 0.1  # seconds for one RF fit with conservative params
    time_per_outer_fold = n_param_combinations * n_inner_cv_folds * time_per_rf_fit
    total_nested_cv_time = n_outer_loocv_folds * time_per_outer_fold
    
    # Final model training (much faster, uses optimal params)
    final_model_time = n_param_combinations * n_inner_cv_folds * time_per_rf_fit * 0.5  # ~50% faster
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Parameter grid: {n_param_combinations} combinations")
    print(f"Cross-validation: {n_outer_loocv_folds} outer folds × {n_inner_cv_folds} inner folds")
    print()
    print(f"Estimated time per outer fold: {time_per_outer_fold:.1f} seconds")
    print(f"Total nested CV time: {total_nested_cv_time/60:.1f} minutes")
    print(f"Final model selection: {final_model_time/60:.1f} minutes")
    print(f"Total estimated runtime: {(total_nested_cv_time + final_model_time)/60:.1f} minutes")
    print(f"                           ~{((total_nested_cv_time + final_model_time)/60/60):.1f} hours")
    print()
    print("Optimization notes:")
    print("- Using conservative parameters (fewer trees, limited depth)")
    print("- Limited to 2 CPU cores for CV to prevent memory issues")
    print("- Progress bars will show real-time status")
    print("- Consider running overnight for full results")
    print("="*60)


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("Phase 3e: Aging Clock Model Building with Random Forest")
    print("=" * 80)
    print("Using conservative Random Forest to prevent overfitting")
    print("Features: Progress tracking, OOB validation, feature importance")
    print("Optimizations: Limited trees, controlled depth, sufficient samples")
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
    print(f"Phase 3e completed successfully!")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final RF model uses {final_model.n_estimators} trees")
    print(f"- Model saved to 'result' directory with '3e_' prefix")
    print(f"- Feature importance rankings saved")
    print(f"- Predicted vs actual plot saved to 'figure' directory")
    print(f"="*60)


if __name__ == "__main__":
    main()
