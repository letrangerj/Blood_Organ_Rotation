#!/usr/bin/env python3
"""
Phase 3d: Aging Clock Model Building with SVR (Support Vector Regression)
Aging Clock Project - SVR Implementation

This script builds the aging clock model using SVR with nested cross-validation.
SVR can handle both linear and nonlinear relationships through kernel selection.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(result_dir="result"):
    """
    Load the selected features and preprocessed data
    
    Args:
        result_dir (str): Directory where previous phase results are stored
    
    Returns:
        tuple: Feature matrix X, target vector y, and feature names
    """
    print("Loading data for SVR model building...")
    
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
    for SVR hyperparameter tuning
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation with SVR...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    print("Tuning parameters: kernel, C, epsilon, gamma (for nonlinear kernels)")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store support vectors count from each outer fold
    support_vectors_list = []
    
    # Define hyperparameter grids for different kernels
    param_grids = {
        'linear': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5, 1.0]
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'gamma': ['scale', 0.001, 0.01, 0.1]
        }
    }
    
    # Store optimal parameters for analysis
    optimal_params = []
    
    # Outer LOOCV loop
    fold_idx = 0
    for train_idx, test_idx in loo.split(X):
        if fold_idx % 20 == 0:
            print(f"  Processing outer fold {fold_idx + 1}/105...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Handle missing values in training and test sets
        # Impute with median of training data
        train_median = np.nanmedian(X_train, axis=0)
        
        # Replace NaN values
        X_train = np.where(np.isnan(X_train), train_median, X_train)
        X_test = np.where(np.isnan(X_test), train_median, X_test)
        
        # Standardize features based on training data (CRITICAL for SVR!)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inner 5-fold CV for hyperparameter selection
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Test different parameter combinations
        best_params = None
        best_score = float('inf')
        
        # Test linear kernel first (simpler, more interpretable)
        for kernel in ['linear', 'rbf']:
            param_grid = param_grids[kernel]
            
            # Generate all parameter combinations for this kernel
            if kernel == 'linear':
                param_combinations = [
                    {'kernel': 'linear', 'C': C, 'epsilon': eps}
                    for C in param_grid['C']
                    for eps in param_grid['epsilon']
                ]
            else:  # rbf
                param_combinations = [
                    {'kernel': 'rbf', 'C': C, 'epsilon': eps, 'gamma': gamma}
                    for C in param_grid['C']
                    for eps in param_grid['epsilon']
                    for gamma in param_grid['gamma']
                ]
            
            for params in param_combinations:
                # Perform inner CV for these parameters
                inner_scores = []
                
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled):
                    X_inner_train, X_inner_val = X_train_scaled[inner_train_idx], X_train_scaled[inner_val_idx]
                    y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                    
                    # Fit SVR
                    svr = SVR(**params)
                    svr.fit(X_inner_train, y_inner_train)
                    
                    # Predict and calculate MAE
                    y_pred = svr.predict(X_inner_val)
                    mae = mean_absolute_error(y_inner_val, y_pred)
                    inner_scores.append(mae)
                
                # Average MAE across inner folds
                avg_mae = np.mean(inner_scores)
                
                if avg_mae < best_score:
                    best_score = avg_mae
                    best_params = params.copy()
        
        optimal_params.append(best_params)
        
        # Train final SVR model with optimal parameters on full training set
        final_svr = SVR(**best_params)
        final_svr.fit(X_train_scaled, y_train)
        
        # Predict the held-out sample
        pred = final_svr.predict(X_test_scaled)[0]
        predictions[test_idx[0]] = pred
        
        # Store support vector count
        support_vectors_list.append(len(final_svr.support_))
        
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
    kernels_used = [params['kernel'] for params in optimal_params]
    print(f"  Kernels used: {dict(zip(*np.unique(kernels_used, return_counts=True)))}")
    print(f"  Most common kernel: {max(set(kernels_used), key=kernels_used.count)}")
    
    # C parameter statistics
    c_values = [params['C'] for params in optimal_params]
    print(f"  C range: {min(c_values)} - {max(c_values)}")
    print(f"  Mean C: {np.mean(c_values):.2f}")
    
    # Epsilon statistics
    eps_values = [params['epsilon'] for params in optimal_params]
    print(f"  Epsilon range: {min(eps_values)} - {max(eps_values)}")
    print(f"  Mean epsilon: {np.mean(eps_values):.3f}")
    
    # Support vectors statistics
    print(f"  Support vectors range: {min(support_vectors_list)} - {max(support_vectors_list)}")
    print(f"  Mean support vectors: {np.mean(support_vectors_list):.1f}")
    print(f"  Support vectors as % of training: {np.mean(support_vectors_list)/104*100:.1f}%")
    
    # Save CV predictions
    cv_results = pd.DataFrame({
        'Sample_ID': data.index,
        'Actual_Age': actual,
        'Predicted_Age': predictions,
        'Age_Acceleration': predictions - actual
    })
    cv_results.to_csv(os.path.join(result_dir, "3d_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'most_common_kernel': max(set(kernels_used), key=kernels_used.count),
        'mean_C': np.mean(c_values),
        'mean_epsilon': np.mean(eps_values),
        'mean_support_vectors': np.mean(support_vectors_list)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "3d_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'optimal_params': optimal_params,
        'support_vectors': support_vectors_list
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final SVR model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model, scaler, and selected features
    """
    print("\nTraining final SVR model on all data...")
    
    # Define hyperparameter grids
    param_grids = {
        'linear': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5, 1.0]
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'gamma': ['scale', 0.001, 0.01, 0.1]
        }
    }
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Standardize features (CRITICAL for SVR!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform 5-fold CV to select optimal parameters
    best_params = None
    best_score = float('inf')
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test both kernels
    for kernel in ['linear', 'rbf']:
        param_grid = param_grids[kernel]
        
        if kernel == 'linear':
            param_combinations = [
                {'kernel': 'linear', 'C': C, 'epsilon': eps}
                for C in param_grid['C']
                for eps in param_grid['epsilon']
            ]
        else:  # rbf
            param_combinations = [
                {'kernel': 'rbf', 'C': C, 'epsilon': eps, 'gamma': gamma}
                for C in param_grid['C']
                for eps in param_grid['epsilon']
                for gamma in param_grid['gamma']
            ]
        
        for params in param_combinations:
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit SVR
                svr = SVR(**params)
                svr.fit(X_train, y_train)
                
                # Predict and calculate MAE
                y_pred = svr.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
            
            avg_mae = np.mean(cv_scores)
            
            if avg_mae < best_score:
                best_score = avg_mae
                best_params = params.copy()
    
    print(f"Optimal parameters selected: {best_params}")
    
    # Train final SVR model with optimal parameters
    final_svr = SVR(**best_params)
    final_svr.fit(X_scaled, y)
    
    # All features are used in SVR (no feature selection)
    selected_cpg_sites = cpg_sites.copy()
    
    # For linear kernel, we can extract coefficients
    if best_params['kernel'] == 'linear':
        selected_coefficients = final_svr.coef_.flatten()
        print(f"Final SVR model uses all {len(selected_cpg_sites)} CpG sites (linear kernel)")
    else:
        selected_coefficients = None
        print(f"Final SVR model uses all {len(selected_cpg_sites)} CpG sites ({best_params['kernel']} kernel)")
        print(f"Number of support vectors: {len(final_svr.support_)}")
    
    # Save model coefficients (if linear)
    if selected_coefficients is not None:
        coefficients_df = pd.DataFrame({
            'CpG_site': selected_cpg_sites,
            'coefficient': selected_coefficients
        })
        coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
        coefficients_df.to_csv(os.path.join(result_dir, "3d_model_coefficients.csv"), index=False)
    
    # Save intercept (for linear kernel)
    if best_params['kernel'] == 'linear':
        intercept_df = pd.DataFrame({
            'intercept': [final_svr.intercept_[0]]
        })
        intercept_df.to_csv(os.path.join(result_dir, "3d_model_intercept.csv"), index=False)
    
    # Save hyperparameters
    hyperparams_df = pd.DataFrame([best_params])
    hyperparams_df.to_csv(os.path.join(result_dir, "3d_model_hyperparameters.csv"), index=False)
    
    # Save support vectors info (for nonlinear kernels)
    if best_params['kernel'] != 'linear':
        sv_info = pd.DataFrame({
            'n_support_vectors': [len(final_svr.support_)],
            'support_vector_ratio': [len(final_svr.support_) / len(y)]
        })
        sv_info.to_csv(os.path.join(result_dir, "3d_support_vectors_info.csv"), index=False)
    
    # Save the trained model and scaler
    model_package = {
        'model': final_svr,
        'scaler': scaler,
        'cpg_sites': cpg_sites,
        'optimal_params': best_params
    }
    
    with open(os.path.join(result_dir, "3d_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"SVR model saved with {best_params['kernel']} kernel")
    print(f"Regularization: C={best_params['C']}, epsilon={best_params['epsilon']}")
    if best_params['kernel'] == 'rbf':
        print(f"RBF gamma: {best_params['gamma']}")
    
    return final_svr, scaler, selected_cpg_sites


def plot_predicted_vs_actual(result_dir="result", figure_dir="figure"):
    """
    Create predicted vs actual age plot using SVR model results
    
    Args:
        result_dir (str): Directory where SVR model results are stored
        figure_dir (str): Directory to save the plot
    """
    print("Creating SVR predicted vs actual plot...")
    
    # Load CV predictions
    predictions_path = os.path.join(result_dir, "3d_cv_predictions.csv")
    df = pd.read_csv(predictions_path)
    
    # Load metrics
    metrics_path = os.path.join(result_dir, "3d_cv_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    metrics = metrics_df.iloc[0]
    
    # Load hyperparameters to get kernel type
    hyperparams_path = os.path.join(result_dir, "3d_model_hyperparameters.csv")
    hyperparams_df = pd.read_csv(hyperparams_path)
    kernel = hyperparams_df['kernel'].iloc[0]
    
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
    ax.set_title(f'SVR Aging Clock: Predicted vs Actual Age\n' + 
                f'({len(df)} samples, {metrics["n_features"]} CpG sites, {kernel} kernel)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)
    
    # Save plot with SVR-specific filename
    output_path = os.path.join(figure_dir, "3d_predicted_vs_actual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"SVR predicted vs actual plot saved to: {output_path}")
    
    return fig


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("Phase 3d: Aging Clock Model Building with SVR")
    print("=" * 80)
    print("Using Support Vector Regression with kernel selection")
    print("Kernels: linear (interpretable) and rbf (nonlinear patterns)")
    print("Standardization: CRITICAL for SVR performance")
    print()
    
    # Load data
    X, y, cpg_sites = load_data()
    data = pd.read_csv("result/1_preprocessed_data.csv", index_col=0)
    
    # Perform nested cross-validation
    cv_results = nested_cross_validation(X, y, cpg_sites, data)
    
    # Train final model
    final_model, scaler, selected_features = train_final_model(X, y, cpg_sites)
    
    # Generate predicted vs actual plot
    plot_predicted_vs_actual()
    
    print(f"\nPhase 3d completed successfully!")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final SVR model uses all {len(selected_features)} CpG sites")
    print(f"- Optimal kernel: {cv_results['metrics']['most_common_kernel']}")
    print(f"- Model saved to 'result' directory with '3d_' prefix")
    print(f"- Predicted vs actual plot saved to 'figure' directory")


if __name__ == "__main__":
    main()