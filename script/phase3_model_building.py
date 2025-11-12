#!/usr/bin/env python3
"""
Phase 3: Aging Clock Model Building
Aging Clock Project

This script builds the aging clock model using LASSO regression with nested cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
import os
import pickle


def load_data(result_dir="result"):
    """
    Load the selected features and preprocessed data
    
    Args:
        result_dir (str): Directory where previous phase results are stored
    
    Returns:
        tuple: Feature matrix X, target vector y, and feature names
    """
    print("Loading data for model building...")
    
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
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store model coefficients from each outer fold
    coefficients_list = []
    
    # Define hyperparameter grid for LASSO
    alphas = np.logspace(-4, 1, 50)  # 10^-4 to 10^1
    
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
        
        # Standardize features based on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inner 5-fold CV for hyperparameter selection
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Fit LASSO with cross-validation
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=inner_cv,
            max_iter=50000,  # Increased from 10000
            tol=1e-3,  # Added: more lenient convergence tolerance
            n_jobs=-1,
            random_state=42
        )
        lasso_cv.fit(X_train_scaled, y_train)
        
        # Get optimal alpha (using 1-SE rule for sparser model)
        optimal_alpha = lasso_cv.alpha_
        mse_path = lasso_cv.mse_path_.mean(axis=-1)
        std_mse_path = lasso_cv.mse_path_.std(axis=-1)
        
        # Find the sparsest model within 1 SE of the best
        min_mse_idx = np.argmin(mse_path)
        min_mse = mse_path[min_mse_idx]
        se_threshold = min_mse + std_mse_path[min_mse_idx] / np.sqrt(5)
        
        # Find the largest alpha (most sparse) that meets the threshold
        candidates = np.where(mse_path <= se_threshold)[0]
        if len(candidates) > 0:
            selected_idx = candidates[-1]  # Last index = largest alpha
            optimal_alpha = lasso_cv.alphas_[selected_idx]
        
            # Retrain with selected alpha
        final_lasso = LassoCV(
            alphas=[optimal_alpha],
            cv=inner_cv,
            max_iter=50000,  # Increased from 10000
            tol=1e-3,  # Added: more lenient convergence tolerance
            n_jobs=-1,
            random_state=42
        )
        final_lasso.fit(X_train_scaled, y_train)
        
        # Predict the held-out sample
        pred = final_lasso.predict(X_test_scaled)[0]
        predictions[test_idx[0]] = pred
        
        # Store coefficients
        coefficients_list.append(final_lasso.coef_)
        
        fold_idx += 1
    
    print("Nested cross-validation completed!")
    
    # Calculate performance metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    correlation = np.corrcoef(actual, predictions)[0, 1]
    
    print(f"\nCross-validation performance:")
    print(f"  MAE: {mae:.2f} years")
    print(f"  RMSE: {rmse:.2f} years")
    print(f"  R²: {r2:.3f}")
    print(f"  Correlation: {correlation:.3f}")
    
    # Save CV predictions
    cv_results = pd.DataFrame({
        'Sample_ID': data.index,
        'Actual_Age': actual,
        'Predicted_Age': predictions,
        'Age_Acceleration': predictions - actual
    })
    cv_results.to_csv(os.path.join(result_dir, "3_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1]
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "3_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'coefficients': coefficients_list
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model, scaler, and selected features
    """
    print("\nTraining final model on all data...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define hyperparameter grid
    alphas = np.logspace(-4, 1, 50)
    
    # Perform 5-fold CV to select optimal alpha
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=5,
        max_iter=10000,
        n_jobs=-1,
        random_state=42
    )
    lasso_cv.fit(X_scaled, y)
    
    # Apply 1-SE rule for sparser model
    optimal_alpha = lasso_cv.alpha_
    mse_path = lasso_cv.mse_path_.mean(axis=-1)
    std_mse_path = lasso_cv.mse_path_.std(axis=-1)
    
    min_mse_idx = np.argmin(mse_path)
    min_mse = mse_path[min_mse_idx]
    se_threshold = min_mse + std_mse_path[min_mse_idx] / np.sqrt(5)
    
    candidates = np.where(mse_path <= se_threshold)[0]
    if len(candidates) > 0:
        selected_idx = candidates[-1]
        optimal_alpha = lasso_cv.alphas_[selected_idx]
    
    print(f"Optimal alpha selected: {optimal_alpha:.6f}")
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train final model with selected alpha
    final_model = LassoCV(
        alphas=[optimal_alpha],
        cv=5,
        max_iter=50000,  # Increased from 10000
        tol=1e-3,  # Added: more lenient convergence tolerance
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X_scaled, y)
    
    # Extract non-zero coefficients
    non_zero_idx = np.where(final_model.coef_ != 0)[0]
    selected_cpg_sites = [cpg_sites[i] for i in non_zero_idx]
    selected_coefficients = final_model.coef_[non_zero_idx]
    
    print(f"Final model selected {len(selected_cpg_sites)} CpG sites out of {len(cpg_sites)}")
    
    # Save model coefficients
    coefficients_df = pd.DataFrame({
        'CpG_site': selected_cpg_sites,
        'coefficient': selected_coefficients
    })
    coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
    coefficients_df.to_csv(os.path.join(result_dir, "3_model_coefficients.csv"), index=False)
    
    # Save intercept
    intercept_df = pd.DataFrame({
        'intercept': [final_model.intercept_]
    })
    intercept_df.to_csv(os.path.join(result_dir, "3_model_intercept.csv"), index=False)
    
    # Save the trained model and scaler
    model_package = {
        'model': final_model,
        'scaler': scaler,
        'cpg_sites': cpg_sites
    }
    
    with open(os.path.join(result_dir, "3_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved with {len(selected_cpg_sites)} features")
    
    return final_model, scaler, selected_cpg_sites


def main():
    """
    Main execution function
    """
    print("Starting Phase 3: Aging Clock Model Building\n")
    
    # Load data
    X, y, cpg_sites = load_data()
    data = pd.read_csv("result/1_preprocessed_data.csv", index_col=0)
    
    # Perform nested cross-validation
    cv_results = nested_cross_validation(X, y, cpg_sites, data)
    
    # Train final model
    final_model, scaler, selected_features = train_final_model(X, y, cpg_sites)
    
    print(f"\nPhase 3 completed successfully!")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final model uses {len(selected_features)} CpG sites")
    print(f"- Model saved to 'result' directory with '3_' prefix")


if __name__ == "__main__":
    main()
