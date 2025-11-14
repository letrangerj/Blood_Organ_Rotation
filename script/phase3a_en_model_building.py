#!/usr/bin/env python3
"""
Phase 3a: Aging Clock Model Building with Elastic Net
Aging Clock Project - Elastic Net Implementation

This script builds the aging clock model using Elastic Net regression with nested cross-validation.
Elastic Net combines L1 and L2 regularization for improved feature selection and stability.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
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
    print("Loading data for Elastic Net model building...")
    
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
    for Elastic Net hyperparameter tuning
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation with Elastic Net...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    print("Tuning parameters: alpha (regularization strength) and l1_ratio (L1 vs L2 balance)")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store model coefficients from each outer fold
    coefficients_list = []
    
    # Define hyperparameter grids for Elastic Net
    alphas = np.logspace(-4, 1, 50)  # 10^-4 to 10^1
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # Balance between L1 and L2
    
    # Store optimal parameters for analysis
    optimal_alphas = []
    optimal_l1_ratios = []
    
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
        
        # Fit Elastic Net with cross-validation
        en_cv = ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=inner_cv,
            max_iter=50000,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
            selection='cyclic'  # Use cyclic coordinate descent for stability
        )
        en_cv.fit(X_train_scaled, y_train)
        
        # Get optimal parameters
        optimal_alpha = en_cv.alpha_
        optimal_l1_ratio = en_cv.l1_ratio_
        optimal_alphas.append(optimal_alpha)
        optimal_l1_ratios.append(optimal_l1_ratio)
        
        # Apply 1-SE rule for sparser model
        mse_path = en_cv.mse_path_.mean(axis=-1)  # Average over CV folds: shape (n_alphas, n_l1_ratios)
        std_mse_path = en_cv.mse_path_.std(axis=-1)
        
        # Find the best performance
        min_mse_idx = np.unravel_index(np.argmin(mse_path), mse_path.shape)
        min_mse = mse_path[min_mse_idx]
        se_threshold = min_mse + std_mse_path[min_mse_idx] / np.sqrt(5)
        
        # Find all parameter combinations within 1 SE of the best
        candidates = np.where(mse_path <= se_threshold)
        
        if len(candidates[0]) > 0:
            # Get all valid candidate combinations
            candidate_alphas = []
            candidate_l1_ratios = []
            candidate_scores = []
            
            for i in range(len(candidates[0])):
                alpha_idx = candidates[0][i]
                l1_idx = candidates[1][i]
                
                # Ensure l1_idx is within valid range
                if l1_idx < len(l1_ratios):
                    alpha = en_cv.alphas_[alpha_idx]
                    l1_ratio = l1_ratios[l1_idx]
                    
                    candidate_alphas.append(alpha)
                    candidate_l1_ratios.append(l1_ratio)
                    
                    # Score: prioritize higher alpha (more regularization) and higher l1_ratio (more L1/sparse)
                    score = np.log10(alpha) + l1_ratio
                    candidate_scores.append(score)
            
            if len(candidate_scores) > 0:
                # Select the candidate with the highest score (most regularization + most L1)
                best_idx = np.argmax(candidate_scores)
                optimal_alpha = candidate_alphas[best_idx]
                optimal_l1_ratio = candidate_l1_ratios[best_idx]
        
        # Retrain with selected parameters
        final_en = ElasticNetCV(
            alphas=[optimal_alpha],
            l1_ratio=[optimal_l1_ratio],
            cv=inner_cv,
            max_iter=50000,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
            selection='cyclic'
        )
        final_en.fit(X_train_scaled, y_train)
        
        # Predict the held-out sample
        pred = final_en.predict(X_test_scaled)[0]
        predictions[test_idx[0]] = pred
        
        # Store coefficients
        coefficients_list.append(final_en.coef_)
        
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
    
    # Hyperparameter statistics
    print(f"\nHyperparameter selection summary:")
    print(f"  Optimal alpha range: {min(optimal_alphas):.6f} - {max(optimal_alphas):.6f}")
    print(f"  Mean optimal alpha: {np.mean(optimal_alphas):.6f}")
    print(f"  Optimal l1_ratio range: {min(optimal_l1_ratios):.2f} - {max(optimal_l1_ratios):.2f}")
    print(f"  Mean optimal l1_ratio: {np.mean(optimal_l1_ratios):.2f}")
    print(f"  L1_ratio distribution: {dict(zip(*np.unique(optimal_l1_ratios, return_counts=True)))}")
    
    # Save CV predictions
    cv_results = pd.DataFrame({
        'Sample_ID': data.index,
        'Actual_Age': actual,
        'Predicted_Age': predictions,
        'Age_Acceleration': predictions - actual
    })
    cv_results.to_csv(os.path.join(result_dir, "5a_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'mean_optimal_alpha': np.mean(optimal_alphas),
        'mean_optimal_l1_ratio': np.mean(optimal_l1_ratios),
        'std_optimal_alpha': np.std(optimal_alphas),
        'std_optimal_l1_ratio': np.std(optimal_l1_ratios)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "5a_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'coefficients': coefficients_list,
        'optimal_alphas': optimal_alphas,
        'optimal_l1_ratios': optimal_l1_ratios
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final Elastic Net model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model, scaler, and selected features
    """
    print("\nTraining final Elastic Net model on all data...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define hyperparameter grids
    alphas = np.logspace(-4, 1, 50)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Perform 5-fold CV to select optimal parameters
    en_cv = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=5,
        max_iter=50000,
        tol=1e-3,
        n_jobs=-1,
        random_state=42,
        selection='cyclic'
    )
    en_cv.fit(X_scaled, y)
    
    # Apply 1-SE rule for sparser model
    optimal_alpha = en_cv.alpha_
    optimal_l1_ratio = en_cv.l1_ratio_
    
    mse_path = en_cv.mse_path_.mean(axis=-1)
    std_mse_path = en_cv.mse_path_.std(axis=-1)
    
    min_mse_idx = np.unravel_index(np.argmin(mse_path), mse_path.shape)
    min_mse = mse_path[min_mse_idx]
    se_threshold = min_mse + std_mse_path[min_mse_idx] / np.sqrt(5)
    
    candidates = np.where(mse_path <= se_threshold)
    if len(candidates[0]) > 0:
        # Get all valid candidate combinations
        candidate_alphas = []
        candidate_l1_ratios = []
        candidate_scores = []
        
        for i in range(len(candidates[0])):
            alpha_idx = candidates[0][i]
            l1_idx = candidates[1][i]
            
            # Ensure l1_idx is within valid range
            if l1_idx < len(l1_ratios):
                alpha = en_cv.alphas_[alpha_idx]
                l1_ratio = l1_ratios[l1_idx]
                
                candidate_alphas.append(alpha)
                candidate_l1_ratios.append(l1_ratio)
                
                # Score: prioritize higher alpha (more regularization) and higher l1_ratio (more L1/sparse)
                score = np.log10(alpha) + l1_ratio
                candidate_scores.append(score)
        
        if len(candidate_scores) > 0:
            # Select the candidate with the highest score (most regularization + most L1)
            best_idx = np.argmax(candidate_scores)
            optimal_alpha = candidate_alphas[best_idx]
            optimal_l1_ratio = candidate_l1_ratios[best_idx]
    
    print(f"Optimal alpha selected: {optimal_alpha:.6f}")
    print(f"Optimal l1_ratio selected: {optimal_l1_ratio:.2f}")
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train final model with selected parameters
    final_model = ElasticNetCV(
        alphas=[optimal_alpha],
        l1_ratio=[optimal_l1_ratio],
        cv=5,
        max_iter=50000,
        tol=1e-3,
        n_jobs=-1,
        random_state=42,
        selection='cyclic'
    )
    final_model.fit(X_scaled, y)
    
    # Extract non-zero coefficients
    non_zero_idx = np.where(final_model.coef_ != 0)[0]
    selected_cpg_sites = [cpg_sites[i] for i in non_zero_idx]
    selected_coefficients = final_model.coef_[non_zero_idx]
    
    print(f"Final Elastic Net model selected {len(selected_cpg_sites)} CpG sites out of {len(cpg_sites)}")
    print(f"Model sparsity: {len(selected_cpg_sites)/len(cpg_sites)*100:.1f}%")
    
    # Save model coefficients
    coefficients_df = pd.DataFrame({
        'CpG_site': selected_cpg_sites,
        'coefficient': selected_coefficients
    })
    coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
    coefficients_df.to_csv(os.path.join(result_dir, "5a_model_coefficients.csv"), index=False)
    
    # Save intercept
    intercept_df = pd.DataFrame({
        'intercept': [final_model.intercept_]
    })
    intercept_df.to_csv(os.path.join(result_dir, "5a_model_intercept.csv"), index=False)
    
    # Save hyperparameters
    hyperparams_df = pd.DataFrame({
        'alpha': [optimal_alpha],
        'l1_ratio': [optimal_l1_ratio]
    })
    hyperparams_df.to_csv(os.path.join(result_dir, "5a_model_hyperparameters.csv"), index=False)
    
    # Save the trained model and scaler
    model_package = {
        'model': final_model,
        'scaler': scaler,
        'cpg_sites': cpg_sites,
        'alpha': optimal_alpha,
        'l1_ratio': optimal_l1_ratio
    }
    
    with open(os.path.join(result_dir, "5a_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Elastic Net model saved with {len(selected_cpg_sites)} features")
    print(f"Regularization: alpha={optimal_alpha:.6f}, l1_ratio={optimal_l1_ratio:.2f}")
    
    return final_model, scaler, selected_cpg_sites


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("Phase 3a: Aging Clock Model Building with Elastic Net")
    print("=" * 80)
    print("Using Elastic Net regression with L1 + L2 regularization")
    print("This combines the sparsity of LASSO with the stability of Ridge regression")
    print()
    
    # Load data
    X, y, cpg_sites = load_data()
    data = pd.read_csv("result/1_preprocessed_data.csv", index_col=0)
    
    # Perform nested cross-validation
    cv_results = nested_cross_validation(X, y, cpg_sites, data)
    
    # Train final model
    final_model, scaler, selected_features = train_final_model(X, y, cpg_sites)
    
    print(f"\nPhase 3a completed successfully!")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final Elastic Net model uses {len(selected_features)} CpG sites")
    print(f"- Model saved to 'result' directory with '5a_' prefix")
    print(f"- Regularization parameters: alpha={final_model.alpha_:.6f}, l1_ratio={final_model.l1_ratio_:.2f}")


if __name__ == "__main__":
    main()