#!/usr/bin/env python3
"""
Phase 3b: Aging Clock Model Building with Ridge Regression
Aging Clock Project - Ridge Regression Implementation

This script builds the aging clock model using Ridge regression with nested cross-validation.
Ridge regression uses L2 regularization for stable linear modeling without feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
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
    print("Loading data for Ridge regression model building...")
    
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
    for Ridge regression hyperparameter tuning
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation with Ridge regression...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    print("Tuning parameter: alpha (regularization strength)")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store model coefficients from each outer fold
    coefficients_list = []
    
    # Define hyperparameter grid for Ridge
    alphas = np.logspace(-4, 4, 100)  # 10^-4 to 10^4
    
    # Store optimal parameters for analysis
    optimal_alphas = []
    
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
        
        # Fit Ridge with cross-validation
        ridge_cv = RidgeCV(
            alphas=alphas,
            cv=inner_cv,
            scoring='neg_mean_absolute_error'
        )
        ridge_cv.fit(X_train_scaled, y_train)
        
        # Get optimal alpha - RidgeCV already finds the best alpha
        optimal_alpha = ridge_cv.alpha_
        optimal_alphas.append(optimal_alpha)
        
        # Use the RidgeCV model directly (no need to retrain)
        final_ridge = ridge_cv
        
        # Predict the held-out sample
        pred = final_ridge.predict(X_test_scaled)[0]
        predictions[test_idx[0]] = pred
        
        # Store coefficients (all features, no sparsity)
        coefficients_list.append(final_ridge.coef_)
        
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
    print(f"  Std optimal alpha: {np.std(optimal_alphas):.6f}")
    print(f"  Note: Ridge regression retains all features (no sparsity)")
    
    # Save CV predictions
    cv_results = pd.DataFrame({
        'Sample_ID': data.index,
        'Actual_Age': actual,
        'Predicted_Age': predictions,
        'Age_Acceleration': predictions - actual
    })
    cv_results.to_csv(os.path.join(result_dir, "3b_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'mean_optimal_alpha': np.mean(optimal_alphas),
        'std_optimal_alpha': np.std(optimal_alphas)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "3b_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'coefficients': coefficients_list,
        'optimal_alphas': optimal_alphas
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final Ridge regression model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model, scaler, and selected features (all features for Ridge)
    """
    print("\nTraining final Ridge regression model on all data...")
    
    # Define hyperparameter grid
    alphas = np.logspace(-4, 4, 100)
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform 5-fold CV to select optimal alpha
    ridge_cv = RidgeCV(
        alphas=alphas,
        cv=5,
        scoring='neg_mean_absolute_error'
    )
    ridge_cv.fit(X_scaled, y)
    
    # RidgeCV already finds the optimal alpha
    optimal_alpha = ridge_cv.alpha_
    
    print(f"Optimal alpha selected: {optimal_alpha:.6f}")
    
    # Use the already trained RidgeCV model (no need to retrain)
    final_model = ridge_cv
    
    # All features are used in Ridge (no sparsity)
    selected_cpg_sites = cpg_sites.copy()
    selected_coefficients = final_model.coef_.copy()
    
    print(f"Final Ridge model uses all {len(selected_cpg_sites)} CpG sites (no feature selection)")
    
    # Save model coefficients
    coefficients_df = pd.DataFrame({
        'CpG_site': selected_cpg_sites,
        'coefficient': selected_coefficients
    })
    coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
    coefficients_df.to_csv(os.path.join(result_dir, "3b_model_coefficients.csv"), index=False)
    
    # Save intercept
    intercept_df = pd.DataFrame({
        'intercept': [final_model.intercept_]
    })
    intercept_df.to_csv(os.path.join(result_dir, "3b_model_intercept.csv"), index=False)
    
    # Save hyperparameters
    hyperparams_df = pd.DataFrame({
        'alpha': [optimal_alpha]
    })
    hyperparams_df.to_csv(os.path.join(result_dir, "3b_model_hyperparameters.csv"), index=False)
    
    # Save the trained model and scaler
    model_package = {
        'model': final_model,
        'scaler': scaler,
        'cpg_sites': cpg_sites,
        'alpha': optimal_alpha
    }
    
    with open(os.path.join(result_dir, "3b_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Ridge regression model saved with {len(selected_cpg_sites)} features")
    print(f"Regularization parameter: alpha={optimal_alpha:.6f}")
    
    return final_model, scaler, selected_cpg_sites


def plot_predicted_vs_actual(result_dir="result", figure_dir="figure"):
    """
    Create predicted vs actual age plot using Ridge model results
    
    Args:
        result_dir (str): Directory where Ridge model results are stored
        figure_dir (str): Directory to save the plot
    """
    print("Creating Ridge predicted vs actual plot...")
    
    # Load CV predictions
    predictions_path = os.path.join(result_dir, "3b_cv_predictions.csv")
    df = pd.read_csv(predictions_path)
    
    # Load metrics
    metrics_path = os.path.join(result_dir, "3b_cv_metrics.csv")
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
    ax.set_title('Ridge Aging Clock: Predicted vs Actual Age\n' + 
                f'({len(df)} samples, {metrics["n_features"]} CpG sites)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)
    
    # Save plot with Ridge-specific filename
    output_path = os.path.join(figure_dir, "3b_predicted_vs_actual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Ridge predicted vs actual plot saved to: {output_path}")
    
    return fig


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("Phase 3b: Aging Clock Model Building with Ridge Regression")
    print("=" * 80)
    print("Using Ridge regression with L2 regularization")
    print("This method retains all features but shrinks coefficients for stability")
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
    
    print(f"\nPhase 3b completed successfully!")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final Ridge model uses all {len(selected_features)} CpG sites (no feature selection)")
    print(f"- Model saved to 'result' directory with '3b_' prefix")
    print(f"- Regularization parameter: alpha={final_model.alpha_:.6f}")
    print(f"- Predicted vs actual plot saved to 'figure' directory")


if __name__ == "__main__":
    main()