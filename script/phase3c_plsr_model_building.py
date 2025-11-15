#!/usr/bin/env python3
"""
Phase 3c: Aging Clock Model Building with PLSR (Partial Least Squares Regression)
Aging Clock Project - PLSR Implementation

This script builds the aging clock model using PLSR with nested cross-validation.
PLSR handles multicollinearity and dimensionality reduction by finding latent components
that explain maximum variance in both features and target.
"""

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
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
    print("Loading data for PLSR model building...")
    
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


def calculate_vip_scores(pls_model, X, y):
    """
    Calculate VIP (Variable Importance in Projection) scores for PLSR
    
    Args:
        pls_model: Fitted PLSRegression model
        X: Feature matrix
        y: Target vector
    
    Returns:
        array: VIP scores for each feature
    """
    # Get the number of components
    n_components = pls_model.n_components
    
    # Get the PLS weights (X_weights_)
    W = pls_model.x_weights_
    
    # Get the PLS loadings (X_loadings_)
    P = pls_model.x_loadings_
    
    # Calculate the sum of squares for each component
    ssq = np.sum(pls_model.x_scores_**2, axis=0) * np.sum(pls_model.y_loadings_**2, axis=1)
    
    # Calculate VIP scores
    vip_scores = np.sqrt(X.shape[1] * np.sum((W**2) * ssq.reshape(1, -1), axis=1) / np.sum(ssq))
    
    return vip_scores


def nested_cross_validation(X, y, cpg_sites, data, result_dir="result"):
    """
    Perform nested cross-validation with LOOCV outer loop and 5-fold inner loop
    for PLSR hyperparameter tuning
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        data (DataFrame): Original data with sample IDs
        result_dir (str): Directory to save results
    
    Returns:
        dict: Cross-validation results
    """
    print("\nStarting nested cross-validation with PLSR...")
    print("Outer loop: Leave-One-Out Cross-Validation (105 folds)")
    print("Inner loop: 5-fold CV for hyperparameter tuning")
    print("Tuning parameter: number of components")
    
    # Initialize LOOCV for outer loop
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Store predictions and actual values
    predictions = np.zeros(n_samples)
    actual = y.copy()
    
    # Store model parameters from each outer fold
    coefficients_list = []
    vip_scores_list = []
    
    # Define hyperparameter grid for PLSR
    max_components = min(X.shape[1], n_samples - 1)  # Prevent overfitting
    component_range = range(1, min(max_components + 1, 21))  # Test 1-20 components
    print(f"Testing components: {list(component_range)}")
    
    # Store optimal parameters for analysis
    optimal_components = []
    
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
        
        # Test different numbers of components
        best_components = 1
        best_score = float('inf')
        
        for n_components in component_range:
            # Perform inner CV for this number of components
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled):
                X_inner_train, X_inner_val = X_train_scaled[inner_train_idx], X_train_scaled[inner_val_idx]
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
                # Fit PLSR
                pls = PLSRegression(n_components=n_components)
                pls.fit(X_inner_train, y_inner_train)
                
                # Predict and calculate MAE
                y_pred = pls.predict(X_inner_val)
                mae = mean_absolute_error(y_inner_val, y_pred)
                inner_scores.append(mae)
            
            # Average MAE across inner folds
            avg_mae = np.mean(inner_scores)
            
            if avg_mae < best_score:
                best_score = avg_mae
                best_components = n_components
        
        optimal_components.append(best_components)
        
        # Train final PLSR model with optimal components on full training set
        final_pls = PLSRegression(n_components=best_components)
        final_pls.fit(X_train_scaled, y_train)
        
        # Predict the held-out sample
        pred = final_pls.predict(X_test_scaled)[0]
        predictions[test_idx[0]] = pred
        
        # Store coefficients (regression coefficients)
        coefficients_list.append(final_pls.coef_.flatten())
        
        # Calculate and store VIP scores
        vip_scores = calculate_vip_scores(final_pls, X_train_scaled, y_train)
        vip_scores_list.append(vip_scores)
        
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
    print(f"  Optimal components range: {min(optimal_components)} - {max(optimal_components)}")
    print(f"  Mean optimal components: {np.mean(optimal_components):.1f}")
    print(f"  Most common components: {max(set(optimal_components), key=optimal_components.count)}")
    print(f"  Note: PLSR uses dimensionality reduction, not feature selection")
    
    # Save CV predictions
    cv_results = pd.DataFrame({
        'Sample_ID': data.index,
        'Actual_Age': actual,
        'Predicted_Age': predictions,
        'Age_Acceleration': predictions - actual
    })
    cv_results.to_csv(os.path.join(result_dir, "3c_cv_predictions.csv"), index=False)
    
    # Save performance metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'mean_optimal_components': np.mean(optimal_components),
        'std_optimal_components': np.std(optimal_components)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_dir, "3c_cv_metrics.csv"), index=False)
    
    return {
        'predictions': predictions,
        'actual': actual,
        'metrics': metrics,
        'coefficients': coefficients_list,
        'vip_scores': vip_scores_list,
        'optimal_components': optimal_components
    }


def train_final_model(X, y, cpg_sites, result_dir="result"):
    """
    Train final PLSR model on all data with optimal hyperparameter selection
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        cpg_sites (list): CpG site names
        result_dir (str): Directory to save results
    
    Returns:
        tuple: Trained model, scaler, and selected features
    """
    print("\nTraining final PLSR model on all data...")
    
    # Define hyperparameter range
    max_components = min(X.shape[1], len(y) - 1)
    component_range = range(1, min(max_components + 1, 21))
    
    # Handle missing values before training final model
    if np.isnan(X).any():
        print("Imputing missing values with median...")
        X_median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), X_median, X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform 5-fold CV to select optimal components
    best_components = 1
    best_score = float('inf')
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for n_components in component_range:
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit PLSR
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train, y_train)
            
            # Predict and calculate MAE
            y_pred = pls.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
        
        avg_mae = np.mean(cv_scores)
        
        if avg_mae < best_score:
            best_score = avg_mae
            best_components = n_components
    
    print(f"Optimal components selected: {best_components}")
    
    # Train final PLSR model with optimal components
    final_pls = PLSRegression(n_components=best_components)
    final_pls.fit(X_scaled, y)
    
    # All features are used in PLSR (dimensionality reduction, not selection)
    selected_cpg_sites = cpg_sites.copy()
    selected_coefficients = final_pls.coef_.flatten()
    
    print(f"Final PLSR model uses {best_components} components from {len(selected_cpg_sites)} CpG sites")
    print(f"Dimensionality reduction: {len(selected_cpg_sites)} → {best_components} components")
    
    # Save model coefficients
    coefficients_df = pd.DataFrame({
        'CpG_site': selected_cpg_sites,
        'coefficient': selected_coefficients
    })
    coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
    coefficients_df.to_csv(os.path.join(result_dir, "3c_model_coefficients.csv"), index=False)
    
    # Save intercept
    intercept_df = pd.DataFrame({
        'intercept': [final_pls.intercept_]
    })
    intercept_df.to_csv(os.path.join(result_dir, "3c_model_intercept.csv"), index=False)
    
    # Save hyperparameters
    hyperparams_df = pd.DataFrame({
        'n_components': [best_components]
    })
    hyperparams_df.to_csv(os.path.join(result_dir, "3c_model_hyperparameters.csv"), index=False)
    
    # Calculate and save VIP scores for final model
    vip_scores = calculate_vip_scores(final_pls, X_scaled, y)
    vip_df = pd.DataFrame({
        'CpG_site': selected_cpg_sites,
        'VIP_score': vip_scores
    })
    vip_df = vip_df.sort_values('VIP_score', ascending=False)
    vip_df.to_csv(os.path.join(result_dir, "3c_vip_scores.csv"), index=False)
    
    # Save the trained model and scaler
    model_package = {
        'model': final_pls,
        'scaler': scaler,
        'cpg_sites': cpg_sites,
        'n_components': best_components,
        'vip_scores': vip_scores
    }
    
    with open(os.path.join(result_dir, "3c_trained_model.pkl"), 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"PLSR model saved with {best_components} components")
    print(f"VIP scores calculated for feature importance ranking")
    
    return final_pls, scaler, selected_cpg_sites


def plot_predicted_vs_actual(result_dir="result", figure_dir="figure"):
    """
    Create predicted vs actual age plot using PLSR model results
    
    Args:
        result_dir (str): Directory where PLSR model results are stored
        figure_dir (str): Directory to save the plot
    """
    print("Creating PLSR predicted vs actual plot...")
    
    # Load CV predictions
    predictions_path = os.path.join(result_dir, "3c_cv_predictions.csv")
    df = pd.read_csv(predictions_path)
    
    # Load metrics
    metrics_path = os.path.join(result_dir, "3c_cv_metrics.csv")
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
    ax.set_title('PLSR Aging Clock: Predicted vs Actual Age\n' + 
                f'({len(df)} samples, {metrics["n_features"]} CpG sites, {int(metrics["mean_optimal_components"])} components)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)
    
    # Save plot with PLSR-specific filename
    output_path = os.path.join(figure_dir, "3c_predicted_vs_actual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PLSR predicted vs actual plot saved to: {output_path}")
    
    return fig


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("Phase 3c: Aging Clock Model Building with PLSR")
    print("=" * 80)
    print("Using Partial Least Squares Regression for dimensionality reduction")
    print("Handles multicollinearity and finds latent components correlated with age")
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
    
    print(f"\nPhase 3c completed successfully!")
    print(f"- Nested CV performance: MAE = {cv_results['metrics']['MAE']:.2f} years")
    print(f"- Final PLSR model uses {final_model.n_components} components from {len(selected_features)} CpG sites")
    print(f"- Model saved to 'result' directory with '3c_' prefix")
    print(f"- VIP scores calculated for feature importance ranking")
    print(f"- Predicted vs actual plot saved to 'figure' directory")


if __name__ == "__main__":
    main()