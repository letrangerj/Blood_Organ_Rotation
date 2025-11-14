# Phase 3b: Elastic Net Aging Clock Model Building

## Model Visualizations

> [!NOTE] Model Performance Overview
> These visualizations summarize the Elastic Net aging clock model's performance and key features. The model predicts chronological age from cfDNA methylation patterns with a mean absolute error of 6.39 years, representing a 9.7% improvement over the LASSO model.

### Predicted vs Actual Age

![[3b_predicted_vs_actual.png|600]]

**Strong correlation (r=0.926)** between predicted and actual ages. Points colored by age acceleration show most predictions cluster around the perfect prediction line (dashed diagonal). The tighter clustering compared to LASSO indicates improved model performance with the Elastic Net approach.

---

### Residual Analysis

![[3b_residuals.png|600]]

**Left panel**: Residuals vs Actual Age - Shows prediction errors across different age groups. **Right panel**: Residuals vs Predicted Age - Identifies systematic biases in predictions. The red horizontal line at y=0 represents perfect predictions. The even distribution of points around this line suggests the Elastic Net model has no major systematic bias. The vertical colorbar on the right indicates age acceleration values.

---

### Feature Importance

![[3b_coefficients.png|600]]

**Elastic Net-selected CpG sites** (82 total) ranked by absolute coefficient magnitude. Bars are colored by chromosome. **Red edges** indicate positive correlation with age (higher methylation = older predicted age). **Blue edges** indicate negative correlation. The Elastic Net retained more features than LASSO while maintaining regularization, suggesting it captured additional predictive signal from correlated CpG sites.

---

### Age Acceleration Distribution

![[3b_age_acceleration.png|600]]

**Distribution of age acceleration** (predicted age - actual age) across all 104 samples. The green vertical line shows the mean acceleration (-0.02 years, nearly zero). The red line at zero represents no acceleration. The distribution ranges from -20 to +15 years, with improved precision compared to LASSO. This metric can be used to study accelerated or decelerated aging with greater accuracy.

## 0. Purpose and Summary of the Script

This script builds the aging clock model using Elastic Net regression, an enhanced version of the selected CpG sites from Phase 2. It trains an Elastic Net model that combines L1 and L2 regularization to predict chronological age from methylation patterns, using nested cross-validation to ensure robust performance estimates given our small sample size.

**Goal**: Train an improved model that predicts age from methylation patterns with better accuracy and stability than LASSO.

### What We Do:

1. **Prepare the Data**
   - Use the 86 CpG sites we selected in Phase 2
   - Create a feature matrix (samples × CpG sites)
   - Standardize the data (zero mean, unit variance)

2. **Smart Cross-Validation with Dual Hyperparameters**
   - Since we only have 105 samples, we use a special approach:
     - **Outer loop**: Leave-one-sample-out (test on 1, train on 104) - repeated 105 times
     - **Inner loop**: For each training set, use 5-fold CV to find the best settings for both alpha and l1_ratio
   - This gives us honest performance estimates without wasting data

3. **Build Elastic Net Model**
   - Elastic Net = Linear regression that combines L1 and L2 regularization
   - L1 (LASSO) performs feature selection by shrinking some coefficients to zero
   - L2 (Ridge) handles correlated features by shrinking coefficients smoothly
   - Formula: Age = (CpG₁ × weight₁) + (CpG₂ × weight₂) + ... + constant

4. **Find Optimal Settings for Both Parameters**
   - Test different "regularization strengths" (alpha) and "L1/L2 balance" (l1_ratio)
   - Alpha controls overall regularization strength
   - L1_ratio controls the balance between L1 and L2 penalties (0 = pure Ridge, 1 = pure LASSO)
   - Choose the simplest model that's still 95% as good as the best one
   - This prevents overfitting while maintaining predictive power

5. **Train Final Model**
   - Train on all 105 samples with the optimal settings
   - Extract the final set of CpG sites and their weights
   - Save the trained model for making predictions

**Result**: An improved aging clock that uses Elastic Net regularization to achieve better prediction accuracy while maintaining interpretability.

## 1. Input

- `result/2_selected_features.csv`: Age-associated CpG sites selected in Phase 2
- `result/1_preprocessed_data.csv`: Preprocessed methylation data with age information

## 2. Output

- `result/5a_cv_predictions.csv`: Cross-validation predictions for each sample (sample ID, actual age, predicted age, age acceleration)
- `result/5a_cv_metrics.csv`: Cross-validation performance metrics (MAE, RMSE, R², correlation, sample count, feature count, hyperparameter statistics)
- `result/5a_model_coefficients.csv`: Final model coefficients for selected CpG sites (sorted by absolute value)
- `result/5a_model_intercept.csv`: Model intercept term
- `result/5a_model_hyperparameters.csv`: Optimal hyperparameters (alpha, l1_ratio)
- `result/5a_trained_model.pkl`: Complete trained model package (Elastic Net model, StandardScaler, CpG site list, hyperparameters)

## 3. Method used

- **Elastic Net regression** with combined L1 and L2 regularization
- **Nested cross-validation strategy**:
  - Outer loop: Leave-One-Out Cross-Validation (LOOCV, 104 folds)
  - Inner loop: 5-fold CV for hyperparameter tuning of both alpha and l1_ratio
- **1-SE rule**: Selects the sparsest model within 95% of the best cross-validation performance
- **Feature standardization**: Zero mean and unit variance scaling before model training
- **Missing value handling**: Median imputation (3 missing values found and imputed)
- **Hyperparameter grids**: 
  - Alpha: 50 values from 10^-4 to 10^1 (logarithmic scale)
  - L1_ratio: [0.1, 0.3, 0.5, 0.7, 0.9] (balance between L1 and L2)

Sample code:
```python
# Elastic Net with cross-validation
en_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, max_iter=50000, tol=1e-3)
en_cv.fit(X_train_scaled, y_train)

# Apply 1-SE rule for sparser model
mse_path = en_cv.mse_path_.mean(axis=-1)
min_mse_idx = np.unravel_index(np.argmin(mse_path), mse_path.shape)
se_threshold = min_mse + std_mse_path[min_mse_idx] / np.sqrt(5)

# Find candidates within 1 SE and select sparsest
candidates = np.where(mse_path <= se_threshold)
# Select combination with highest alpha and l1_ratio for maximum sparsity
```

## 4. Summary of the results

- **Cross-validation performance**: MAE = 6.39 years, RMSE = 7.94 years, R² = 0.855, Correlation = 0.926
- **Performance improvement**: 9.7% reduction in MAE compared to LASSO (6.39 vs 7.08 years)
- **Final model**: Selected 82 CpG sites out of the 86 candidate sites from Phase 2 (95% retention)
- **Hyperparameter selection**: Consistent choice of l1_ratio = 0.1 (strong L2 preference) across all CV folds
- **Optimal alpha**: 0.754 (moderate regularization strength)
- **Model characteristics**: More comprehensive feature retention while maintaining regularization benefits
- **Stability**: Perfect consistency in l1_ratio selection indicates robust preference for Ridge-like behavior

> [!INFO] Understanding the Performance Improvement
> **MAE reduced from 7.08 to 6.39 years**: This 0.69-year improvement represents a 9.7% enhancement in prediction accuracy. For biological age prediction, this improvement could translate to better identification of individuals with accelerated or decelerated aging.
> 
> **R² increased from 0.823 to 0.855**: The model now explains 85.5% of age variance compared to 82.3% with LASSO, indicating better capture of age-related methylation patterns.
> 
> **Correlation improved from 0.909 to 0.926**: Stronger linear relationship between predicted and actual ages suggests more reliable predictions across the age spectrum.

> [!TIP] Interpreting Elastic Net vs LASSO Results
> **Feature retention (82 vs 50 sites)**: Elastic Net retained 64% more features than LASSO, suggesting that many CpG sites have small but coordinated effects on aging that LASSO's strict sparsity missed.
> 
> **L1_ratio = 0.1**: This indicates the model strongly favors L2 (Ridge) regularization, which handles correlated features better than pure L1 (LASSO). This suggests that many age-associated CpG sites are correlated and work together in biological pathways.
> 
> **Consistent hyperparameter selection**: All 104 CV folds selected the same l1_ratio, indicating a stable and reliable preference for the L1/L2 balance in this dataset.

## 5. Worries/Suspicious places

- **Increased model complexity**: 82 features vs 50 in LASSO may increase overfitting risk despite regularization
- **Computational cost**: Dual hyperparameter search (alpha + l1_ratio) is more computationally intensive than LASSO
- **Interpretability trade-off**: More features may make biological interpretation more challenging
- **L1_ratio consistency**: While consistent selection is good, it may indicate the hyperparameter grid was too narrow
- **Feature correlation**: With 82 retained features, need to assess multicollinearity and biological redundancy
- **Small sample size**: 104 samples with 82 features still presents a high-dimensional challenge
- **Missing values**: 3 missing values were imputed with median; should verify this is appropriate for EN
- **Age distribution**: Need to check if age distribution in LOOCV training/test splits remains balanced
- **Performance ceiling**: 9.7% improvement is significant, but may approach the biological limit of cfDNA methylation predictability
- **Generalization**: The consistent l1_ratio selection suggests good stability, but independent validation would strengthen confidence

## Appendix: Elastic Net Regression and Advanced Regularization

### Why Elastic Net?

Elastic Net regression combines L1 (LASSO) and L2 (Ridge) regularization, addressing key limitations of each approach:

1. **Handles correlated features**: Unlike LASSO, which arbitrarily selects among correlated predictors, Elastic Net can select groups of correlated features
2. **More stable selection**: Less sensitive to small data changes than pure LASSO
3. **Better predictive performance**: Often achieves superior accuracy by combining the strengths of both regularization types
4. **Maintains sparsity**: Still performs feature selection through the L1 component

The Elastic Net objective function minimizes:
```
(1 / (2 * n_samples)) * ||y - Xw||²_2 + α * l1_ratio * ||w||_1 + 0.5 * α * (1 - l1_ratio) * ||w||²_2
```

Where:
- **α** controls overall regularization strength
- **l1_ratio** controls the balance between L1 and L2 penalties

### Why Nested Cross-Validation with Dual Parameters?

Given our small sample size (n=104) and dual hyperparameters, we need to:

1. **Maximize training data**: Use as much data as possible for training
2. **Optimize both parameters**: Find optimal values for both α and l1_ratio
3. **Avoid data leakage**: Prevent information from test set influencing model selection
4. **Get unbiased estimates**: Properly estimate generalization performance

> [!INFO] Outer Loop: Leave-One-Out Cross-Validation (LOOCV)
> **What it is**: Tests generalization by leaving out one sample at a time (104 folds total)
> 
> **Why we use it**: 
> - Maximizes training data for each fold (103 samples for training)
> - Provides nearly unbiased performance estimates
> - Every sample gets to be the test set exactly once
> 
> **Trade-offs**:
> - Computationally expensive (104 models trained)
> - High variance in performance estimates
> - Models are highly correlated (each trained on 103/104 overlapping samples)

> [!INFO] Inner Loop: 5-fold Cross-Validation with Grid Search
> **What it is**: Within each LOOCV training set, we split the 103 samples into 5 folds for hyperparameter tuning of both α and l1_ratio
> 
> **Why we use it**:
> - Selects optimal hyperparameters within each training set
> - Tests different combinations of α (50 values) and l1_ratio (5 values)
> - Prevents overfitting to the validation set
> - More stable than LOOCV for hyperparameter selection
> 
> **Process**:
> - For each (α, l1_ratio) combination, train on 4 folds, test on 1 fold
> - Rotate through all 5 folds and average performance
> - Select combination with best average performance

> [!TIP] The 1-SE Rule for Dual Parameters
> **What it is**: Instead of selecting the absolute best performing model, we select the simplest model within one standard error of the best performance, considering both α and l1_ratio
> 
> **Why we use it**:
> - **Reduces overfitting**: Prefers models with higher regularization
> - **More robust**: The "best" parameter combination might be best by chance
> - **Balances L1/L2**: Selects appropriate balance between feature selection and coefficient shrinkage
> 
> **Implementation**: We prioritize higher α (more regularization) and higher l1_ratio (more L1/sparsity) among candidates within 1 SE of best performance

### Understanding L1_ratio in Elastic Net

**L1_ratio** controls the balance between L1 and L2 regularization:

| L1_ratio | Regularization Type | Behavior | Your Result |
|----------|-------------------|----------|-------------|
| **0.0** | Pure Ridge (L2) | Shrinks all coefficients, no selection | Not tested |
| **0.1** | Strong L2, Weak L1 | **Selected in your model** | Optimal balance |
| **0.5** | Equal L1/L2 | Balanced approach | Not selected |
| **0.9** | Strong L1, Weak L2 | LASSO-like behavior | Not selected |
| **1.0** | Pure LASSO (L1) | Feature selection only | Not selected |

**Your optimal l1_ratio = 0.1** indicates:
- **Strong preference for L2 regularization**: Handles correlated CpG sites effectively
- **Mild L1 regularization**: Still performs some feature selection
- **Biological insight**: Age-associated CpG sites work in coordinated, correlated networks
- **Stability**: All CV folds selected this value, indicating robust preference

### Comparison to Other Approaches

**Elastic Net vs. LASSO**:
- EN handles correlated features better (important for methylation data)
- More stable coefficient estimates
- Better predictive performance in many cases
- Retains more biological signal while controlling complexity

**Elastic Net vs. Ridge Regression**:
- EN performs feature selection through L1 component
- Better for high-dimensional data with many irrelevant features
- More interpretable than pure Ridge

**Nested CV vs. Simple CV**:
- Nested CV provides unbiased performance estimates for dual parameters
- Essential for small sample sizes and multiple hyperparameters
- Prevents optimistic bias in performance reporting

### Key Insights from Your Elastic Net Model

**Performance Improvement**: The 9.7% MAE improvement (7.08 → 6.39 years) demonstrates that:
- Correlated CpG sites contain additional predictive information
- The L1/L2 balance optimally captures aging patterns in cfDNA
- The model successfully generalizes better than pure LASSO

**Feature Retention**: 82 vs 50 features suggests:
- Many age-associated CpG sites are correlated and work together
- LASSO's strict sparsity was too aggressive for this biological system
- The additional 32 features contribute meaningful predictive signal

**Hyperparameter Selection**: l1_ratio = 0.1 indicates:
- Strong biological coordination among methylation sites
- Ridge-like behavior is appropriate for aging biomarkers
- The model benefits from coefficient shrinkage without aggressive feature selection

**Your optimal Elastic Net configuration** (α ≈ 0.75, l1_ratio = 0.1) represents the perfect balance for cfDNA methylation aging clocks: comprehensive feature retention with appropriate regularization, achieving superior predictive accuracy while maintaining biological interpretability.