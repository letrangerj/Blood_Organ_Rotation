# Aging Clock Development Plan: Detailed Task Breakdown

## Project Overview
Build comprehensive aging clock models using cell-free DNA (cfDNA) methylation data to predict chronological age based on CpG site methylation patterns. This project implements multiple machine learning algorithms (LASSO, Elastic Net, Ridge, PLSR, SVR, Random Forest, XGBoost, LightGBM) with nested cross-validation and includes age-correlation analysis to correct for systematic prediction bias.

---

## Data Structure Understanding

### 1. overlap_cfDNA.tsv
- **Structure**: Methylation matrix
- **Rows**: Genomic loci (CpG sites) with coordinates
  - Column 1: `chr` (chromosome)
  - Column 2: `start` (genomic start position)
  - Column 3: `end` (genomic end position)
  - Column 4: `MD` (metadata/descriptor)
- **Columns 5+**: 105 sample IDs with methylation beta values
- **Values**: Continuous methylation levels (0-1 range)

### 2. Metadata_PY_104.csv
- **Sample**: Sample IDs matching column names in methylation matrix
- **Gender**: M/F
- **Age**: Chronological age in years (range: 20-84)
- **Annotation**: Sample type (all cfDNA)
- **Total samples**: 105

---

## Phase 0: Environment Setup

### Step 0.1: Create Conda Environment with Micromamba

#### Install Micromamba (if not already installed)
```bash
# Download and install micromamba (Linux)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Or if already installed, verify installation
micromamba --version
```

#### Create the `blood_organ` Environment
```bash
# Create new environment with Python 3.10
micromamba create -n blood_organ python=3.10 -y

# Activate the environment
micromamba activate blood_organ
```

#### Install Required Packages
```bash
# Core data science packages
micromamba install -c conda-forge \
    pandas \
    numpy \
    scipy \
    statsmodels \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    -y

# Add environment to Jupyter kernels (optional, if using notebooks)
python -m ipykernel install --user --name=blood_organ --display-name="Python (blood_organ)"
```

#### Package Versions (Recommended)
- **pandas**: ≥2.0.0 (data manipulation)
- **numpy**: ≥1.24.0 (numerical operations)
- **scipy**: ≥1.10.0 (statistical functions)
- **statsmodels**: ≥0.14.0 (statistical modeling, multiple testing)
- **scikit-learn**: ≥1.3.0 (machine learning models)
- **matplotlib**: ≥3.7.0 (plotting)
- **seaborn**: ≥0.12.0 (statistical visualization)

#### Verify Installation
```bash
# Check installed packages
micromamba list

# Test imports in Python
python -c "import pandas, numpy, scipy, sklearn, matplotlib, seaborn, statsmodels; print('All packages loaded successfully!')"
```

#### Environment Management
```bash
# Deactivate environment
micromamba deactivate

# Reactivate when needed
micromamba activate blood_organ

# Export environment for reproducibility
micromamba env export -n blood_organ > environment.yml

# Remove environment (if needed)
micromamba env remove -n blood_organ
```

---

## Detailed Implementation Plan

### Phase 1: Data Preparation & Exploration

#### Step 1.1: Load and Validate Data
- Import methylation matrix (overlap_cfDNA.tsv)
- Import metadata (Metadata_PY_104.csv)
- Verify sample ID correspondence between both files
- Check data dimensions and structure

#### Step 1.2: Quality Control
- Identify missing values in methylation data
- Check value ranges (should be 0-1 for beta values)
- Examine age distribution in the cohort
- Check for outliers or anomalous samples
- Verify no duplicate samples

#### Step 1.3: Data Transformation
- Transpose methylation matrix: rows = samples, columns = CpG sites
- Merge age information with methylation data
- Create unified dataset for analysis
- Generate summary statistics:
  - Number of CpG sites
  - Age distribution (mean, median, range)
  - Methylation distribution per site (mean, variance)

---

### Phase 2: Feature Selection via Correlation Analysis

#### Step 2.1: Calculate Pearson Correlation Coefficients
For each CpG site:
- Compute PCC between methylation level and age across all samples
- Calculate p-values for statistical significance
- Store results: CpG site ID, PCC value, p-value

#### Step 2.2: Multiple Testing Correction
- Apply correction method (Benjamini-Hochberg FDR or Bonferroni)
- Generate adjusted p-values
- Prevent false positives from testing thousands of sites

#### Step 2.3: Select Age-Associated Sites
- Define selection criteria:
  - Significance threshold: adjusted p-value < 0.05
  - Effect size threshold: |PCC| > 0.2 (or optimize based on data)
- Filter CpG sites meeting criteria
- Separate positively and negatively correlated sites
- Document number of selected features

#### Step 2.4: Exploratory Visualization (Optional)
- Histogram of PCC distribution across all sites
- Volcano plot: PCC vs. -log10(p-value)
- Scatter plots of top correlated sites vs. age
- Correlation heatmap of selected sites

---

### Phase 3: Aging Clock Model Building

#### Step 3.1: Prepare Modeling Dataset
- Extract methylation values for selected CpG sites only
- Create feature matrix X (samples × selected CpG sites)
- Create target vector y (ages)
- Check for multicollinearity among selected features
- Standardize features to have zero mean and unit variance for proper regularization

#### Step 3.2: Data Splitting Strategy
Given small sample size (n=105):
- Use nested cross-validation:
  - **Outer loop**: Leave-one-out cross-validation (LOOCV) for performance evaluation
  - **Inner loop**: 5-fold cross-validation for hyperparameter tuning
  - This strategy maximizes data usage while providing unbiased performance estimates
  - Ensures representative age distribution in both loops

#### Step 3.3: Build LASSO Regression Model (Primary Choice)
**Primary approach: LASSO (Least Absolute Selection and Shrinkage Operator)**
- L1 regularization that promotes sparsity by shrinking some coefficients to exactly zero
- Age = β₀ + β₁×CpG₁ + β₂×CpG₂ + ... + βₙ×CpGₙ
- Hyperparameter α controls the strength of regularization (larger α = more sparse model)

**Alternative approaches (for comparison):**
- Ridge Regression: L2 regularization, prevents overfitting but doesn't perform feature selection
- Elastic Net: Combines L1 and L2 regularization

#### Step 3.4: Hyperparameter Selection Strategy
- For each fold of the outer LOOCV:
  - Hold out one sample for testing
  - On the remaining 104 samples, perform inner 5-fold CV to select optimal α:
    - Perform grid search across a range of α values (e.g., 10^-4 to 10^1 on log scale)
    - For each α value in the inner CV, calculate average performance metric (e.g., MAE)
    - Identify the α value that yields the best inner CV performance
    - Select the sparsest model within 95% of the best inner CV performance (1-SE rule)
  - Fit LASSO model on the 104 samples with the selected α
  - Predict the held-out sample

#### Step 3.5: Model Training and Selection
- Perform nested CV with outer LOOCV to evaluate final model performance
- For final model selection, use entire dataset with the optimal hyperparameter selection approach:
  - Use 5-fold CV to select optimal regularization parameter α
  - Select the sparsest model within 95% of the best cross-validation performance (1-SE rule)
- Fit final LASSO model with the optimally selected α value
- Extract model coefficients for non-zero features
- Identify the set of CpG sites selected by LASSO (non-zero coefficients)
- Document the final number of CpG sites in the model
- Save the trained model for prediction and evaluation

---

### Phase 4: Model Evaluation & Validation

#### Step 4.1: Calculate Performance Metrics
- **Mean Absolute Error (MAE)**: Average absolute deviation in years
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors
- **R² Score**: Proportion of variance explained
- **Pearson/Spearman Correlation**: Between predicted and actual age

#### Step 4.2: Cross-Validation Assessment
- Perform nested cross-validation as implemented in Phase 3:
  - Use outer LOOCV results for final performance assessment
  - Calculate metrics for each of the 105 test samples
  - Report overall performance metrics (MAE, RMSE, R², correlation)
  - Assess model stability and generalizability

#### Step 4.3: Age Acceleration Analysis
- Calculate for each sample using the nested CV predictions:
  - Age acceleration = Predicted age - Chronological age
  - Positive values: accelerated aging
  - Negative values: decelerated aging
- Examine distribution of age acceleration
- Check for systematic bias across age ranges

#### Step 4.4: Subgroup Analysis
- Compare model performance by:
  - Gender (M vs F)
  - Age groups (young, middle-aged, elderly)
- Identify potential biases or limitations

#### Step 4.5: Residual Analysis
- Plot residuals vs. actual age
- Check for heteroscedasticity
- Identify systematic patterns in errors
- Validate model assumptions

---

### Phase 5: Extended Model Reproduction with Multiple Algorithms

#### Step 5.1: Reproduce Models with Additional Algorithms
Build aging clock models using different machine learning algorithms following the exact same framework as Phase 3 and 3a:

**Models to implement:**
- **Ridge Regression**: L2 regularization for stable linear modeling
- **PLSR (Partial Least Squares Regression)**: Handles multicollinearity and dimensionality reduction
- **SVR (Support Vector Regression)**: Non-linear modeling with kernel methods
- **RF (Random Forest)**: Ensemble tree-based method for non-linear patterns
- **XGBoost**: Gradient boosting for complex interactions
- **LightGBM**: Efficient gradient boosting implementation

**Implementation Requirements:**
- Use exact same input as `phase3_model_building.py` and `phase3a_en_model_building.py`
- Load data from: `result/2_selected_features.csv` and `result/1_preprocessed_data.csv`
- Use same nested CV framework (LOOCV outer, 5-fold inner)
- Follow same data preprocessing and feature selection from Phase 2
- Apply appropriate hyperparameter grids for each algorithm
- Use 1-SE rule for model selection where applicable
- Maintain exact same output structure as existing scripts, only change prefix
- Script naming: `phase3b_...`, `phase3c_...`, etc. for consistency

#### Step 5.2: Age Correlation Analysis Based on Residuals
**Objective**: Address systematic bias where model predicts young people older and old people younger

**Key Result**: Generate "corrected age diff" vector for each sample - the final corrected age acceleration values

**Methodology:**
1. **Extract residuals** from nested CV predictions for each model
2. **Linear regression of residuals vs. actual age**:
   - Model: residual = β₀ + β₁ × actual_age + ε
   - Test significance of age coefficient (β₁)
   - Calculate R² for residual-age relationship
3. **Apply age-based correction**:
   - If significant correlation exists, adjust predictions:
   - Corrected_prediction = Original_prediction - (β₀ + β₁ × actual_age)
4. **Calculate corrected age diff**:
   - Corrected_age_diff = Corrected_prediction - Actual_age
   - This is the key result vector for each sample
5. **Evaluate correction effectiveness**:
   - Compare MAE/RMSE before and after correction
   - Assess residual patterns post-correction
   - Check if age-related bias is eliminated

**Implementation Details:**
- Create separate script `phase5_residual_analysis.py` to analyze all models
- Load predictions from each model's CV results (from phase3, phase3a, phase3b-g)
- Perform linear regression: `residual ~ actual_age` for each model
- Calculate correlation coefficient, p-value, and R²
- Apply correction: `corrected_pred = original_pred - (β₀ + β₁ × actual_age)`
- Generate corrected_age_diff vector for each sample
- Save correction parameters and corrected age diff vectors
- Generate residual plots for visual assessment

#### Step 5.3: Model Comparison and Ensemble Analysis
- **Performance comparison** across all models (LASSO, Elastic Net, Ridge, PLSR, SVR, RF, XGB, LGBM)
- **Corrected age diff comparison** - primary focus on the key result vectors
- **Residual pattern analysis** for each model
- **Feature importance comparison** where applicable
- **Correlation analysis** between different model predictions
- **Ensemble modeling** potential (simple averaging, weighted averaging based on corrected performance)

#### Step 5.4: Script Structure and Output Consistency

**Input Requirements (Exact Same as Existing Scripts):**
- Load selected features from: `result/2_selected_features.csv`
- Load preprocessed data from: `result/1_preprocessed_data.csv`
- Use same data preprocessing pipeline as `phase3_model_building.py`
- Handle missing values identically (median imputation)
- Use same StandardScaler approach

**Output Requirements (Same Structure, Different Prefix):**
- CV predictions: `[prefix]_cv_predictions.csv` with columns: Sample_ID, Actual_Age, Predicted_Age, Age_Acceleration
- CV metrics: `[prefix]_cv_metrics.csv` with MAE, RMSE, R², Correlation
- Model coefficients: `[prefix]_model_coefficients.csv` with CpG_site, coefficient
- Model intercept: `[prefix]_model_intercept.csv`
- Trained model: `[prefix]_trained_model.pkl`
- **Predicted vs Actual Age Plot**: Generate plot using same settings as `3a_model_visualization.py` and save as `[prefix]_predicted_vs_actual.png` in `figure` directory
- Additional model-specific files where applicable (hyperparameters, feature importance)

**Script Naming Convention:**
- Ridge: `phase3b_ridge_model_building.py`
- PLSR: `phase3c_plsr_model_building.py`
- SVR: `phase3d_svr_model_building.py`
- Random Forest: `phase3e_rf_model_building.py`
- XGBoost: `phase3f_xgb_model_building.py`
- LightGBM: `phase3g_lgbm_model_building.py`

**Visualization Requirements for All Models:**
- Each model reproduction script must include a `plot_predicted_vs_actual()` function
- Use exact same settings as `3a_model_visualization.py` and `3b_en_model_visualization.py`
- Generate scatter plot: Actual Age vs Predicted Age
- Color points by Age Acceleration using `RdYlBu_r` colormap
- Include perfect prediction line (black dashed line)
- Add metrics text box with MAE, RMSE, R², and Correlation
- Save plot as `[prefix]_predicted_vs_actual.png` in `figure` directory
- Call plot function in main execution after model training is complete

**Ridge Regression Implementation:**
- Use `RidgeCV` with built-in cross-validation
- Hyperparameter grid: `alphas = np.logspace(-4, 4, 100)`
- No feature selection (all features retained)
- Apply same nested CV framework as LASSO/Elastic Net
- **Include predicted vs actual age plot**: Add visualization function to generate plot with same settings as `3a_model_visualization.py` (scatter plot with color-coded age acceleration, perfect prediction line, metrics box, saved as `3b_predicted_vs_actual.png`)

**PLSR Implementation:**
- Use `PLSRegression` from `sklearn.cross_decomposition`
- Hyperparameter: number of components (1 to min(n_samples, n_features))
- Use cross-validation to select optimal components
- Extract VIP (Variable Importance in Projection) scores for feature importance

**SVR Implementation:**
- Use `SVR` with RBF kernel as default
- Hyperparameters: `C` (regularization), `gamma` (kernel coefficient), `epsilon` (tube width)
- Grid search: `C: [0.1, 1, 10, 100]`, `gamma: ['scale', 0.001, 0.01, 0.1]`, `epsilon: [0.01, 0.1, 0.5]`
- Standardize features before SVR (important for kernel methods)

**Random Forest Implementation:**
- Use `RandomForestRegressor` with nested CV
- Hyperparameters: `n_estimators: [100, 300, 500]`, `max_depth: [3, 5, 10, None]`, `min_samples_split: [2, 5, 10]`
- Extract feature importance using `feature_importances_`
- No standardization required (tree-based method)

**XGBoost Implementation:**
- Use `XGBRegressor` with early stopping
- Hyperparameters: `n_estimators: [100, 300, 500]`, `max_depth: [3, 5, 7]`, `learning_rate: [0.01, 0.1, 0.3]`, `subsample: [0.8, 1.0]`
- Extract feature importance using `feature_importances_` with gain metric
- Handle missing values natively

**LightGBM Implementation:**
- Use `LGBMRegressor` with early stopping
- Hyperparameters: `n_estimators: [100, 300, 500]`, `num_leaves: [31, 50, 100]`, `learning_rate: [0.01, 0.1, 0.3]`, `subsample: [0.8, 1.0]`
- Extract feature importance using `feature_importances_` with gain metric
- Handle missing values natively

#### Step 5.5: Implementation Details for Additional Models

**Ridge Regression Implementation:**
- Use `RidgeCV` with built-in cross-validation
- Hyperparameter grid: `alphas = np.logspace(-4, 4, 100)`
- No feature selection (all features retained)
- Apply same nested CV framework as LASSO/Elastic Net

**PLSR Implementation:**
- Use `PLSRegression` from `sklearn.cross_decomposition`
- Hyperparameter: number of components (1 to min(n_samples, n_features))
- Use cross-validation to select optimal components
- Extract VIP (Variable Importance in Projection) scores for feature importance

**SVR Implementation:**
- Use `SVR` with RBF kernel as default
- Hyperparameters: `C` (regularization), `gamma` (kernel coefficient), `epsilon` (tube width)
- Grid search: `C: [0.1, 1, 10, 100]`, `gamma: ['scale', 0.001, 0.01, 0.1]`, `epsilon: [0.01, 0.1, 0.5]`
- Standardize features before SVR (important for kernel methods)

**Random Forest Implementation:**
- Use `RandomForestRegressor` with nested CV
- Hyperparameters: `n_estimators: [100, 300, 500]`, `max_depth: [3, 5, 10, None]`, `min_samples_split: [2, 5, 10]`
- Extract feature importance using `feature_importances_`
- No standardization required (tree-based method)

**XGBoost Implementation:**
- Use `XGBRegressor` with early stopping
- Hyperparameters: `n_estimators: [100, 300, 500]`, `max_depth: [3, 5, 7]`, `learning_rate: [0.01, 0.1, 0.3]`, `subsample: [0.8, 1.0]`
- Extract feature importance using `feature_importances_` with gain metric
- Handle missing values natively

**LightGBM Implementation:**
- Use `LGBMRegressor` with early stopping
- Hyperparameters: `n_estimators: [100, 300, 500]`, `num_leaves: [31, 50, 100]`, `learning_rate: [0.01, 0.1, 0.3]`, `subsample: [0.8, 1.0]`
- Extract feature importance using `feature_importances_` with gain metric
- Handle missing values natively

#### Step 5.6: Residual Analysis and Corrected Age Diff Generation

**Primary Output - Corrected Age Diff Vectors:**
- Generate corrected age diff for each sample from each model
- Save as: `phase5_corrected_age_diffs.csv` with columns:
  - Sample_ID
  - [model]_corrected_age_diff (e.g., lasso_corrected_age_diff, ridge_corrected_age_diff, etc.)
  - [model]_correction_applied (boolean indicating if significant age-residual correlation was found)

**Correction Parameters Output:**
- Save correction parameters for each model: `phase5_correction_parameters.csv`
- Columns: model, beta_0, beta_1, r_squared, p_value, correlation_significant

**Residual Analysis Results:**
- Before/after correction comparison: `phase5_residual_analysis_summary.csv`
- Residual vs age plots for each model
- Distribution of corrected age diffs across age ranges
1. **Model comparison summary**
   - Performance metrics for all 8 models (before and after correction)
   - Corrected age diff statistics and distributions
   - Feature counts and selection stability

2. **Corrected age diff vectors** (Primary Result)
   - Corrected age acceleration for each sample from each model
   - Correction parameters and significance tests
   - Before/after correction comparison

3. **Model coefficients/feature importance**
   - Ridge: All coefficients (no sparsity)
   - PLSR: Component loadings and VIP scores
   - SVR: Support vectors and coefficients
   - Tree-based: Feature importance rankings
   - Boosting: Gain-based importance metrics

4. **Residual analysis results**
   - Age-residual correlation coefficients and p-values
   - Correction parameters (β₀, β₁) for each model
   - Residual plots and bias assessment

---

## Key Considerations & Best Practices

### Statistical Considerations
- **Sample size**: n=105 is modest
  - Avoid overfitting: limit features or use regularization
  - Use nested cross-validation (LOOCV outer, 5-fold inner) for unbiased evaluation
  - Consider dimensionality: features << samples

- **Feature selection**:
  - Balance between too few (underfitting) and too many (overfitting)
  - Rule of thumb: 1 feature per 10-15 samples
  - Consider step-wise selection or regularization-based selection

- **Model validation**:
  - Never evaluate on training data
  - Nested cross-validation is essential with limited samples for unbiased performance estimates
  - Report confidence intervals where possible
  - Use 1-SE rule to select sparsest model within 95% of best performance

### Biological Considerations
- **Age-correlated CpGs**:
  - Hypermethylation: often in CpG islands, developmental genes
  - Hypomethylation: often in gene bodies, repetitive elements
  - Both reflect distinct biological aging processes

- **cfDNA specificity**:
  - Cell-free DNA reflects tissue-of-origin
  - May capture multi-tissue aging signatures
  - Different from tissue-specific clocks

### Technical Considerations
- **Data preprocessing**:
  - Handle missing values appropriately
  - Consider normalization if needed
  - Check for batch effects

- **Reproducibility**:
  - Set random seeds for splitting/CV
  - Document all parameters
  - Save trained model for future predictions

---

## Expected Deliverables

1. **Feature List**: Age-correlated CpG sites with statistics from Phase 2
2. **Trained Models**: 
   - LASSO model with selected CpG sites and their coefficients
   - Elastic Net model with selected CpG sites and their coefficients
   - Ridge Regression model with all coefficients
   - PLSR model with component loadings and VIP scores
   - SVR model with support vectors and coefficients
   - Random Forest model with feature importance rankings
   - XGBoost model with gain-based importance metrics
   - LightGBM model with feature importance rankings
3. **Performance Report**: All evaluation metrics including cross-validation results for all 8 models
4. **Predictions Table**: Ages and acceleration for all samples from each model
5. **Residual Analysis**: Age correlation results and correction factors for bias adjustment
6. **Visualization Suite**: All diagnostic and interpretation plots including predicted vs actual age plots for all 8 models
7. **Analysis Scripts**: Documented, reproducible code for all models following consistent structure
8. **Model Parameters**: Optimal hyperparameters for each algorithm
9. **Model Comparison**: Comprehensive performance comparison across all 8 methods
10. **Corrected Age Diff Vectors**: Primary result - bias-corrected age acceleration values for each sample from all 8 models

---

## Tools & Libraries (Python)

- **Data handling**: pandas, numpy
- **Statistical analysis**: scipy, statsmodels
- **Machine learning**: scikit-learn, scikit-learn.cross_decomposition (PLSR)
- **Advanced models**: 
  - XGBoost: xgboost
  - LightGBM: lightgbm
- **Visualization**: matplotlib, seaborn
- **Multiple testing**: statsmodels.stats.multitest
- **Model persistence**: pickle, joblib

---

## Quality Checkpoints

- [ ] Data loaded correctly with matching sample IDs
- [ ] No missing values or handled appropriately
- [ ] Correlation analysis shows significant age-associated sites
- [ ] Feature selection criteria clearly defined and applied
- [ ] All 8 models (LASSO, Elastic Net, Ridge, PLSR, SVR, RF, XGB, LGBM) implemented consistently
- [ ] Nested cross-validation framework applied uniformly across all models
- [ ] Each model shows reasonable performance (MAE < 10 years target)
- [ ] Cross-validation results are stable and reproducible
- [ ] Residual-age correlation analysis completed for bias detection
- [ ] Age-based correction applied where significant bias detected
- [ ] Corrected age diff vectors generated for all samples (primary result)
- [ ] Model comparison metrics calculated and documented
- [ ] Feature importance/coefficients extracted for interpretability
- [ ] Visualizations are clear and interpretable
- [ ] Predicted vs actual age plot generated for each model (Ridge, PLSR, SVR, RF, XGB, LGBM) with consistent settings
- [ ] Results are biologically plausible
- [ ] All scripts follow consistent structure and naming conventions (phase3b, phase3c, etc.)
- [ ] Input/output consistency maintained with existing scripts
- [ ] Correction parameters properly saved and documented

---

*This plan provides a comprehensive framework for developing an aging clock. Adjust thresholds and methods based on initial exploratory results.*
