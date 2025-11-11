# Aging Clock Development Plan: Detailed Task Breakdown

## Project Overview
Build an aging clock using cell-free DNA (cfDNA) methylation data to predict chronological age based on CpG site methylation patterns.

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

### Phase 5: Results Interpretation & Visualization

#### Step 5.1: Model Performance Visualization
- **Predicted vs. Actual Age Plot**
  - Scatter plot using nested CV predictions (outer LOOCV) with diagonal reference line
  - Color-code by gender or age group
  - Add correlation coefficient and R²

- **Residual Plot**
  - Residuals vs. actual age using nested CV predictions
  - Horizontal reference line at y=0
  - Assess systematic bias

- **Bland-Altman Plot**
  - Agreement between predicted and actual age using nested CV predictions
  - Mean difference and limits of agreement

- **Hyperparameter Selection Visualization**
  - Plot cross-validation error vs. regularization parameter α
  - Indicate the selected sparsest model within 95% of best performance

#### Step 5.2: Feature Importance Analysis
- Rank CpG sites by absolute coefficient values
- Bar plot of top contributing features
- Separate plots for age-positive and age-negative sites
- Annotate with genomic locations and nearby genes

#### Step 5.3: Biological Interpretation
- Heatmap of selected CpG methylation patterns
  - Rows: samples (ordered by age)
  - Columns: top CpG sites
  - Color scale: methylation level
- Identify genomic context of top sites:
  - Gene associations
  - Regulatory regions
  - Biological pathways

#### Step 5.4: Generate Output Files
1. **Selected features**
   - List of CpG sites (chr, start, end) selected by LASSO
   - PCC values from Phase 2
   - Final LASSO model coefficients
   - p-values

2. **Model predictions**
   - Sample ID
   - Actual age
   - Predicted age
   - Age acceleration
   - Prediction interval (if applicable)

3. **Performance summary**
   - All metrics (MAE, RMSE, R², correlation)
   - Cross-validation results
   - Model parameters
   - Final number of CpG sites in the LASSO model
   - Selected regularization parameter (α) value

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
2. **Trained Model**: LASSO model with selected CpG sites and their coefficients
3. **Performance Report**: All evaluation metrics including cross-validation results
4. **Predictions Table**: Ages and acceleration for all samples
5. **Visualization Suite**: All diagnostic and interpretation plots
6. **Analysis Script**: Documented, reproducible code
7. **Model Parameters**: Optimal regularization parameter (α) and selected feature count

---

## Tools & Libraries (Python)

- **Data handling**: pandas, numpy
- **Statistical analysis**: scipy, statsmodels
- **Machine learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Multiple testing**: statsmodels.stats.multitest

---

## Quality Checkpoints

- [ ] Data loaded correctly with matching sample IDs
- [ ] No missing values or handled appropriately
- [ ] Correlation analysis shows significant age-associated sites
- [ ] Feature selection criteria clearly defined and applied
- [ ] Model shows reasonable performance (MAE < 10 years target)
- [ ] Cross-validation results are stable
- [ ] No systematic bias in residuals
- [ ] Visualizations are clear and interpretable
- [ ] Results are biologically plausible

---

*This plan provides a comprehensive framework for developing an aging clock. Adjust thresholds and methods based on initial exploratory results.*
