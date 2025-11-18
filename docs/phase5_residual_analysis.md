# Phase 5: Residual Analysis and Age-Bias Correction

Date: 2025-11-18T07:35:43.136Z

> [!EXAMPLE] Residual vs Age (All Models)
> ![[figure/5_residual_vs_age_all_models.png]]
> Figure: Residuals (Predicted − Actual) vs Actual Age before (red) and after (green) correction. Lines show fitted residual-age trend on original residuals.

## Objectives
- Detect and quantify age-dependent bias (correlation between residuals and actual age) for each model.
- Apply a simple linear correction to remove age bias from predictions.
- Re-compute model performance with corrected predictions and summarize improvements.
- Produce key diagnostics and corrected outputs for downstream use.

## Data Inputs
- CV predictions from previous phases per model: `result/{prefix}cv_predictions.csv`
  - Columns: `Sample_ID, Actual_Age, Predicted_Age, Age_Acceleration (Predicted-Actual)`

Models covered:
- LASSO (3_)
- Elastic Net (5a_)
- Ridge (3b_)
- PLSR (3c_)
- SVR (3d_)
- Random Forest (3e_)
- XGBoost (3f_)
- LightGBM (3g_)

## Method
### 1) Test age–residual correlation
- Define residuals r = Predicted_Age − Actual_Age (a.k.a. age acceleration).
- Fit r = β0 + β1·Actual_Age via ordinary least squares.
- Report: β0, β1, R², Pearson r, p-value.

### 2) Apply age-bias correction
- Correct each sample’s prediction by removing the fitted residual component:
  - Corrected_Pred = Original_Pred − (β0 + β1·Actual_Age)
  - Corrected_Age_Diff = Corrected_Pred − Actual_Age
- This shifts predictions so residuals are uncorrelated with age under the fitted linear model.
- Correction is applied to all models regardless of p-value for consistency.

### 3) Recompute performance
For each model, compute before/after metrics using Actual_Age vs predictions:
- MAE, RMSE, R², Pearson correlation.
- Improvement (%) is computed for MAE and RMSE as (before−after)/before × 100.

## Key Results (After Correction)
| Model | β1 (slope) | MAE_before | MAE_after |
|---|---|---|---|
| LASSO | -0.1191 | 7.0780 | 6.7463 |
| Elastic Net | -0.1921 | 6.3943 | 5.2756 |
| Ridge | -0.1753 | 6.4151 | 5.4611 |
| PLSR | -0.1340 | 5.9210 | 5.3629 |
| SVR | -0.1633 | 6.8402 | 6.1688 |
| Random Forest | -0.5018 | 10.0731 | 4.8287 |
| XGBoost | -0.3070 | 8.1491 | 6.4786 |
| LightGBM | -0.3053 | 8.2355 | 6.6313 |

## Outputs
| File | Meaning |
|---|
| result/phase5_corrected_age_diffs.csv | Wide table per sample with: `Sample_ID`, `Actual_Age`, `<Model>_original_prediction`, `<Model>_original_residual` (Pred−Actual), `<Model>_corrected_prediction`, `<Model>_corrected_age_diff` (Corrected_Pred−Actual). Note: corrected_age_diff is the primary corrected residual; redundant columns were removed. |
| result/phase5_correction_parameters.csv | One row per model with: `model, beta_0, beta_1, r_squared, p_value, correlation_coef, correlation_significant, mae_before, rmse_before, r2_before, corr_before, mae_after, rmse_after, r2_after, corr_after, mae_improvement_pct, rmse_improvement_pct`. |
| result/phase5_residual_analysis_summary.csv | Compact summary per model of key before/after metrics and significance flag. |
| figure/5_residual_vs_age_all_models.png | Diagnostic plot: residuals vs actual age for each model, showing original and corrected residuals with the fitted residual-age regression line. |

## Reproduction
- Script: `script/phase5_residual_analysis.py`
- Run: `python script/phase5_residual_analysis.py`

## Interpretation
- β1 close to 0 and high p-value -> little age-dependent bias; small change after correction.
- Significant β1 (p < 0.05) -> noticeable age bias; correction typically reduces MAE/RMSE and flattens residual–age trend.
- Use `phase5_corrected_age_diffs.csv` for downstream age-acceleration analyses; `<Model>_corrected_age_diff` is the corrected residual per sample.
