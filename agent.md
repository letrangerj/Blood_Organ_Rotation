# Aging Clock Project: Implementation Guide

## 1. Development Environment

**Environment**: `blood_organ` (micromamba)

Key packages: pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn

**Setup instructions**: See Phase 0 in `task1_detailed_plan.md`

**Quick start**:
```bash
micromamba create -n blood_organ python=3.10 -y
micromamba activate blood_organ
micromamba install -c conda-forge pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter -y
```

---

## 2. Project Plan

**Objective**: Build an aging clock using cfDNA methylation data to predict chronological age.

**Workflow**:
1. Data preparation & QC
2. Feature selection (PCC correlation with age)
3. Linear regression model building
4. Model evaluation & validation
5. Results visualization & interpretation

**Detailed breakdown**: See `task1_detailed_plan.md` for comprehensive step-by-step instructions.

---

## 3. Folder Structure

```
task1/
├── data/                          # Input data files
│   ├── overlap_cfDNA.tsv         # Methylation matrix (CpG sites × samples)
│   └── Metadata_PY_104.csv       # Sample metadata (ID, Age, Gender)
│
├── script/                        # Analysis scripts (one per phase)
│   ├── phase1_data_preparation.py
│   ├── phase2_feature_selection.py
│   ├── phase3_model_building.py
│   ├── phase4_model_evaluation.py
│   └── phase5_visualization.py
│
├── result/                        # Output files
│   ├── selected_features.csv     # Age-correlated CpG sites
│   ├── model_coefficients.csv    # Trained model parameters
│   ├── predictions.csv           # Sample-level predictions & age acceleration
│   └── performance_metrics.txt   # MAE, RMSE, R², correlation
│
├── figure/                        # Visualization outputs
│   ├── predicted_vs_actual.png
│   ├── residual_plot.png
│   ├── feature_importance.png
│   └── methylation_heatmap.png
│
├── task1.txt                      # Original task description
├── task1_detailed_plan.md         # Comprehensive implementation plan
└── agent.md                       # This file
```

---

## 4. Script Organization

Each phase corresponds to one Python script:

| Script | Phase | Key Functions |
|--------|-------|---------------|
| **phase1_data_preparation.py** | Data Prep & Exploration | Load data, QC checks, transpose matrix, merge metadata |
| **phase2_feature_selection.py** | Feature Selection | Calculate PCC, multiple testing correction, filter significant CpG sites |
| **phase3_model_building.py** | Model Building | Prepare features, train linear/regularized regression, save model |
| **phase4_model_evaluation.py** | Evaluation | Cross-validation, calculate metrics (MAE/RMSE/R²), age acceleration |
| **phase5_visualization.py** | Visualization | Generate plots for model performance and feature importance |

**Execution order**: Run scripts sequentially (phase1 → phase2 → phase3 → phase4 → phase5)

**Input/Output**: Each script reads from `data/` or `result/` (previous phase) and writes to `result/` or `figure/`

---

## Quick Reference

- **Input data**: 105 samples × ~thousands of CpG sites, ages 20-84 years
- **Target metric**: MAE < 10 years
- **Model approach**: Linear regression with regularization (Elastic Net recommended)
- **Validation**: k-fold cross-validation (k=5 or 10)
- **Key consideration**: Small sample size (n=105) → avoid overfitting
