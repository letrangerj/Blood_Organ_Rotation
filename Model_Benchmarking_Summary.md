# Aging Clock Model Benchmarking Summary

> *Comprehensive analysis of 8 machine learning models for cfDNA methylation-based age prediction*

## 📊 Executive Summary

This document presents a systematic benchmarking of 8 machine learning models for aging clock development using cfDNA methylation data. The analysis reveals that **linear models (PLSR and Ridge) significantly outperform ensemble methods** for this dataset, with PLSR achieving the best performance (MAE = 5.92 years, R² = 0.868).

---

## 🧠 Model Overview & Core Algorithms

### 1. LASSO Regression `phase3`

**Core Algorithm**: LASSO minimizes the residual sum of squares with an L1 penalty:

$$
\hat{\beta} = \arg\min_{\beta} \left\{ \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}
$$

where $\lambda$ controls the strength of L1 regularization, driving some coefficients to exactly zero.

**Key Properties**:
- **Automatic feature selection** through L1 penalty
- **Sparse solutions** - many coefficients become exactly zero
- **Handles high-dimensional data** well (p >> n scenarios)

**Implementation**: Nested CV with LOOCV outer loop, 5-fold inner CV for α tuning, 1-SE rule for final model selection.

### 2. Elastic Net Regression `phase3a`

**Core Algorithm**: Combines L1 and L2 penalties in the objective function:

$$
\hat{\beta} = \arg\min_{\beta} \left\{ \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda \left[ \alpha \sum_{j=1}^p |\beta_j| + (1-\alpha) \sum_{j=1}^p \beta_j^2 \right] \right\}
$$

where $\lambda$ controls overall regularization strength and $\alpha \in [0,1]$ balances L1 vs L2 penalty.

**Key Properties**:
- **Balances feature selection and coefficient shrinkage**
- **Handles correlated features** better than pure L1
- **Maintains some sparsity** while improving stability

### 3. Ridge Regression `phase3b`

**Core Algorithm**: Minimizes residual sum of squares with L2 penalty:

$$
\hat{\beta} = \arg\min_{\beta} \left\{ \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}
$$

where $\lambda$ controls the strength of L2 regularization.

**Key Properties**:
- **Shrinks coefficients toward zero** without eliminating features
- **Excellent for correlated features** (common in methylation data)
- **Provides stable predictions** with reduced variance

### 4. PLSR (Partial Least Squares Regression) `phase3c`

**Core Algorithm**: Finds latent components that maximize covariance between X and y:

$$
\text{maximize } \text{Cov}(Xw, y) \text{ subject to } ||w|| = 1$$

**Mathematical Process**:
1. **Extract components** $t = Xw$ that explain maximum covariance
2. **Regress y on components**: $y = Tq + \epsilon$ where $T = [t_1, t_2, ..., t_k]$
3. **Iterative process**: Deflate X and y, repeat for next component

**Key Properties**:
- **Handles multicollinearity** naturally
- **Supervised dimensionality reduction** (unlike PCA)
- **Perfect for high-dimensional data** with correlated features

### 5. SVR (Support Vector Regression) `phase3d`

**Core Algorithm**: Finds function $f(x) = w^T x + b$ that deviates from targets by no more than ε while maximizing margin:

$$
\min_{w,b,\xi,\xi^*} \frac{1}{2}||w||^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$

subject to: $y_i - w^T x_i - b \leq \epsilon + \xi_i$ and $w^T x_i + b - y_i \leq \epsilon + \xi_i^*$

**Key Properties**:
- **ε-insensitive loss** - ignores errors smaller than ε
- **Margin maximization** for better generalization
- **Kernel flexibility** - linear, polynomial, RBF kernels

### 6. Random Forest `phase3e`

**Core Algorithm**: Ensemble of decision trees trained on bootstrap samples:

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

where each tree $T_b$ is trained on a bootstrap sample with random feature subset at each split.

**Training Process**:
1. **Bootstrap sampling**: Sample n observations with replacement
2. **Random feature selection**: Choose m features at each split (typically $\sqrt{p}$)
3. **Grow trees to maximum depth** (or other stopping criteria)
4. **Average predictions** across all trees

**Key Properties**:
- **Reduces variance** through averaging
- **Handles missing values** naturally
- **Provides feature importance** rankings
- **Robust to outliers**

### 7. XGBoost `phase3f`

**Core Algorithm**: Gradient boosting with second-order optimization:

$$\text{Objective} = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

where $l$ is the loss function and $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$ is regularization.

**Training Process**:
1. **Initialize**: $\hat{y}_i^{(0)} = 0$
2. **For k = 1 to K trees**:
   - Compute gradients: $g_i = \partial_{\hat{y}_i^{(k-1)}} l(y_i, \hat{y}_i^{(k-1)})$
   - Compute Hessians: $h_i = \partial_{\hat{y}_i^{(k-1)}}^2 l(y_i, \hat{y}_i^{(k-1)})$
   - Build tree to fit negative gradients
   - Update: $\hat{y}_i^{(k)} = \hat{y}_i^{(k-1)} + \eta f_k(x_i)$

**Key Properties**:
- **Second-order optimization** (uses Hessian information)
- **Built-in regularization** (L1/L2 penalties)
- **Native missing value handling**
- **Fast C++ implementation**

### 8. LightGBM `phase3g`

**Core Algorithm**: Gradient boosting with leaf-wise tree growth and gradient-based sampling:

**Leaf-wise Growth**: Selects the leaf with maximum delta loss to split:
$$\text{Choose leaf } j \text{ that maximizes } |L_{\text{split}} - L_{\text{before}}|$$

**GOSS (Gradient-based One-Side Sampling)**:
- Keeps samples with large gradients (important for learning)
- Randomly drops samples with small gradients (less important)
- Maintains accuracy while reducing computation

**Training Process**:
1. **GOSS sampling**: Keep high-gradient samples, randomly sample low-gradient ones
2. **EFB bundling**: Group mutually exclusive features
3. **Leaf-wise splitting**: Always split the leaf with maximum gain
4. **Gradient boosting**: Standard gradient boosting with regularization

**Key Properties**:
- **Leaf-wise growth** often achieves better accuracy with fewer trees
- **Memory efficient** through GOSS/EFB sampling
- **Native missing value handling** with optimal split directions
- **Faster than XGBoost** for most datasets

---

## ⚙️ Implementation Comparison

### **Hyperparameter Strategy Comparison**

| Model | Key Parameters | Grid Strategy | Final Selection | Rationale |
|-------|----------------|---------------|-----------------|-----------|
| **LASSO** | α (regularization) | 10^-4 to 10^1 log scale | 1-SE rule | Balance sparsity vs performance |
| **Elastic Net** | α, l1_ratio | α: log scale, l1_ratio: 0.1-0.9 | 1-SE rule | Balance L1/L2 based on data |
| **Ridge** | α (L2 penalty) | 10^-4 to 10^4 log scale | 1-SE rule | Maximize regularization benefit |
| **PLSR** | n_components | 1-20 components | CV minimum | Prevent overfitting, small n |
| **SVR** | C, ε, kernel | Linear kernel, C: 0.1-100, ε: 0.01-1.0 | CV minimum | Linear chosen over RBF via CV |
| **Random Forest** | n_estimators, max_depth, min_samples | Conservative: 200-700 trees, depth 5-15, samples 10-20 | CV minimum | Prevent overfitting on small data |
| **XGBoost** | n_estimators, max_depth, learning_rate, reg_alpha/lambda | Conservative: 100-2000 trees, depth 2-4, lr 0.01-0.1, strong L1/L2 | CV minimum | Strong regularization essential |
| **LightGBM** | n_estimators, num_leaves, learning_rate, lambda_l1/l2 | Conservative: 100-1500 trees, leaves 10-50, lr 0.005-0.1, strong L1/L2 | CV minimum | Leaf-wise needs careful control |

### **Script Structure Comparison**

All scripts maintain **identical architecture**:

```
load_data() → nested_cross_validation() → train_final_model() → plot_predicted_vs_actual() → main()
```

**Common elements across all models:**
- **LOOCV outer loop** (105 folds) for unbiased evaluation
- **5-fold inner CV** for hyperparameter tuning
- **Conservative parameter grids** tailored for n=105 samples
- **Progress tracking** with tqdm for long operations
- **Identical output structure** (3*, 3a*, 3b*, etc. prefixes)
- **Same visualization function** for predicted vs actual plots

---

## 📈 Model Performance Comparison

### **Performance Ranking Table**

| Rank | Model | MAE (years) | R² | Method Type | Performance Notes |
|------|-------|-------------|-----|-------------|-------------------|
| **1** | **PLSR** | **5.92** | **0.868** | Linear | 🏆 **Champion** - Best overall performance |
| **2** | **Ridge** | 6.42 | 0.854 | Linear | Excellent linear performance |
| **3** | **SVR** | 6.84 | 0.842 | Linear | Good linear performance, linear kernel chosen |
| **4** | **XGBoost** | 8.15 | 0.759 | Ensemble | Best ensemble method, conservative approach |
| **5** | **LightGBM** | **8.24** | **0.747** | Ensemble | Slightly worse than XGBoost, leaf-wise growth |
| **6** | **Random Forest** | 10.07 | 0.660 | Ensemble | Poor performance, overfitting despite conservative params |

### **Key Performance Insights**

> **Major Finding**: **Linear models significantly outperform ensemble methods** for this cfDNA methylation aging dataset. The consistent superiority suggests that **aging methylation patterns are fundamentally linear and systematic** rather than complex and nonlinear.

> **Performance Gap Analysis**:
- **Linear vs Ensemble gap**: ~2.3 years MAE difference (5.92 vs 8.24)
- **Best vs Worst gap**: ~4.2 years MAE difference (5.92 vs 10.07)
- **All ensemble methods cluster**: 8.15-10.07 years MAE range
- **Linear models cluster**: 5.92-6.84 years MAE range

---

## 📊 Model Visualization Gallery

### **Predicted vs Actual Age Plots - All 8 Models**

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">

![[figure/3_predicted_vs_actual.png|400]]
*LASSO Regression - L1 regularization with automatic feature selection*

![[figure/3a_predicted_vs_actual.png|400]]
*Elastic Net - Combined L1/L2 regularization for balanced feature selection*

![[figure/3b_predicted_vs_actual.png|400]]
*Ridge Regression - L2 regularization for stable coefficient estimation*

![[figure/3c_predicted_vs_actual.png|400]]
*PLSR (Champion) - Partial Least Squares with supervised dimensionality reduction*

![[figure/3d_predicted_vs_actual.png|400]]
*Support Vector Regression - ε-insensitive loss with margin maximization*

![[figure/3e_predicted_vs_actual.png|400]]
*Random Forest - Ensemble of decision trees with bootstrap sampling*

![[figure/3f_predicted_vs_actual.png|400]]
*XGBoost - Gradient boosting with second-order optimization*

![[figure/3g_predicted_vs_actual.png|400]]
*LightGBM - Leaf-wise growth with GOSS sampling and EFB bundling*

</div>

> **Visual Patterns**:
- **Linear models** show tight clustering around the perfect prediction line
- **Ensemble methods** show more scatter, especially Random Forest
- **Age acceleration patterns** are consistent across all models
- **No systematic bias** visible in any model (good calibration)

---

## 🧬 Biological Insights

### **Consistent Top Features Across Models**

These CpG sites appear in top features across multiple models, suggesting they are fundamental aging biomarkers:

| CpG Site | Chromosome | LASSO | Ridge | PLSR | SVR | RF | XGBoost | LightGBM |
|----------|------------|-------|-------|------|-----|----|---------|----------|
| chr10:111078376-111078905 | 10 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| chr17:17700269-17700870 | 17 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| chr8:10652508-10655054 | 8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| chr5:93570546-93571113 | 5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> **Chromosomal Patterns**: **Chromosomes 5, 8, 10, 12, 17, 19, 22** consistently contain top aging-related CpG sites across all models, suggesting these regions contain fundamental aging regulatory elements.

### **Model-Specific Insights**

#### **Linear Models (LASSO, Ridge, PLSR)**
- **Systematic, coordinated patterns** across the genome
- **Smooth, continuous methylation changes** with age
- **Biologically interpretable** coefficient patterns

#### **Ensemble Methods (RF, XGBoost, LightGBM)**
- **Threshold-based patterns** and interactions
- **Complex, non-linear relationships** captured
- **Feature importance rankings** reveal key biomarkers
- **Missing value patterns** may indicate biological mechanisms

---

## 📋 Technical Implementation Notes

### **Conservative Design Philosophy**
All models used **conservative hyperparameter grids** specifically designed for small datasets (n=105):
- **Shallow trees** (depth 2-4 for tree-based models)
- **Strong regularization** (L1/L2 penalties)
- **Aggressive sampling** (feature/row subsampling)
- **Slow learning rates** (0.01-0.1 range)
- **Sufficient minimum samples** per split/leaf

### **Cross-Validation Strategy**
- **LOOCV outer loop**: 105 folds for unbiased performance estimation
- **5-fold inner CV**: For hyperparameter tuning
- **1-SE rule**: Selects simplest model within 1 standard error of best
- **Progress tracking**: Real-time updates with tqdm

### **Reproducibility Standards**
- **Fixed random seeds** across all models
- **Consistent data preprocessing** pipeline
- **Standardized output formats** with descriptive filenames
- **Comprehensive documentation** in code comments

---

## 🎯 Conclusions & Recommendations

### **Primary Finding**
**Linear models (PLSR and Ridge) significantly outperform ensemble methods for cfDNA methylation aging prediction.** This suggests that aging methylation patterns are fundamentally **linear, systematic, and coordinated** rather than complex and nonlinear.

### **Model Selection Recommendation**
**Use PLSR as the primary aging clock model** with Ridge as backup. The 2-component PLSR model achieves optimal performance while maintaining biological interpretability.

### **Scientific Implications**
1. **cfDNA methylation aging follows linear patterns** across the genome
2. **Complex ensemble methods are unnecessary** for this biological system
3. **2 latent components capture most aging information** in methylation data
4. **Consistent biomarkers exist** across chromosomes 5, 8, 10, 12, 17, 19, 22

### **Future Directions**
- **Validate PLSR model** on independent cfDNA datasets
- **Investigate biological mechanisms** behind top CpG sites
- **Develop clinical applications** using the 2-component PLSR model
- **Explore ensemble of linear models** rather than tree-based methods

---

## 📚 References & Further Reading

- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer.
- **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2013). *An Introduction to Statistical Learning*. Springer.
- **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- **Ke, G., et al.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NIPS*.
- **Tibshirani, R.** (1996). Regression Shrinkage and Selection via the Lasso. *JRSSB*.
- **T code and data**: Available in this repository under appropriate licenses.

---

*Document created: $(date +"%Y-%m-%d %H:%M")*  
*Last updated: $(date +"%Y-%m-%d %H:%M")*  
*Repository: [Blood_Organ_Rotation](https://github.com/letrangerj/Blood_Organ_Rotation)*