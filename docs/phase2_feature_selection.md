# Phase 2: Feature Selection via Correlation Analysis

## 0. Purpose of the script
This script identifies age-associated CpG sites by calculating Pearson correlation coefficients between methylation levels and chronological age, then applying multiple testing correction to select significantly associated sites for building the aging clock model.

## 1. Input
- `result/1_preprocessed_data.csv`: Output from Phase 1 containing samples as rows, CpG sites and age as columns

## 2. Output
- `result/2_correlation_results.csv`: Complete results including CpG site IDs, correlation coefficients, p-values, sample counts, adjusted p-values, and significance status for all analyzed sites
- `result/2_selected_features.csv`: Age-associated CpG sites meeting the selection criteria
- `result/2_summary.txt`: Summary statistics including total sites analyzed, significant sites, and selected sites counts

## 3. Method used
- Calculated Pearson correlation coefficients between methylation levels and age for each CpG site
- Applied Benjamini-Hochberg FDR (false discovery rate) multiple testing correction
- Selected sites based on two criteria: adjusted p-value < 0.05 AND absolute correlation coefficient > 0.2
- Counted positive and negative correlations separately

Sample code:
```python
# Calculate Pearson correlation
corr, p_value = stats.pearsonr(age_aligned, meth_aligned)

# Apply multiple testing correction
corrected_pvals = multipletests(corr_df['p_value'], method='fdr_bh')
corr_df['adj_p_value'] = corrected_pvals[1]
corr_df['significant'] = corrected_pvals[0]

# Select features based on criteria
selected = corr_df[
    (corr_df['adj_p_value'] < pval_threshold) & 
    (abs(corr_df['correlation']) > effect_threshold)
]
```

## 4. Summary of the results
- Analyzed 1,188 CpG sites (46 sites from the original 1,234 were excluded due to insufficient data)
- Identified 86 significantly age-associated CpG sites after multiple testing correction
- Of the selected sites: 45 showed positive correlation with age and 41 showed negative correlation
- All significant sites met the effect size threshold (|correlation| > 0.2)

## 5. Worries/Suspicious places
- 46 CpG sites were excluded due to insufficient data (likely missing values), which may bias the selection toward more complete sites
- The correlation threshold of 0.2 is relatively low; need to verify if this will lead to a robust model
- The sample size (104 samples) is relatively small for the number of features being analyzed; this may lead to overfitting in downstream modeling
- The model will depend on these 86 selected features, which is a significant reduction from the original 1,188 sites

## Appendix: Multiple Testing Correction Method

### Reason for Selection
When testing the association between age and methylation levels across 1,188 CpG sites, we are performing multiple statistical tests simultaneously. Without correction, the probability of obtaining false positive results increases dramatically. For example, with a significance threshold of α = 0.05 and 1,188 tests, we would expect about 59 false positives by chance alone. Multiple testing correction is essential to control the false discovery rate and ensure the reliability of our selected age-associated CpG sites.

### Benjamini-Hochberg FDR Method
The Benjamini-Hochberg (BH) procedure controls the False Discovery Rate (FDR), which is the expected proportion of false discoveries among all discoveries. The algorithm works as follows:

1. Sort all p-values from smallest to largest: p(1) ≤ p(2) ≤ ... ≤ p(m), where m is the total number of tests
2. Find the largest k such that p(k) ≤ (k/m) * α, where α is the desired significance level (0.05 in our case)
3. Reject all null hypotheses for p-values p(1), p(2), ..., p(k)
4. Calculate adjusted p-values using the formula for each p-value: adj_p_value(i) = min(1, m * p(i) / i) after appropriate ordering

### Pros and Cons vs. Other Methods

**Benjamini-Hochberg FDR vs. Bonferroni:**
- Pros: More powerful, less conservative, allows for some false positives to maintain more true discoveries
- Cons: Less stringent control of type I error compared to Bonferroni

**Benjamini-Hochberg FDR vs. Holm-Bonferroni:**
- Pros: Simpler to understand and implement, widely accepted in genomics studies
- Cons: Holm-Bonferroni is uniformly more powerful but more complex

**Benjamini-Hochberg FDR vs. No correction:**
- Pros: Controls for false discoveries, increases reliability of results
- Cons: May miss some true associations due to increased stringency

The FDR approach was chosen because it strikes a good balance between discovering true age-associated CpG sites while controlling for false positives, which is particularly important in genomic studies where many features are tested simultaneously.