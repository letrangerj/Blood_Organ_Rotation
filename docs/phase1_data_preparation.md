# Phase 1: Data Preparation & Exploration

## 0. Purpose of the script
This script performs initial data loading, validation, quality control, transformation, and summary statistics generation for the aging clock project. It takes the raw methylation matrix and metadata files and prepares them for downstream analysis.

## 1. Input
- `data/overlap_cfDNA.tsv`: Methylation matrix with CpG sites as rows and samples as columns, including genomic coordinates (chr, start, end, MD)
- `data/Metadata_PY_104.csv`: Sample metadata file with Sample ID, Age, Gender information

## 2. Output
- `result/1_preprocessed_data.csv`: Transposed methylation matrix with samples as rows and CpG sites as columns, merged with age information
- `result/1_summary.txt`: Summary statistics including number of samples, CpG sites, age range, mean age, missing value statistics, and out-of-range methylation values

## 3. Method used
- Loaded data from TSV and CSV files
- Validated sample ID correspondence between files
- Performed quality control checks for missing values, methylation value ranges (0-1)
- Examined age distribution and gender distribution
- Transposed the methylation matrix so samples become rows and CpG sites become columns
- Merged methylation data with age information

```python
# Transpose methylation matrix: rows = samples, columns = CpG sites
methylation_df = methylation_df.set_index(['chr', 'start', 'end', 'MD'])
methylation_t = methylation_df.T

# Merge with metadata based on sample IDs
merged_df = pd.concat([metadata_filtered, methylation_t], axis=1)
```

## 4. Summary of the results
- Processed 104 samples and 1,234 CpG sites
- Age range: 20-84 years with mean age of 50.42 years
- 5,153 missing values (4.02% of total data)
- 0 out-of-range methylation values
- Successfully merged methylation and age data for downstream analysis

## 5. Worries/Suspicious places
- Relatively high percentage of missing data (4.02%) may impact downstream analyses
- The actual number of samples in the results is 104 instead of 105 as mentioned in the detailed plan
- Need to verify if the transposition and merging handled all samples correctly before proceeding to Phase 2
