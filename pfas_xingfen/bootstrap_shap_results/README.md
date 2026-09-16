# Bootstrap SHAP stability analysis

This folder contains the data outputs for the 500-replicate Bootstrap SHAP analysis reported in the Supporting Information. The analysis uses a fixed set of 30 variables: 11 biological endpoint covariates, `Effect Measurement_LOO`, and 18 RDKit descriptors.

## Method summary

For each replicate, 263 observations were sampled with replacement, a CatBoost model was retrained with fixed model settings, and SHAP values were calculated on the complete 263-record reference dataset. The analysis evaluates SHAP stability conditional on this fixed variable set.

## Contents

- `bootstrap_feature_order.json`: fixed 30-variable order used for the Bootstrap workflow.
- `bootstrap_metadata.json`: dataset size, class counts, seed, and model settings.
- `bootstrap_replicates.csv`: mean absolute and signed SHAP values from all 500 refitted models; this is the primary Bootstrap result file.
- `shap_bootstrap_summary.csv`: 30-variable summary derived from the replicate results.
- `analyze_rdkit_descriptor_stability.py`: derives the descriptor-only rank and contribution stability statistics used in the Supporting Information.
- `rdkit_descriptor_stability/`: 18-descriptor stability table, replicate-wise Spearman correlations, and summary statistics.
- `bootstrap_shap_stability_summary.png`: compact visual overview of the 18 RDKit descriptors. Points are bootstrap mean absolute SHAP values and horizontal bars are empirical 95% bootstrap intervals.
- `bootstrap_shap_stability_summary.json`: machine-readable overview of the number of refits, Spearman ranking-stability statistics, and the five largest bootstrap mean absolute SHAP values.

## Reproduction

From the `pfas_xingfen/` directory:

```bash
python bootstrap_shap_analysis.py
python bootstrap_shap_results/analyze_rdkit_descriptor_stability.py
```

The first script resumes from `bootstrap_replicates.csv` when available. Use `--fresh` only to intentionally replace the 500 Bootstrap replicates.

To recreate the compact overview figure:

```bash
python bootstrap_shap_results/generate_bootstrap_summary.py
```
