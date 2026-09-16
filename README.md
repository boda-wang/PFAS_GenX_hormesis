# GenX hormesis prediction with interpretable machine learning

This repository contains the data, code, and compact numerical outputs supporting the GenX hormesis prediction analysis in zebrafish.

## Main files

- `pfas_xingfen/curated_dataset.xlsx`: curated source dataset.
- `pfas_xingfen/dataset_with_biological_categories.xlsx`: intermediate dataset after biological endpoint mapping.
- `pfas_xingfen/modeling_dataset.xlsx`: final training dataset (263 records).
- `pfas_xingfen/train_main_model.py`: six-model benchmarking, five-fold out-of-fold evaluation, and final CatBoost model training.
- `pfas_xingfen/main_model_results/`: saved CatBoost model, selected 30-variable order, performance tables, and run metadata.
- `pfas_xingfen/bootstrap_shap_analysis.py`: 500-replicate Bootstrap SHAP analysis.
- `pfas_xingfen/bootstrap_shap_results/`: Bootstrap raw results, descriptor-stability summaries, and a compact overview figure.

The optional preprocessing scripts `map_biological_endpoints.py` and `data_process_Leave_One.py` document the conversion from the curated source data to the final modelling dataset.

## Run

From the `pfas_xingfen` directory:

```bash
python train_main_model.py
python bootstrap_shap_analysis.py
python bootstrap_shap_results/analyze_rdkit_descriptor_stability.py
```

The Bootstrap script resumes from the saved replicate results unless `--fresh` is supplied.
