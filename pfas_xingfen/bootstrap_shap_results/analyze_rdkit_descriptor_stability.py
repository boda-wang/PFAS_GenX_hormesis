"""Summarize Bootstrap SHAP stability for the 18 retained RDKit descriptors.

The script uses the completed Bootstrap replicate file. It recreates the
primary CatBoost fit using the fixed Bootstrap feature order and writes only
tabular results; publication figures are intentionally not generated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
DATA_FILE = PROJECT_DIR / "modeling_dataset.xlsx"
FEATURE_FILE = HERE / "bootstrap_feature_order.json"
RAW_FILE = HERE / "bootstrap_replicates.csv"
OUTPUT_DIR = HERE / "rdkit_descriptor_stability"
TARGET_COLUMN = "Target_Hormesis"

CONTEXT_FEATURES = {
    "Apoptosis_Proliferation", "Behavior_Neuro", "Cardiovascular_Angiogenesis",
    "Endocrine_Reproductive", "Endocrine_Thyroid", "General_Fitness",
    "Hematotoxicity", "Immunotoxicity", "Metabolism_Energy",
    "Morphological_Development", "Oxidative_Stress", "Effect Measurement_LOO",
}

MODEL_PARAMETERS = {
    "iterations": 800,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
    "bagging_temperature": 0.8,
    "random_strength": 2,
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


def primary_shap_importance(model: CatBoostClassifier, x: pd.DataFrame, n_features: int) -> np.ndarray:
    values = shap.TreeExplainer(model).shap_values(x)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.shape[1] == n_features + 1:
        values = values[:, :n_features]
    if values.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return np.abs(values).mean(axis=0)


def main() -> None:
    features = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    data = pd.read_excel(DATA_FILE)
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(data[features]), columns=features)
    y = data[TARGET_COLUMN].astype(int)
    raw = pd.read_csv(RAW_FILE).sort_values("replicate").reset_index(drop=True)

    if len(raw) != 500:
        raise ValueError(f"Expected 500 completed replicates, found {len(raw)}")
    descriptors = [feature for feature in features if feature not in CONTEXT_FEATURES]
    if len(descriptors) != 18:
        raise ValueError(f"Expected 18 RDKit descriptors, found {len(descriptors)}")

    primary_model = CatBoostClassifier(**MODEL_PARAMETERS)
    primary_model.fit(x, y)
    primary_importance = pd.Series(primary_shap_importance(primary_model, x, len(features)), index=features).loc[descriptors]
    primary_rank = primary_importance.rank(ascending=False, method="average")

    bootstrap_importance = raw[[f"shap_abs__{feature}" for feature in descriptors]].copy()
    bootstrap_importance.columns = descriptors
    bootstrap_rank = bootstrap_importance.rank(axis=1, ascending=False, method="average")

    correlations = []
    for replicate, ranks in zip(raw["replicate"], bootstrap_rank.to_numpy()):
        rho, p_value = spearmanr(primary_rank.to_numpy(), ranks)
        correlations.append({"replicate": int(replicate), "spearman_rho": float(rho), "p_value": float(p_value)})
    correlation_table = pd.DataFrame(correlations)

    rows = []
    for descriptor in descriptors:
        ranks = bootstrap_rank[descriptor].to_numpy(float)
        values = bootstrap_importance[descriptor].to_numpy(float)
        rows.append({
            "descriptor": descriptor,
            "primary_abs_shap": float(primary_importance[descriptor]),
            "primary_rank": float(primary_rank[descriptor]),
            "bootstrap_mean_abs_shap": float(values.mean()),
            "shap_ci95_low": float(np.percentile(values, 2.5)),
            "shap_ci95_high": float(np.percentile(values, 97.5)),
            "bootstrap_median_rank": float(np.median(ranks)),
            "rank_ci95_low": float(np.percentile(ranks, 2.5)),
            "rank_ci95_high": float(np.percentile(ranks, 97.5)),
            "rank_iqr": float(np.percentile(ranks, 75) - np.percentile(ranks, 25)),
            "rank_sd": float(np.std(ranks, ddof=1)),
            "top3_frequency_within_18": float(np.mean(ranks <= 3)),
            "top5_frequency_within_18": float(np.mean(ranks <= 5)),
        })
    descriptor_table = pd.DataFrame(rows).sort_values("primary_rank").reset_index(drop=True)
    summary = {
        "n_replicates": int(len(correlation_table)),
        "mean_rho": float(correlation_table.spearman_rho.mean()),
        "median_rho": float(correlation_table.spearman_rho.median()),
        "sd_rho": float(correlation_table.spearman_rho.std(ddof=1)),
        "ci95_low": float(np.percentile(correlation_table.spearman_rho, 2.5)),
        "ci95_high": float(np.percentile(correlation_table.spearman_rho, 97.5)),
        "rank_scope": "18 RDKit descriptors only",
        "primary_rank_source": "CatBoost refit on the full reference dataset using the fixed Bootstrap feature order",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptor_table.to_csv(OUTPUT_DIR / "descriptor_stability.csv", index=False, encoding="utf-8-sig")
    correlation_table.to_csv(OUTPUT_DIR / "spearman_rank_correlations.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "spearman_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
