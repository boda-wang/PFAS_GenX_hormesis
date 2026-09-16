"""Bootstrap stability analysis for the fixed 30-feature CatBoost model.

This script is independent of the main-model training script. It does not change any existing model,
report, or dataset. Each bootstrap replicate fits a fresh CatBoost model using
the same 30 feature columns and the same hyperparameters, then evaluates SHAP
values on the same full reference dataset so the replicates are comparable.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DATA = SCRIPT_DIR / "modeling_dataset.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "bootstrap_shap_results"
SOURCE_FEATURES = OUTPUT_DIR / "bootstrap_feature_order.json"
RAW_RESULTS = OUTPUT_DIR / "bootstrap_replicates.csv"
SUMMARY_RESULTS = OUTPUT_DIR / "shap_bootstrap_summary.csv"
METADATA = OUTPUT_DIR / "bootstrap_metadata.json"


MODEL_PARAMS = {
    "iterations": 800,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
    "bagging_temperature": 0.8,
    "random_strength": 2,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "allow_writing_files": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap SHAP stability analysis")
    parser.add_argument("--replicates", type=int, default=500, help="Number of bootstrap fits")
    parser.add_argument("--seed", type=int, default=20260907, help="Random seed")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save progress every N fits")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing checkpoint and start again")
    return parser.parse_args()


def load_fixed_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Data file not found: {SOURCE_DATA}")
    if not SOURCE_FEATURES.exists():
        raise FileNotFoundError(f"Feature list not found: {SOURCE_FEATURES}")

    df = pd.read_excel(SOURCE_DATA)
    features = json.loads(SOURCE_FEATURES.read_text(encoding="utf-8"))
    target_col = "Target_Hormesis"
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Fixed features missing from dataset: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    x = df[features].copy()
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=features)
    y = df[target_col].astype(int).reset_index(drop=True)
    return x.reset_index(drop=True), y, features


def shap_importance(model: CatBoostClassifier, x_reference: pd.DataFrame, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_reference)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    if values.shape[1] == feature_count + 1:
        values = values[:, :feature_count]
    if values.shape[1] != feature_count:
        raise ValueError(f"SHAP feature count {values.shape[1]} != expected {feature_count}")
    return values.mean(axis=0), np.abs(values).mean(axis=0)


def summarize(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    importance_columns = [f"shap_abs__{feature}" for feature in features]
    signed_columns = [f"shap_signed__{feature}" for feature in features]
    importance = raw[importance_columns].to_numpy(dtype=float)
    signed = raw[signed_columns].to_numpy(dtype=float)
    ranks = pd.DataFrame(-importance, columns=features).rank(axis=1, method="average").to_numpy()

    rows = []
    for index, feature in enumerate(features):
        values = importance[:, index]
        feature_ranks = ranks[:, index]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": values.mean(),
                "median_abs_shap": np.median(values),
                "ci95_low": np.percentile(values, 2.5),
                "ci95_high": np.percentile(values, 97.5),
                "mean_signed_shap": signed[:, index].mean(),
                "mean_rank": feature_ranks.mean(),
                "median_rank": np.median(feature_ranks),
                "top5_frequency": np.mean(feature_ranks <= 5),
                "top10_frequency": np.mean(feature_ranks <= 10),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_reference, y, features = load_fixed_data()
    n_samples = len(x_reference)

    existing = pd.DataFrame()
    if RAW_RESULTS.exists() and not args.fresh:
        existing = pd.read_csv(RAW_RESULTS)
        if not existing.empty:
            existing = existing[existing["replicate"] < args.replicates].copy()

    completed = set(existing["replicate"].astype(int)) if not existing.empty else set()
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = [rng.integers(0, n_samples, size=n_samples) for _ in range(args.replicates)]
    pending = [i for i in range(args.replicates) if i not in completed]

    metadata = {
        "source_data": "../modeling_dataset.xlsx",
        "source_features": "bootstrap_feature_order.json",
        "fixed_feature_count": len(features),
        "fixed_features": features,
        "sample_count": n_samples,
        "positive_count": int(y.sum()),
        "negative_count": int((y == 0).sum()),
        "replicates_requested": args.replicates,
        "seed": args.seed,
        "evaluation_reference": "full fixed dataset after median imputation",
        "model_parameters": MODEL_PARAMS,
        "status": "running",
    }
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Fixed features: {len(features)} | samples: {n_samples}")
    print(f"Bootstrap replicates requested: {args.replicates} | pending: {len(pending)}")
    rows = existing.to_dict("records") if not existing.empty else []

    for count, replicate in enumerate(pending, start=1):
        indices = bootstrap_indices[replicate]
        y_boot = y.iloc[indices].reset_index(drop=True)
        if y_boot.nunique() < 2:
            print(f"Skipping replicate {replicate}: bootstrap sample has one class")
            continue

        model = CatBoostClassifier(**MODEL_PARAMS, random_seed=args.seed + replicate)
        model.fit(x_reference.iloc[indices].reset_index(drop=True), y_boot)
        signed, absolute = shap_importance(model, x_reference, len(features))
        row = {
            "replicate": replicate,
            "n_unique_samples": int(np.unique(indices).size),
            "n_oob_samples": int(n_samples - np.unique(indices).size),
        }
        row.update({f"shap_signed__{feature}": float(value) for feature, value in zip(features, signed)})
        row.update({f"shap_abs__{feature}": float(value) for feature, value in zip(features, absolute)})
        rows.append(row)

        if count == 1 or count % args.checkpoint_every == 0 or count == len(pending):
            checkpoint = pd.DataFrame(rows).sort_values("replicate")
            checkpoint.to_csv(RAW_RESULTS, index=False)
            summary = summarize(checkpoint, features)
            summary.to_csv(SUMMARY_RESULTS, index=False)
            print(f"Completed {count}/{len(pending)} pending replicates (replicate id {replicate})")

    raw = pd.DataFrame(rows).sort_values("replicate") if rows else pd.DataFrame()
    if raw.empty:
        raise RuntimeError("No bootstrap replicate completed")
    raw.to_csv(RAW_RESULTS, index=False)
    summary = summarize(raw, features)
    summary.to_csv(SUMMARY_RESULTS, index=False)

    metadata["completed_replicates"] = int(len(raw))
    metadata["status"] = "complete" if len(raw) >= args.replicates else "partial"
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed replicates: {len(raw)}/{args.replicates}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
