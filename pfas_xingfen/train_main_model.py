"""Train and internally validate the CatBoost hormesis-classification workflow.

Input: the final processed dataset of 263 records.
Outputs: model comparison metrics, out-of-fold predictions, selected features,
and the fitted master model. Figure generation is intentionally omitted because
the publication figures are archived separately.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "modeling_dataset.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "main_model_results"
TARGET_COLUMN = "Target_Hormesis"
N_SELECTED_FEATURES = 30
RANDOM_SEED = 42

# Fixed order retained for the 11 pre-specified biological endpoint covariates.
MANDATORY_FEATURES = [
    "Oxidative_Stress",
    "Morphological_Development",
    "Metabolism_Energy",
    "Immunotoxicity",
    "Hematotoxicity",
    "General_Fitness",
    "Endocrine_Thyroid",
    "Endocrine_Reproductive",
    "Cardiovascular_Angiogenesis",
    "Behavior_Neuro",
    "Apoptosis_Proliferation",
]


def optimal_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose the probability threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = 2 * precision * recall / (precision + recall + 1e-10)
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]) if best_index < len(thresholds) else 0.5


def prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed dataset and median-impute numerical variables."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}")

    data = pd.read_excel(DATA_FILE)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    y = data[TARGET_COLUMN].astype(int)
    excluded = [column for column in ("Chemical Name", "SMILES", TARGET_COLUMN) if column in data.columns]
    x = data.drop(columns=excluded).select_dtypes(include=[np.number])
    x = x.loc[:, x.std() != 0]
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=x.columns)
    return x, y


def select_features(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Retain endpoint covariates and select remaining variables by CatBoost importance."""
    mandatory = [feature for feature in MANDATORY_FEATURES if feature in x.columns]
    if len(mandatory) != len(MANDATORY_FEATURES):
        missing = sorted(set(MANDATORY_FEATURES) - set(mandatory))
        raise ValueError(f"Missing pre-specified endpoint covariates: {missing}")

    selector = CatBoostClassifier(
        iterations=300,
        depth=4,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    selector.fit(x, y)
    importance = pd.Series(selector.get_feature_importance(), index=x.columns)
    candidates = importance.drop(labels=mandatory).sort_values(ascending=False, kind="mergesort")
    selected = mandatory + candidates.head(N_SELECTED_FEATURES - len(mandatory)).index.tolist()

    if len(selected) != N_SELECTED_FEATURES:
        raise RuntimeError(f"Expected {N_SELECTED_FEATURES} selected variables; obtained {len(selected)}")
    return x[selected], selected


def build_models(y: pd.Series) -> dict[str, object]:
    """Define the six candidate algorithms used for benchmarking."""
    positive_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() else 1.0
    return {
        "LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2", C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=RANDOM_SEED
            ),
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(scale_pos_weight=positive_weight, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(class_weight="balanced", random_state=RANDOM_SEED, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostClassifier(
            iterations=800,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=10,
            bagging_temperature=0.8,
            random_strength=2,
            auto_class_weights="Balanced",
            early_stopping_rounds=100,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def benchmark_models(models: dict[str, object], x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate pooled five-fold out-of-fold predictions for each candidate model."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows: list[dict[str, float | str]] = []
    probabilities: dict[str, np.ndarray] = {}

    for name, model in models.items():
        oof_probability = np.zeros(len(x))
        for train_index, validation_index in splitter.split(x, y):
            model.fit(x.iloc[train_index], y.iloc[train_index])
            oof_probability[validation_index] = model.predict_proba(x.iloc[validation_index])[:, 1]

        threshold = optimal_f1_threshold(y, oof_probability)
        oof_prediction = (oof_probability >= threshold).astype(int)
        probabilities[name] = oof_probability
        rows.append(
            {
                "Model": name,
                "AUC": roc_auc_score(y, oof_probability),
                "Best_Threshold": threshold,
                "F1_Score": f1_score(y, oof_prediction),
                "Accuracy": accuracy_score(y, oof_prediction),
            }
        )

    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True), probabilities


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = prepare_data()
    x_selected, selected_features = select_features(x, y)
    models = build_models(y)
    comparison, oof_probabilities = benchmark_models(models, x_selected, y)

    best_model_name = str(comparison.loc[0, "Model"])
    best_threshold = float(comparison.loc[0, "Best_Threshold"])
    best_oof_probability = oof_probabilities[best_model_name]
    best_oof_prediction = (best_oof_probability >= best_threshold).astype(int)

    comparison.to_excel(OUTPUT_DIR / "6Models_Performance_Comparison.xlsx", index=False)
    pd.DataFrame(
        {
            "真实标签": y,
            "模型预测结果": best_oof_prediction,
            "置信度(纯数字)": best_oof_probability,
            "是否预测正确": y.to_numpy() == best_oof_prediction,
        }
    ).sort_values("置信度(纯数字)", ascending=False).to_excel(
        OUTPUT_DIR / "report_oof_end_BestModel.xlsx", index=False
    )

    master_model = models[best_model_name]
    master_model.fit(x_selected, y)
    joblib.dump(master_model, OUTPUT_DIR / f"master_{best_model_name}.pkl")
    (OUTPUT_DIR / "saved_features.json").write_text(json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8")

    training_probability = master_model.predict_proba(x_selected)[:, 1]
    training_prediction = (training_probability >= best_threshold).astype(int)
    metadata = {
        "n_records": int(len(y)),
        "n_selected_features": len(selected_features),
        "best_model": best_model_name,
        "pooled_oof_auc": float(comparison.loc[0, "AUC"]),
        "pooled_oof_accuracy": float(comparison.loc[0, "Accuracy"]),
        "best_threshold": best_threshold,
        "master_training_auc": float(roc_auc_score(y, training_probability)),
        "master_training_accuracy_at_oof_threshold": float(accuracy_score(y, training_prediction)),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
