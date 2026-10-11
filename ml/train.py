"""Crop recommendation pipeline: clean, merge, label, compare models, fit the best.

Enhanced with calibration, drift detection, input validation, feature importances, 
rolling backtests, and deployment guardrails.

Run:
python crop_recommender.py --out-dir artifacts

Outputs (in --out-dir):
    rolling_backtest_comparison.csv  scores averaged across multiple time splits
    class_balance.csv                share of each recommended-crop class
    summary.json                     selected model, metadata, threshold checks
    feature_importance.csv           feature importances of the best base model
    crop_recommender.joblib          calibrated model, encoders, and metadata
"""
from __future__ import annotations

import argparse
import datetime
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, 
    top_k_accuracy_score, 
    recall_score, 
    precision_score, 
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constants and configuration
# --------------------------------------------------------------------------- #
YIELD_REQUIRED: list[str] = ["crop", "year", "season", "state", "area",
                             "production", "fertilizer", "pesticide", "yield"]
SOIL_REQUIRED: list[str] = ["state", "N", "P", "K", "pH"]
WEATHER_REQUIRED: list[str] = ["state", "year", "avg_temp_c",
                               "total_rainfall_mm", "avg_humidity_percent"]
NUMERIC_YIELD: list[str] = ["area", "production", "fertilizer", "pesticide", "yield"]

KEY: list[str] = ["crop", "state", "season", "year"]
GROUP: list[str] = ["state", "season", "year"]
FEATURE_COLS: list[str] = ["N", "P", "K", "pH", "avg_temp_c",
                           "total_rainfall_mm", "avg_humidity_percent"]
LEAKY_COLS: set[str] = {"area", "production", "yield", "fertilizer", "pesticide",
                        "fertilizer_per_ha", "pesticide_per_ha", "state", "year", "label"}
STATE_ALIASES: dict[str, str] = {}

BASELINE: str = "baseline_prior"
LOOKUP: str = "state_season_lookup"

YIELD_PATH: Path = Path(r"d:\CropAdvisory\data\raw\crop_yield.csv")
SOIL_PATH: Path = Path(r"d:\CropAdvisory\data\raw\state_soil_data.csv")
WEATHER_PATH: Path = Path(r"d:\CropAdvisory\data\raw\state_weather_data_1997_2020.csv")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_splits: int = 10
    top_k: int = 3
    backtest_years: list[int] = field(default_factory=lambda: [2013, 2014, 2015])
    min_crop_rows: int = 100
    min_crops_per_group: int = 3
    min_class_rows: int = 30
    min_classes: int = 8
    max_class_share: float = 0.5
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    class_weight: str | None = None
    run_state_cv: bool = True
    strict: bool = False
    
    # Deployment Thresholds
    min_top_k_acc: float = 0.80
    min_macro_f1: float = 0.50


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    years: pd.Series
    seasons: pd.Series
    label_encoder: LabelEncoder
    columns: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.label_encoder.classes_)


# --------------------------------------------------------------------------- #
# Cleaning & Merging
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

def _norm_text(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip()
             .str.replace(r"\s+", " ", regex=True).str.title())

def _norm_state(s: pd.Series) -> pd.Series:
    return _norm_text(s).replace(STATE_ALIASES)

def clean_yield(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, YIELD_REQUIRED, "df_yield")
    out = df[YIELD_REQUIRED].copy()
    for c in ("crop", "season"):
        out[c] = _norm_text(out[c])
    out["state"] = _norm_state(out["state"])
    for c in NUMERIC_YIELD + ["year"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=KEY + ["area", "production"]).drop_duplicates()
    out = out[(out["area"] > 0) & (out["production"] > 0)].copy()
    out["year"] = out["year"].astype(int)
    out = out.sort_values("area", ascending=False).drop_duplicates(KEY, keep="first").sort_values(KEY).reset_index(drop=True)
    out["yield"] = out["production"] / out["area"]
    out["fertilizer_per_ha"] = out["fertilizer"] / out["area"]
    out["pesticide_per_ha"] = out["pesticide"] / out["area"]
    return out

def clean_soil(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, SOIL_REQUIRED, "df_soil")
    out = df[SOIL_REQUIRED].copy()
    out["state"] = _norm_state(out["state"])
    for c in ("N", "P", "K", "pH"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    if out["state"].duplicated().any():
        raise ValueError("df_soil has duplicate states")
    return out

def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, WEATHER_REQUIRED, "df_weather")
    out = df[WEATHER_REQUIRED].copy()
    out["state"] = _norm_state(out["state"])
    for c in WEATHER_REQUIRED[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    if out.duplicated(["state", "year"]).any():
        raise ValueError("df_weather has duplicate (state, year) rows")
    return out

def merge_tables(y: pd.DataFrame, weather: pd.DataFrame, soil: pd.DataFrame) -> pd.DataFrame:
    merged = y.merge(weather, on=["state", "year"], how="left", validate="m:1") \
              .merge(soil, on="state", how="left", validate="m:1")
    unmatched = merged[FEATURE_COLS].isna().any(axis=1)
    if unmatched.any():
        raise ValueError(f"{int(unmatched.sum())} yield rows have no soil/weather match")
    return merged

# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def build_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    counts = df["crop"].value_counts()
    keep = counts[counts >= cfg.min_crop_rows].index
    d = df[df["crop"].isin(keep)].reset_index(drop=True)

    lo_q, hi_q = cfg.clip_quantiles
    train = d[d["year"] <= max(cfg.backtest_years)]
    stats = train.groupby("crop")["yield"].quantile([lo_q, 0.5, hi_q]).unstack()
    stats.columns = ["lo", "med", "hi"]
    d = d.join(stats, on="crop").dropna(subset=["med"]).reset_index(drop=True)
    
    d["rel_yield"] = d["yield"].clip(lower=d["lo"], upper=d["hi"]) / d["med"]
    d["n_candidates"] = d.groupby(GROUP)["crop"].transform("size")
    d = d[d["n_candidates"] >= cfg.min_crops_per_group].reset_index(drop=True)

    best_idx = d.groupby(GROUP)["rel_yield"].idxmax()
    best = d.loc[best_idx, GROUP + ["crop"]].rename(columns={"crop": "recommended_crop"})
    context = d.groupby(GROUP, as_index=False)[FEATURE_COLS].first()
    data = context.merge(best, on=GROUP, validate="1:1")

    wins = data["recommended_crop"].value_counts()
    data = data[~data["recommended_crop"].isin(wins[wins < cfg.min_class_rows].index)].reset_index(drop=True)
    return data

def build_dataset(data: pd.DataFrame) -> Dataset:
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(data["recommended_crop"].astype(str)), index=data.index, name="label")
    season = pd.get_dummies(data["season"].astype(str), prefix="season", dtype=int)
    X = pd.concat([data[FEATURE_COLS], season], axis=1)
    return Dataset(X=X, y=y, groups=data["state"].astype(str), years=data["year"],
                   seasons=data["season"].astype(str), label_encoder=le, columns=list(X.columns))

# --------------------------------------------------------------------------- #
# Models, Scoring & Backtesting
# --------------------------------------------------------------------------- #
def make_models(cfg: Config) -> dict[str, Any]:
    cw = cfg.class_weight
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight=cw, n_jobs=-1, random_state=cfg.seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight=cw, n_jobs=-1, random_state=cfg.seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight=cw, random_state=cfg.seed),
        "lightgbm": LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight=cw, random_state=cfg.seed, n_jobs=-1, verbose=-1),
    }

def _full_proba(model: Any, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    p = np.zeros((len(X), n_classes))
    p[:, model.classes_] = model.predict_proba(X)
    return p

def _metrics(y_true: pd.Series, proba: np.ndarray, cfg: Config) -> dict[str, float]:
    labels = np.arange(proba.shape[1])
    y_pred = proba.argmax(axis=1)
    
    metrics = {
        "top1": top_k_accuracy_score(y_true, proba, k=1, labels=labels),
        f"top{cfg.top_k}": top_k_accuracy_score(y_true, proba, k=cfg.top_k, labels=labels),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    
    # One-vs-Rest ROC-AUC is standard for multiclass
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["roc_auc"] = np.nan
        
    return metrics

def lookup_proba(ds: Dataset, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    n = ds.n_classes
    tr = pd.DataFrame({"state": ds.groups[train].to_numpy(), "season": ds.seasons[train].to_numpy(), "label": ds.y[train].to_numpy()})
    def counts(by: list[str]) -> pd.DataFrame:
        return tr.groupby(by + ["label"]).size().unstack(fill_value=0).reindex(columns=range(n), fill_value=0)
    by_state_season = counts(["state", "season"])
    by_season = counts(["season"])
    overall = np.bincount(tr["label"], minlength=n).astype(float)
    
    proba = np.zeros((int(test.sum()), n))
    for i, (st, se) in enumerate(zip(ds.groups[test], ds.seasons[test])):
        if (st, se) in by_state_season.index: row = by_state_season.loc[(st, se)].to_numpy(dtype=float)
        elif se in by_season.index: row = by_season.loc[se].to_numpy(dtype=float)
        else: row = overall
        proba[i] = row / row.sum()
    return proba + 1e-6 * overall / overall.sum()

def rolling_time_backtest(models: dict[str, Any], ds: Dataset, cfg: Config) -> pd.DataFrame:
    n = ds.n_classes
    metrics_list = []
    
    for split_yr in cfg.backtest_years:
        train = (ds.years <= split_yr).to_numpy()
        test = (ds.years > split_yr).to_numpy()
        if not train.any() or not test.any(): continue
        
        y_test = ds.y[test]
        probas = {}
        for name, model in models.items():
            m = clone(model).fit(ds.X[train], ds.y[train])
            probas[name] = _full_proba(m, ds.X[test], n)
        probas[LOOKUP] = lookup_proba(ds, train, test)
        
        df_metrics = pd.DataFrame({name: _metrics(y_test, p, cfg) for name, p in probas.items()}).T
        metrics_list.append(df_metrics)
        
    avg_results = pd.concat(metrics_list).groupby(level=0).mean()
    key = f"top{cfg.top_k}"
    return avg_results.sort_values([key, "macro_f1"], ascending=False)

def extract_feature_importance(model: Any, columns: list[str]) -> pd.Series | None:
    est = model.named_steps[-1] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=columns).sort_values(ascending=False)
    elif hasattr(est, "coef_"):
        return pd.Series(np.abs(est.coef_).mean(axis=0), index=columns).sort_values(ascending=False)
    return None

# --------------------------------------------------------------------------- #
# Persistence and Inference
# --------------------------------------------------------------------------- #
def save_artifact(model: Any, ds: Dataset, model_name: str, path: Path, cfg: Config) -> None:
    joblib.dump({
        "model": model,
        "label_encoder": ds.label_encoder,
        "features": ds.columns,
        "model_name": model_name,
        "known_states": ds.groups.unique().tolist(),
        "feature_stats": ds.X[FEATURE_COLS].agg(['min', 'max', 'mean', 'std']).to_dict(),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_period_end": max(cfg.backtest_years)
    }, path)

def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)

def recommend_top_k(artifact: dict[str, Any], features: pd.DataFrame, k: int = 3) -> list[dict[str, Any]]:
    """Predicts crops with confidence scoring, anomaly detection, and validation."""
    model = artifact["model"]
    le: LabelEncoder = artifact["label_encoder"]
    stats = artifact["feature_stats"]
    known_states = set(artifact["known_states"])
    results = []

    for idx, row in features.iterrows():
        warnings_list = []
        
        # 1. Unknown-state protection
        if "state" in row and row["state"] not in known_states:
            raise ValueError(f"State '{row['state']}' is unknown. Supported states: {known_states}")
            
        # 2. Input Validation
        if not (0 <= row.get("pH", 7) <= 14): warnings_list.append(f"pH {row.get('pH')} out of range (0-14).")
        if not (0 <= row.get("avg_humidity_percent", 50) <= 100): warnings_list.append("Humidity out of range (0-100%).")
        for col in ["N", "P", "K", "total_rainfall_mm"]:
            if row.get(col, 0) < 0: warnings_list.append(f"{col} cannot be negative.")

        # 3. Drift Detection
        for col in FEATURE_COLS:
            val = row.get(col)
            if val is not None and col in stats:
                c_min, c_max, c_mean, c_std = stats[col]["min"], stats[col]["max"], stats[col]["mean"], stats[col]["std"]
                if val < c_min or val > c_max:
                    warnings_list.append(f"{col} value ({val:.2f}) is outside historical training range [{c_min:.2f}, {c_max:.2f}].")
                elif abs(val - c_mean) > 3 * c_std:
                    warnings_list.append(f"{col} value ({val:.2f}) is > 3 std devs from training mean.")
                    
        # Predict
        row_df = pd.DataFrame([row])[artifact["features"]]
        proba = model.predict_proba(row_df)[0]
        top_idx = np.argsort(proba)[::-1][:k]
        
        crop_names = le.inverse_transform(model.classes_[top_idx])
        confidences = proba[top_idx]
        
        recs = [{"crop": name, "probability": float(p)} for name, p in zip(crop_names, confidences)]
        
        results.append({
            "recommendations": recs,
            "confidence_score": float(confidences[0]),
            "low_confidence": bool(confidences[0] < 0.40),
            "warnings": warnings_list
        })
        
    return results

# --------------------------------------------------------------------------- #
# Pipeline Execution
# --------------------------------------------------------------------------- #
def run(df_yield: pd.DataFrame, df_soil: pd.DataFrame, df_weather: pd.DataFrame, out_dir: Path, cfg: Config) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"top{cfg.top_k}"

    merged = merge_tables(clean_yield(df_yield), clean_weather(df_weather), clean_soil(df_soil))
    data = build_labels(merged, cfg)
    ds = build_dataset(data)
    
    (data["recommended_crop"].value_counts(normalize=True).rename("share").to_csv(out_dir / "class_balance.csv"))

    # Select via Rolling Backtests
    models = make_models(cfg)
    results = rolling_time_backtest(models, ds, cfg)
    results.to_csv(out_dir / "rolling_backtest_comparison.csv")
    
    candidates = results.drop(index=[BASELINE, LOOKUP], errors="ignore")
    best_name = candidates.sort_values([key, "macro_f1"], ascending=False).index[0]
    best_scores = results.loc[best_name]

    # Deployment Threshold Checks
    if best_scores[key] < cfg.min_top_k_acc:
        raise RuntimeError(f"Deployment rejected: Top-{cfg.top_k} accuracy {best_scores[key]:.3f} < {cfg.min_top_k_acc}")
    if best_scores["macro_f1"] < cfg.min_macro_f1:
        raise RuntimeError(f"Deployment rejected: Macro F1 {best_scores['macro_f1']:.3f} < {cfg.min_macro_f1}")

    # Final Fit & Calibration
    base_final_model = clone(models[best_name]).fit(ds.X, ds.y)
    
    fi = extract_feature_importance(base_final_model, ds.columns)
    if fi is not None: fi.to_csv(out_dir / "feature_importance.csv")

    calibrated_model = CalibratedClassifierCV(estimator=base_final_model, cv=5, method='isotonic')
    calibrated_model.fit(ds.X, ds.y)
    
    save_artifact(calibrated_model, ds, best_name, out_dir / "crop_recommender.joblib", cfg)

    summary = {
        "selected_model": best_name,
        "selection_basis": f"Rolling backtests over years {cfg.backtest_years}",
        "n_rows": int(len(ds.X)),
        "n_classes": int(ds.n_classes),
        "backtest_avg_scores": {k: float(v) for k, v in results.loc[best_name].items()},
        "lookup_baseline_scores": {k: float(v) for k, v in results.loc[LOOKUP].items()} if LOOKUP in results.index else {},
        "meets_deployment_thresholds": True,
        "calibrated": True
    }
    
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"results": results, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the crop recommender.")
    p.add_argument("--out-dir", default=Path("artifacts"), type=Path)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = Config(seed=args.seed, class_weight=None if args.class_weight == "none" else "balanced")
    
    try:
        out = run(pd.read_csv(YIELD_PATH), pd.read_csv(SOIL_PATH), pd.read_csv(WEATHER_PATH), args.out_dir, cfg)
        summary = out["summary"]

        print(f"Rolling Backtest Scores (Avg across splits {cfg.backtest_years}):")
        print(out["results"].round(3).to_string())
        print(f"\nSelected & Calibrated model: {summary['selected_model']}")
        print("Scores:", {k: round(v, 3) for k, v in summary["backtest_avg_scores"].items()})
        print("Model passed deployment thresholds and artifact is ready for inference.")

    except Exception as e:
        print(f"Pipeline Failed: {str(e)}")


if __name__ == "__main__":
    main()