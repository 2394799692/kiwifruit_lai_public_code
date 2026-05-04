#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-strategy ablation for single-plant LAI estimation.

This script compares LAI modeling performance among feature tables generated
under different ROI extraction strategies, such as fixed-radius ROIs and dynamic
P/Q settings. It does not re-extract raster features; instead, it evaluates the
feature CSV/XLSX files produced by the extraction scripts.

Required input
--------------
A strategy configuration CSV with columns:
    Strategy, Feature_Table
Example:
    Rfix_1.50, outputs/features/features_Rfix_1p50.csv
    Rfix_3.00, outputs/features/features_Rfix_3p00.csv
    P90_Q70,   outputs/features/features_P90_Q70.csv

Each feature table must contain LAI_Observed and candidate feature columns.
Tree_ID or ID is optional but recommended for grouped cross-validation.

Outputs
-------
- roi_strategy_ablation_metrics.csv
- roi_strategy_ablation_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


DEFAULT_STRATEGY_CONFIG = "example_roi_strategy_config.csv"
DEFAULT_OUTPUT_DIR = "outputs/roi_ablation"

VI_24 = [
    "NDVI", "RVI", "DVI", "TNDVI", "RDVI", "NGRDI", "RI", "MSR", "MSAVI", "TVI",
    "WDRVI", "GRVI", "NLI", "MTVI2", "CRI", "GNDVI", "OSAVI", "SAVI", "PVI",
    "MCARI", "RGRI", "NDRE", "TCARI_OSAVI", "RECI",
]
TEX_8 = [
    "TexNIR_Contrast_Avg", "TexNIR_Dissimilarity_Avg", "TexNIR_Homogeneity_Avg",
    "TexNIR_Energy_Avg", "TexNIR_Correlation_Avg", "TexNIR_ASM_Avg",
    "TexNIR_Entropy_Avg", "TexNIR_Mean_Avg",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RPD": float(np.std(y_true, ddof=1) / rmse) if rmse > 0 and len(y_true) > 1 else np.nan,
    }


def get_feature_pool(df: pd.DataFrame) -> List[str]:
    tex_cols = [c for c in TEX_8 if c in df.columns]
    if not tex_cols:
        tex_cols = [c for c in df.columns if str(c).startswith("TexNIR_")]
    return [c for c in VI_24 + tex_cols if c in df.columns]


def top_k_by_abs_corr(df: pd.DataFrame, features: List[str], k: int) -> List[str]:
    pairs = []
    y = pd.to_numeric(df["LAI_Observed"], errors="coerce")
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 8 or x[mask].nunique() <= 1:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(r):
            pairs.append((f, abs(float(r))))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [f for f, _ in pairs[: min(k, len(pairs))]]


def make_cv(df: pd.DataFrame, n_splits: int, grouped: bool):
    if grouped and ("Tree_ID" in df.columns or "ID" in df.columns):
        group_col = "Tree_ID" if "Tree_ID" in df.columns else "ID"
        groups = df[group_col].astype(str).to_numpy()
        if len(np.unique(groups)) >= n_splits:
            return GroupKFold(n_splits=n_splits), groups, "GroupKFold", group_col
    return KFold(n_splits=n_splits, shuffle=True, random_state=42), None, "KFold", None


def evaluate_strategy(strategy: str, table_path: str | Path, top_k: int, n_splits: int, grouped: bool):
    df = read_table(table_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower().eq("success")].copy()
    if "LAI_Observed" not in df.columns:
        raise ValueError(f"{table_path} must contain LAI_Observed.")

    features = get_feature_pool(df)
    selected = top_k_by_abs_corr(df, features, top_k)
    if not selected:
        raise ValueError(f"No valid features in {table_path}.")
    df_eval = df.dropna(subset=selected + ["LAI_Observed"]).copy()
    if len(df_eval) < max(n_splits, 10):
        raise ValueError(f"Too few complete samples in {table_path}: {len(df_eval)}")

    X = df_eval[selected].to_numpy(dtype=float)
    y = df_eval["LAI_Observed"].to_numpy(dtype=float)
    cv, groups, cv_name, group_col = make_cv(df_eval, n_splits=n_splits, grouped=grouped)
    model = GradientBoostingRegressor(random_state=42)
    if groups is None:
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    else:
        pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)

    row = {
        "Strategy": strategy,
        "Feature_Table": str(table_path),
        "Sample_Size": int(len(y)),
        "Num_Features": int(len(selected)),
        "Selected_Features": ";".join(selected),
        "CV": cv_name,
        **metrics(y, pred),
    }

    pred_df = pd.DataFrame({
        "Strategy": strategy,
        "Tree_ID": df_eval.get("Tree_ID", df_eval.get("ID", pd.Series(np.arange(len(df_eval))))).values,
        "Date": df_eval.get("Date", pd.Series([""] * len(df_eval))).values,
        "Observed_LAI": y,
        "Predicted_LAI": pred,
        "Residual": y - pred,
    })
    return row, pred_df


def run_ablation(strategy_config: str, output_dir: str, top_k: int, n_splits: int, grouped: bool) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = pd.read_csv(strategy_config)
    required = {"Strategy", "Feature_Table"}
    if not required.issubset(set(cfg.columns)):
        raise ValueError("Strategy config must contain Strategy and Feature_Table columns.")

    rows = []
    preds = []
    for _, r in cfg.iterrows():
        strategy = str(r["Strategy"])
        table = str(r["Feature_Table"])
        row, pred = evaluate_strategy(strategy, table, top_k=top_k, n_splits=n_splits, grouped=grouped)
        rows.append(row)
        preds.append(pred)
        print(f"{strategy:20s} | N={row['Sample_Size']:3d} | R2={row['R2']:.3f} | RMSE={row['RMSE']:.3f}")

    pd.DataFrame(rows).sort_values("R2", ascending=False).to_csv(
        out_dir / "roi_strategy_ablation_metrics.csv", index=False, encoding="utf-8-sig"
    )
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(
            out_dir / "roi_strategy_ablation_predictions.csv", index=False, encoding="utf-8-sig"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LAI model performance across ROI feature tables.")
    parser.add_argument("--strategy-config", default=DEFAULT_STRATEGY_CONFIG, help="CSV with Strategy and Feature_Table columns.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--top-k", type=int, default=29, help="Number of top-ranked features used for each strategy.")
    parser.add_argument("--n-splits", type=int, default=10, help="CV folds.")
    parser.add_argument("--no-grouped", action="store_true", help="Use shuffled KFold instead of grouped CV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ablation(
        strategy_config=args.strategy_config,
        output_dir=args.output_dir,
        top_k=args.top_k,
        n_splits=args.n_splits,
        grouped=not args.no_grouped,
    )


if __name__ == "__main__":
    main()
