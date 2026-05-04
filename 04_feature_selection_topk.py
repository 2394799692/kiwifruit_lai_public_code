#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature subset learning curve and Top-K selection for LAI modeling.

The script ranks candidate spectral vegetation indices and NIR texture features,
then evaluates progressively larger feature subsets using cross-validation. It is
intended to reproduce the feature-dimension learning-curve step used before final
model comparison.

Required input
--------------
A cleaned feature table with LAI_Observed and candidate features.
Tree_ID or ID is optional but recommended for grouped cross-validation.

Outputs
-------
- feature_ranking.csv
- feature_subset_learning_curve.csv
- selected_features_topK.txt
- selected_features_topK.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


DEFAULT_INPUT = "outputs/features/single_plant_lai_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/feature_selection"

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


def build_candidate_features(df: pd.DataFrame, include_bands: bool = False) -> List[str]:
    features = [c for c in VI_24 + TEX_8 if c in df.columns]
    if include_bands:
        features += [c for c in ["Green_Refl", "Red_Refl", "RedEdge_Refl", "NIR_Refl",
                                "Green_Std", "Red_Std", "RedEdge_Std", "NIR_Std"] if c in df.columns]
    return list(dict.fromkeys(features))


def rank_features(df: pd.DataFrame, features: List[str], method: str) -> pd.DataFrame:
    clean = df[features + ["LAI_Observed"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10:
        raise ValueError("Too few complete samples for feature ranking.")

    if method == "gbdt_importance":
        model = GradientBoostingRegressor(random_state=42)
        model.fit(clean[features].to_numpy(dtype=float), clean["LAI_Observed"].to_numpy(dtype=float))
        scores = model.feature_importances_
    else:
        scores = []
        y = clean["LAI_Observed"].to_numpy(dtype=float)
        for f in features:
            x = clean[f].to_numpy(dtype=float)
            if np.std(x) <= 0:
                scores.append(0.0)
            else:
                scores.append(abs(float(np.corrcoef(x, y)[0, 1])))
        scores = np.array(scores)

    ranking = pd.DataFrame({"Feature": features, "Score": scores})
    ranking = ranking.replace([np.inf, -np.inf], np.nan).dropna(subset=["Score"])
    ranking.sort_values("Score", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["Rank", "Feature", "Score"]]


def make_cv(df: pd.DataFrame, n_splits: int, grouped: bool):
    if grouped and ("Tree_ID" in df.columns or "ID" in df.columns):
        group_col = "Tree_ID" if "Tree_ID" in df.columns else "ID"
        groups = df[group_col].astype(str).to_numpy()
        unique_n = len(np.unique(groups))
        if unique_n >= n_splits:
            return GroupKFold(n_splits=n_splits), groups, "GroupKFold"
    return KFold(n_splits=n_splits, shuffle=True, random_state=42), None, "KFold"


def evaluate_subsets(df: pd.DataFrame, ranked_features: List[str], max_k: int, n_splits: int, grouped: bool) -> pd.DataFrame:
    rows = []
    for k in range(1, min(max_k, len(ranked_features)) + 1):
        feats = ranked_features[:k]
        sub = df[feats + ["LAI_Observed"] + (["Tree_ID"] if "Tree_ID" in df.columns else []) + (["ID"] if "ID" in df.columns else [])]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ["LAI_Observed"])
        if len(sub) < max(n_splits, 10):
            continue
        X = sub[feats].to_numpy(dtype=float)
        y = sub["LAI_Observed"].to_numpy(dtype=float)
        cv, groups, cv_name = make_cv(sub, n_splits=n_splits, grouped=grouped)
        model = GradientBoostingRegressor(random_state=42)
        if groups is None:
            pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        else:
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
        row = {"K": k, "Num_Samples": int(len(y)), "CV": cv_name, "Features": ";".join(feats), **metrics(y, pred)}
        rows.append(row)
        print(f"K={k:02d} | N={len(y):3d} | {cv_name:10s} | R2={row['R2']:.3f} | RMSE={row['RMSE']:.3f}")
    return pd.DataFrame(rows)


def run_feature_selection(input_file: str, output_dir: str, max_k: int, optimal_k: int, n_splits: int,
                          grouped: bool, rank_method: str, include_bands: bool) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(input_file)
    if "LAI_Observed" not in df.columns:
        raise ValueError("Input table must contain LAI_Observed.")
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower().eq("success")].copy()

    features = build_candidate_features(df, include_bands=include_bands)
    if not features:
        raise ValueError("No candidate features were found.")

    ranking = rank_features(df, features, method=rank_method)
    ranking.to_csv(out_dir / "feature_ranking.csv", index=False, encoding="utf-8-sig")
    ranked_features = ranking["Feature"].tolist()

    curve = evaluate_subsets(df, ranked_features, max_k=max_k, n_splits=n_splits, grouped=grouped)
    curve.to_csv(out_dir / "feature_subset_learning_curve.csv", index=False, encoding="utf-8-sig")

    selected = ranked_features[: min(optimal_k, len(ranked_features))]
    (out_dir / "selected_features_topK.txt").write_text("\n".join(selected), encoding="utf-8")
    pd.DataFrame({"Feature": selected}).to_csv(out_dir / "selected_features_topK.csv", index=False, encoding="utf-8-sig")
    print(f"Selected {len(selected)} features saved to {out_dir / 'selected_features_topK.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature ranking and Top-K learning curve for LAI estimation.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Feature table.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--max-k", type=int, default=32, help="Maximum subset size evaluated.")
    parser.add_argument("--optimal-k", type=int, default=29, help="Feature subset size saved as selected features.")
    parser.add_argument("--n-splits", type=int, default=10, help="CV folds.")
    parser.add_argument("--no-grouped", action="store_true", help="Use shuffled KFold even if Tree_ID/ID is available.")
    parser.add_argument("--rank-method", choices=["pearson", "gbdt_importance"], default="pearson")
    parser.add_argument("--include-bands", action="store_true", help="Include base reflectance and standard deviation features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_feature_selection(
        input_file=args.input,
        output_dir=args.output_dir,
        max_k=args.max_k,
        optimal_k=args.optimal_k,
        n_splits=args.n_splits,
        grouped=not args.no_grouped,
        rank_method=args.rank_method,
        include_bands=args.include_bands,
    )


if __name__ == "__main__":
    main()
