#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vegetation-index and NIR texture feature extraction using dynamic ROI masks.

This script reuses the dynamic ROI extraction functions in
01_dynamic_roi_extraction.py, and then calculates:

1) Mean and standard deviation of Green, Red, RedEdge, and NIR reflectance within
   each dynamic ROI.
2) Twenty-four vegetation indices from the mean band reflectance values.
3) Eight NIR GLCM texture metrics averaged over four directions: 0, 45, 90, and
   135 degrees.

Required input files
--------------------
The same files used by 01_dynamic_roi_extraction.py:
- Sample table: CSV or XLSX with Tree_ID/ID, X/E, Y/N, optional Date and
  LAI_Observed.
- Imagery configuration CSV with Date, Green, Red, RedEdge, NIR, and optional
  calibration coefficients for each band.

Output
------
- single_plant_lai_features.csv: one row per sample-date observation, including
  dynamic ROI information, band reflectance statistics, 24 vegetation indices,
  8 averaged NIR texture metrics, and observed LAI if provided.

Important
---------
Keep this file in the same folder as 01_dynamic_roi_extraction.py, or install
that file as a module, because this script imports its dynamic ROI functions.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings("ignore")


# =============================================================================
# Import the dynamic ROI module from the companion script.
# =============================================================================
THIS_DIR = Path(__file__).resolve().parent
DYNAMIC_ROI_SCRIPT = THIS_DIR / "01_dynamic_roi_extraction.py"

if not DYNAMIC_ROI_SCRIPT.exists():
    raise FileNotFoundError(
        "Cannot find 01_dynamic_roi_extraction.py. Please place this script in "
        "the same directory as the dynamic ROI script."
    )

spec = importlib.util.spec_from_file_location("dynamic_roi_extraction", DYNAMIC_ROI_SCRIPT)
dyn = importlib.util.module_from_spec(spec)
sys.modules["dynamic_roi_extraction"] = dyn
assert spec.loader is not None
spec.loader.exec_module(dyn)


# =============================================================================
# User configuration. These placeholders can be edited directly, or values can be
# provided through command-line arguments.
# =============================================================================
SAMPLE_FILE = "path/to/sample_tree_locations.csv"
IMAGERY_CONFIG_FILE = "path/to/imagery_config.csv"
OUTPUT_FILE = "outputs/features/single_plant_lai_features.csv"
SAMPLE_CRS = "EPSG:4546"


VI_24 = [
    "NDVI", "RVI", "DVI", "TNDVI", "RDVI", "NGRDI", "RI", "MSR", "MSAVI", "TVI",
    "WDRVI", "GRVI", "NLI", "MTVI2", "CRI", "GNDVI", "OSAVI", "SAVI", "PVI",
    "MCARI", "RGRI", "NDRE", "TCARI_OSAVI", "RECI",
]

TEX_8_AVG = [
    "TexNIR_Contrast_Avg", "TexNIR_Dissimilarity_Avg", "TexNIR_Homogeneity_Avg",
    "TexNIR_Energy_Avg", "TexNIR_Correlation_Avg", "TexNIR_ASM_Avg",
    "TexNIR_Entropy_Avg", "TexNIR_Mean_Avg",
]


# =============================================================================
# Feature calculation
# =============================================================================
def safe_array_value(x: float) -> float:
    """Return a finite float or NaN."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def calculate_24_indices_from_means(G: float, R: float, RE: float, N: float) -> Dict[str, float]:
    """Calculate 24 vegetation indices from mean band reflectance values."""
    eps = 1e-6
    G = safe_array_value(G) + eps
    R = safe_array_value(R) + eps
    RE = safe_array_value(RE) + eps
    N = safe_array_value(N) + eps

    out: Dict[str, float] = {}
    out["NDVI"] = (N - R) / (N + R)
    out["RVI"] = N / R
    out["DVI"] = N - R
    out["TNDVI"] = np.sqrt(np.clip(out["NDVI"] + 0.5, 0, None))
    out["RDVI"] = (N - R) / np.sqrt(np.clip(N + R, eps, None))
    out["NGRDI"] = (G - R) / (G + R)
    out["RI"] = (R - G) / (R + G)

    sr = N / R
    out["MSR"] = (sr - 1) / np.sqrt(np.clip(sr + 1, eps, None))
    out["MSAVI"] = (2 * N + 1 - np.sqrt(np.clip((2 * N + 1) ** 2 - 8 * (N - R), eps, None))) / 2
    out["TVI"] = 0.5 * (120 * (RE - G) - 200 * (R - G))
    out["WDRVI"] = (0.1 * N - R) / (0.1 * N + R)
    out["GRVI"] = G / R
    out["NLI"] = (N ** 2 - R) / (N ** 2 + R)

    mtvi2_num = 1.5 * (1.2 * (N - G) - 2.5 * (R - G))
    mtvi2_den = np.sqrt(
        np.clip((2 * N + 1) ** 2 - (6 * N - 5 * np.sqrt(np.clip(R, 0, None))) - 0.5, eps, None)
    )
    out["MTVI2"] = mtvi2_num / mtvi2_den
    out["CRI"] = (1 / G) - (1 / RE)
    out["GNDVI"] = (N - G) / (N + G)
    out["OSAVI"] = (N - R) / (N + R + 0.16)
    out["SAVI"] = 1.5 * (N - R) / (N + R + 0.5)

    # PVI uses a simple red-NIR soil line. This matches the common operational
    # form and avoids requiring site-specific soil-line fitting in the public code.
    out["PVI"] = (N - R) / np.sqrt(2)

    out["MCARI"] = ((RE - R) - 0.2 * (RE - G)) * (RE / R)
    out["RGRI"] = R / G
    out["NDRE"] = (N - RE) / (N + RE)
    tcari = 3 * ((RE - R) - 0.2 * (RE - G) * (RE / R))
    out["TCARI_OSAVI"] = tcari / (out["OSAVI"] + eps)
    out["RECI"] = (N / RE) - 1

    return {k: float(v) if np.isfinite(v) else np.nan for k, v in out.items()}


def band_statistics(refl: Dict[str, np.ndarray], roi_mask: np.ndarray) -> Dict[str, float]:
    """Calculate mean and standard deviation of each band within the ROI mask."""
    out: Dict[str, float] = {}
    if int(np.sum(roi_mask)) == 0:
        for band in dyn.BAND_NAMES:
            out[f"{band}_Refl"] = np.nan
            out[f"{band}_Std"] = np.nan
        return out

    for band in dyn.BAND_NAMES:
        vals = refl[band][roi_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[f"{band}_Refl"] = np.nan
            out[f"{band}_Std"] = np.nan
        else:
            out[f"{band}_Refl"] = float(np.nanmean(vals))
            out[f"{band}_Std"] = float(np.nanstd(vals))
    return out


def calculate_nir_glcm_texture(
    nir_refl: np.ndarray,
    roi_mask: np.ndarray,
    levels: int = 32,
    distance: int = 1,
) -> Dict[str, float]:
    """
    Calculate 8 NIR GLCM texture metrics averaged over four directions.

    Metrics: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM,
    Entropy, and Mean.
    """
    out = {name: np.nan for name in TEX_8_AVG}
    if nir_refl is None or roi_mask is None or int(np.sum(roi_mask)) < 9:
        return out

    ys, xs = np.where(roi_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    sub_img = nir_refl[y0 : y1 + 1, x0 : x1 + 1].copy()
    sub_mask = roi_mask[y0 : y1 + 1, x0 : x1 + 1].copy()

    vals = sub_img[sub_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < 9:
        return out

    min_v, max_v = float(np.nanmin(vals)), float(np.nanmax(vals))
    if max_v <= min_v:
        return out

    quantized = np.zeros_like(sub_img, dtype=np.uint8)
    scaled_vals = ((sub_img[sub_mask] - min_v) / (max_v - min_v) * (levels - 1)).astype(np.int32)
    quantized[sub_mask] = np.clip(scaled_vals, 0, levels - 1).astype(np.uint8)

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        quantized,
        distances=[distance],
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    properties = {
        "TexNIR_Contrast_Avg": "contrast",
        "TexNIR_Dissimilarity_Avg": "dissimilarity",
        "TexNIR_Homogeneity_Avg": "homogeneity",
        "TexNIR_Energy_Avg": "energy",
        "TexNIR_Correlation_Avg": "correlation",
        "TexNIR_ASM_Avg": "ASM",
    }
    for out_name, prop in properties.items():
        out[out_name] = float(np.nanmean(graycoprops(glcm, prop)))

    entropy_values: List[float] = []
    mean_values: List[float] = []
    for i in range(len(angles)):
        P = glcm[:, :, 0, i]
        P_nonzero = P[P > 0]
        entropy_values.append(float(-np.sum(P_nonzero * np.log(P_nonzero))))

        p_i = np.sum(P, axis=1)
        idx = np.arange(P.shape[0], dtype=float)
        mean_values.append(float(np.sum(idx * p_i)))

    out["TexNIR_Entropy_Avg"] = float(np.nanmean(entropy_values))
    out["TexNIR_Mean_Avg"] = float(np.nanmean(mean_values))
    return out


def extract_features_for_one_sample(
    point,
    src_dict,
    coeffs,
    params,
) -> Dict[str, object]:
    """Extract ROI statistics, vegetation indices, and texture features for one sample."""
    roi = dyn.estimate_roi_from_point(point, src_dict, coeffs, params)
    out = {
        "Status": roi["status"],
        "ROI_Radius_m": roi["radius_m"],
        "Pixel_Count": roi["pixel_count"],
        "Core_Pixel_Count": roi["core_pixel_count"],
    }

    if roi["pixel_count"] <= 0:
        for band in dyn.BAND_NAMES:
            out[f"{band}_Refl"] = np.nan
            out[f"{band}_Std"] = np.nan
        for vi in VI_24:
            out[vi] = np.nan
        for tex in TEX_8_AVG:
            out[tex] = np.nan
        return out

    band_stats = band_statistics(roi["refl"], roi["mask"])
    out.update(band_stats)

    vi = calculate_24_indices_from_means(
        G=band_stats["Green_Refl"],
        R=band_stats["Red_Refl"],
        RE=band_stats["RedEdge_Refl"],
        N=band_stats["NIR_Refl"],
    )
    out.update(vi)

    texture = calculate_nir_glcm_texture(roi["refl"]["NIR"], roi["mask"], levels=32, distance=1)
    out.update(texture)
    return out


# =============================================================================
# Main workflow
# =============================================================================
def run_feature_extraction(
    sample_file: str | Path,
    imagery_config_file: str | Path,
    output_file: str | Path,
    sample_crs: str,
    params,
) -> None:
    """Run feature extraction for all sample-date observations."""
    sample_df = dyn.normalize_sample_columns(dyn.read_table(sample_file))
    sample_gdf = dyn.samples_to_geodataframe(sample_df, sample_crs)
    imagery_cfg = dyn.load_imagery_config(imagery_config_file)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    feature_rows: List[dict] = []

    for _, cfg_row in imagery_cfg.iterrows():
        date_str = str(cfg_row["Date"])
        print(f"\nExtracting features for date: {date_str}")
        src_dict = dyn.open_band_sources(cfg_row)
        coeffs = dyn.get_calibration_coefficients(cfg_row)

        try:
            raster_crs = src_dict["NIR"].crs
            if raster_crs is None:
                raise ValueError("Raster CRS is undefined. Please assign a valid CRS to the imagery.")

            gdf_date = sample_gdf.copy()
            if "Date" in gdf_date.columns:
                gdf_date = gdf_date[gdf_date["Date"].astype(str) == date_str].copy()
                if gdf_date.empty:
                    print(f"  No samples matched Date = {date_str}; skipping.")
                    continue
            gdf_date_raster = gdf_date.to_crs(raster_crs)

            for idx, row in gdf_date_raster.iterrows():
                base_record = {
                    "Tree_ID": row["Tree_ID"],
                    "Date": date_str,
                    "X": float(sample_gdf.loc[idx, "X"]),
                    "Y": float(sample_gdf.loc[idx, "Y"]),
                }
                if "LAI_Observed" in sample_gdf.columns:
                    base_record["LAI_Observed"] = sample_gdf.loc[idx, "LAI_Observed"]

                try:
                    feat = extract_features_for_one_sample(row.geometry, src_dict, coeffs, params)
                    base_record.update(feat)
                except Exception as exc:
                    base_record["Status"] = f"Error: {exc}"
                    for band in dyn.BAND_NAMES:
                        base_record[f"{band}_Refl"] = np.nan
                        base_record[f"{band}_Std"] = np.nan
                    for vi in VI_24:
                        base_record[vi] = np.nan
                    for tex in TEX_8_AVG:
                        base_record[tex] = np.nan

                feature_rows.append(base_record)
        finally:
            dyn.close_band_sources(src_dict)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved feature table: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract spectral and NIR texture features within dynamic ROIs.")
    parser.add_argument("--samples", default=SAMPLE_FILE, help="Sample tree table: CSV or XLSX.")
    parser.add_argument("--imagery-config", default=IMAGERY_CONFIG_FILE, help="Imagery configuration CSV.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output feature CSV file.")
    parser.add_argument("--sample-crs", default=SAMPLE_CRS, help="CRS of sample coordinates, e.g., EPSG:4546.")
    parser.add_argument("--q", type=float, default=70.0, help="Local spectral purity percentile Q.")
    parser.add_argument("--p", type=float, default=90.0, help="Spatial distance percentile P.")
    parser.add_argument("--explore-radius", type=float, default=2.0, help="Initial exploration radius in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = dyn.ROIParameters(
        exploration_radius_m=args.explore_radius,
        q_percentile=args.q,
        p_percentile=args.p,
    )
    run_feature_extraction(
        sample_file=args.samples,
        imagery_config_file=args.imagery_config,
        output_file=args.output,
        sample_crs=args.sample_crs,
        params=params,
    )


if __name__ == "__main__":
    main()
