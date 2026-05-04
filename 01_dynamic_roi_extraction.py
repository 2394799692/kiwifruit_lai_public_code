#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic ROI extraction for single-plant LAI estimation in trellis-trained kiwifruit.

This script implements the spatial-spectral dynamic ROI strategy used in the
manuscript. It anchors each sample tree by an RTK-measured center point, builds
an initial exploration domain, generates a local vegetation mask using NDVI and
NIR percentile thresholds, identifies the core canopy component, and estimates a
plant-specific dynamic ROI radius from the spatial distance percentile.

Required input files
--------------------
1) Sample table: CSV or XLSX
   Required columns:
       Tree_ID or ID       unique sample-tree identifier
       X or E              projected easting / x coordinate
       Y or N              projected northing / y coordinate
   Optional columns:
       Date                image date; if absent, all samples are processed for
                           every date in the imagery configuration file
       LAI_Observed        observed LAI value retained in the output

2) Imagery configuration table: CSV
   Required columns:
       Date, Green, Red, RedEdge, NIR
   Each band column should contain the path to the corresponding single-band
   orthomosaic GeoTIFF. Bands must be co-registered and share the same CRS,
   pixel size, and raster extent.

   Optional columns for empirical-line / relative radiometric correction:
       Green_a, Green_b, Red_a, Red_b, RedEdge_a, RedEdge_b, NIR_a, NIR_b
   Reflectance is calculated as: reflectance = a * DN + b.
   If your input images are already reflectance products, either leave these
   columns empty or set a = 1 and b = 0.

Outputs
-------
- dynamic_roi_summary.csv: radius, retained pixels, and processing status for
  each sample.
- dynamic_roi_polygons.gpkg: optional vector polygons of the extracted dynamic
  ROI masks. Enable SAVE_ROI_POLYGONS below if needed.

Notes
-----
- Default parameter values follow the submitted manuscript: initial exploration
  radius = 2.0 m, Q = 70, P = 90, Rmin = 0.2 m, Rmax = 2.6 m, fallback radius
  = 1.5 m, and minimum effective vegetation pixels = 30.
- For commercial orchards, this script assumes that a tree-inventory map or RTK
  tree centers are available. It does not perform fully unsupervised detection
  of every tree from the image.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as raster_shapes
from rasterio.mask import mask as raster_mask
from rasterio.transform import rowcol
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, disk

warnings.filterwarnings("ignore")


# =============================================================================
# User configuration. These values are used when the script is run without CLI
# arguments. File paths below are placeholders and should be edited by the user.
# =============================================================================
SAMPLE_FILE = "path/to/sample_tree_locations.csv"
IMAGERY_CONFIG_FILE = "path/to/imagery_config.csv"
OUTPUT_DIR = "outputs/dynamic_roi"
SAMPLE_CRS = "EPSG:4546"  # CRS of sample coordinates, e.g., CGCS2000 / 3-degree GK zone

SAVE_ROI_POLYGONS = True


@dataclass
class ROIParameters:
    """Parameter set for the dynamic ROI algorithm."""

    exploration_radius_m: float = 2.0
    q_percentile: float = 70.0
    p_percentile: float = 90.0
    r_min_m: float = 0.2
    r_max_m: float = 2.6
    alpha: float = 1.0
    fallback_radius_m: float = 1.5
    min_valid_pixels: int = 30
    morphology_close_radius: int = 2
    morphology_open_radius: int = 1


BAND_NAMES = ["Green", "Red", "RedEdge", "NIR"]


# =============================================================================
# Basic I/O utilities
# =============================================================================
def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common sample-table column names to Tree_ID, X, and Y."""
    df = df.copy()
    rename_map = {}
    if "Tree_ID" not in df.columns and "ID" in df.columns:
        rename_map["ID"] = "Tree_ID"
    if "X" not in df.columns and "E" in df.columns:
        rename_map["E"] = "X"
    if "Y" not in df.columns and "N" in df.columns:
        rename_map["N"] = "Y"
    df.rename(columns=rename_map, inplace=True)

    required = ["Tree_ID", "X", "Y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The sample table must contain Tree_ID/ID, X/E, and Y/N columns. "
            f"Missing normalized columns: {missing}"
        )
    return df


def samples_to_geodataframe(sample_df: pd.DataFrame, sample_crs: str) -> gpd.GeoDataFrame:
    """Convert sample coordinate table to a GeoDataFrame."""
    geometry = [Point(float(x), float(y)) for x, y in zip(sample_df["X"], sample_df["Y"])]
    return gpd.GeoDataFrame(sample_df.copy(), geometry=geometry, crs=sample_crs)


def load_imagery_config(path: str | Path) -> pd.DataFrame:
    """Load imagery configuration and validate required columns."""
    cfg = read_table(path)
    required = ["Date"] + BAND_NAMES
    missing = [c for c in required if c not in cfg.columns]
    if missing:
        raise ValueError(
            "The imagery configuration table must contain Date, Green, Red, "
            f"RedEdge, and NIR columns. Missing: {missing}"
        )
    return cfg


def open_band_sources(config_row: pd.Series) -> Dict[str, rasterio.io.DatasetReader]:
    """Open the four band rasters for one date."""
    src_dict = {}
    for band in BAND_NAMES:
        path = Path(str(config_row[band]))
        if not path.exists():
            raise FileNotFoundError(f"{band} raster not found: {path}")
        src_dict[band] = rasterio.open(path)
    return src_dict


def close_band_sources(src_dict: Dict[str, rasterio.io.DatasetReader]) -> None:
    """Close all open raster datasets."""
    for src in src_dict.values():
        src.close()


def get_calibration_coefficients(config_row: pd.Series) -> Dict[str, Tuple[float, float]]:
    """
    Read optional calibration coefficients from the imagery configuration row.

    Returns a dictionary: {band: (a, b)}, where reflectance = a * DN + b.
    Missing coefficients are ignored; the original raster values are then used.
    """
    coeffs: Dict[str, Tuple[float, float]] = {}
    for band in BAND_NAMES:
        a_col, b_col = f"{band}_a", f"{band}_b"
        if a_col not in config_row.index or b_col not in config_row.index:
            continue
        try:
            a = float(config_row[a_col])
            b = float(config_row[b_col])
        except (TypeError, ValueError):
            continue
        if np.isfinite(a) and np.isfinite(b):
            coeffs[band] = (a, b)
    return coeffs


def apply_calibration(arr: np.ndarray, band: str, coeffs: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Convert DN to reflectance if coefficients are provided."""
    arr = arr.astype(np.float32)
    if band in coeffs:
        a, b = coeffs[band]
        arr = a * arr + b
        arr = np.clip(arr, 0.001, 1.0)
    return arr.astype(np.float32)


# =============================================================================
# Dynamic ROI algorithm
# =============================================================================
def _buffer_radius_in_raster_units(gdf: gpd.GeoDataFrame, radius_m: float) -> float:
    """
    Convert a metric radius to raster CRS units.

    If the raster CRS is geographic, an approximate conversion is used. Projected
    CRS is recommended for accurate single-plant ROI extraction.
    """
    if gdf.crs is not None and gdf.crs.is_geographic:
        return radius_m / 111_320.0
    return radius_m


def crop_reflectance_bands(
    src_dict: Dict[str, rasterio.io.DatasetReader],
    geometry,
    coeffs: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, np.ndarray], rasterio.Affine]:
    """Crop all bands to the same geometry and convert to reflectance if needed."""
    arrays: Dict[str, np.ndarray] = {}
    out_transform = None
    for band, src in src_dict.items():
        out_image, out_transform = raster_mask(src, [geometry], crop=True, nodata=0)
        arrays[band] = apply_calibration(out_image[0], band, coeffs)
    if out_transform is None:
        raise RuntimeError("Failed to crop raster bands.")
    return arrays, out_transform


def build_local_vegetation_mask(
    refl: Dict[str, np.ndarray],
    params: ROIParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a local vegetation mask using NDVI and NIR percentile thresholds.

    This corresponds to the Q-based local spectral purity constraint.
    """
    nir = refl["NIR"]
    red = refl["Red"]
    ndvi = (nir - red) / (nir + red + 1e-6)

    valid = np.isfinite(nir) & np.isfinite(red) & (nir > 0) & (red > 0)
    ndvi_vals = ndvi[valid]
    nir_vals = nir[valid]

    if ndvi_vals.size < 10:
        veg_mask = valid & (ndvi > 0.25)
    else:
        q = params.q_percentile
        ndvi_thr = max(0.15, float(np.nanpercentile(ndvi_vals, q)))
        nir_thr = float(np.nanpercentile(nir_vals, q))
        veg_mask = valid & (ndvi >= ndvi_thr) & (nir >= nir_thr)

        # Relax the threshold once if too few pixels are retained.
        if int(np.sum(veg_mask)) < params.min_valid_pixels:
            relaxed_q = max(50.0, q - 10.0)
            ndvi_thr = max(0.10, float(np.nanpercentile(ndvi_vals, relaxed_q)))
            nir_thr = float(np.nanpercentile(nir_vals, relaxed_q))
            veg_mask = valid & (ndvi >= ndvi_thr) & (nir >= nir_thr)

    if int(np.sum(veg_mask)) > 0:
        veg_mask = binary_closing(veg_mask, disk(params.morphology_close_radius))
        veg_mask = binary_opening(veg_mask, disk(params.morphology_open_radius))

    return ndvi, veg_mask.astype(bool)


def select_core_component(veg_mask: np.ndarray, center_row: int, center_col: int) -> np.ndarray:
    """
    Select the connected canopy component associated with the RTK tree center.

    If the exact center pixel is not vegetation, the nearest connected component
    to the center is selected. This fallback is useful when the trunk center is
    shadowed or falls in a small canopy gap.
    """
    labeled = label(veg_mask, connectivity=2)
    if labeled.max() == 0:
        return np.zeros_like(veg_mask, dtype=bool)

    nrows, ncols = veg_mask.shape
    if 0 <= center_row < nrows and 0 <= center_col < ncols:
        center_label = labeled[center_row, center_col]
        if center_label > 0:
            return labeled == center_label

    # Fallback: choose the labeled component closest to the RTK center.
    best_label = 0
    best_dist = np.inf
    for lab in range(1, int(labeled.max()) + 1):
        rr, cc = np.where(labeled == lab)
        if rr.size == 0:
            continue
        dist = np.min((rr - center_row) ** 2 + (cc - center_col) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_label = lab
    return labeled == best_label


def estimate_roi_from_point(
    point: Point,
    src_dict: Dict[str, rasterio.io.DatasetReader],
    coeffs: Dict[str, Tuple[float, float]],
    params: ROIParameters,
) -> Dict[str, object]:
    """
    Estimate the dynamic ROI for one sample tree.

    Returns a dictionary containing the final ROI mask, cropped reflectance arrays,
    crop transform, radius, pixel count, and status fields.
    """
    ref_src = src_dict["NIR"]
    point_gdf = gpd.GeoDataFrame({"_id": [1]}, geometry=[point], crs=ref_src.crs)
    radius_units = _buffer_radius_in_raster_units(point_gdf, params.exploration_radius_m)
    explore_geom = point.buffer(radius_units)

    refl, out_transform = crop_reflectance_bands(src_dict, explore_geom, coeffs)
    _, veg_mask = build_local_vegetation_mask(refl, params)

    center_row, center_col = rowcol(out_transform, point.x, point.y)
    center_row, center_col = int(center_row), int(center_col)

    core_mask = select_core_component(veg_mask, center_row, center_col)
    core_pixels = int(np.sum(core_mask))

    rows, cols = np.where(core_mask)
    if core_pixels == 0:
        return {
            "status": "Failed_no_core_component",
            "mask": np.zeros_like(veg_mask, dtype=bool),
            "refl": refl,
            "transform": out_transform,
            "radius_m": np.nan,
            "pixel_count": 0,
        }

    # Pixel size in meters. Non-rotated orthomosaics are assumed.
    pixel_width = abs(out_transform.a)
    pixel_height = abs(out_transform.e)
    dists = np.sqrt(((cols - center_col) * pixel_width) ** 2 + ((rows - center_row) * pixel_height) ** 2)

    if core_pixels < params.min_valid_pixels:
        radius_m = params.fallback_radius_m
        status = "Fallback_too_few_valid_pixels"
    else:
        radius_m = params.alpha * float(np.nanpercentile(dists, params.p_percentile))
        radius_m = max(params.r_min_m, min(params.r_max_m, radius_m))
        status = "Success"

    # Apply the final spatial truncation around the RTK center.
    all_rows, all_cols = np.indices(core_mask.shape)
    all_dists = np.sqrt(
        ((all_cols - center_col) * pixel_width) ** 2 + ((all_rows - center_row) * pixel_height) ** 2
    )
    final_mask = core_mask & (all_dists <= radius_m)

    return {
        "status": status,
        "mask": final_mask.astype(bool),
        "core_mask": core_mask.astype(bool),
        "vegetation_mask": veg_mask.astype(bool),
        "refl": refl,
        "transform": out_transform,
        "radius_m": float(radius_m),
        "pixel_count": int(np.sum(final_mask)),
        "core_pixel_count": core_pixels,
    }


def roi_mask_to_polygon(mask: np.ndarray, transform) -> Optional[object]:
    """Convert a binary ROI mask to a shapely polygon or multipolygon."""
    if int(np.sum(mask)) == 0:
        return None
    geoms = []
    for geom, value in raster_shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if int(value) == 1:
            geoms.append(shape(geom))
    if not geoms:
        return None
    return unary_union(geoms)


# =============================================================================
# Main workflow
# =============================================================================
def run_dynamic_roi_extraction(
    sample_file: str | Path,
    imagery_config_file: str | Path,
    output_dir: str | Path,
    sample_crs: str,
    params: ROIParameters,
    save_polygons: bool = True,
) -> None:
    """Run dynamic ROI extraction for all samples and dates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df = normalize_sample_columns(read_table(sample_file))
    sample_gdf = samples_to_geodataframe(sample_df, sample_crs)
    imagery_cfg = load_imagery_config(imagery_config_file)

    summary_rows: List[dict] = []
    polygon_rows: List[dict] = []

    for _, cfg_row in imagery_cfg.iterrows():
        date_str = str(cfg_row["Date"])
        print(f"\nProcessing date: {date_str}")
        src_dict = open_band_sources(cfg_row)
        coeffs = get_calibration_coefficients(cfg_row)

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
                tree_id = row["Tree_ID"]
                record = {
                    "Tree_ID": tree_id,
                    "Date": date_str,
                    "X": float(sample_gdf.loc[idx, "X"]),
                    "Y": float(sample_gdf.loc[idx, "Y"]),
                    "Q_percentile": params.q_percentile,
                    "P_percentile": params.p_percentile,
                }
                if "LAI_Observed" in sample_gdf.columns:
                    record["LAI_Observed"] = sample_gdf.loc[idx, "LAI_Observed"]

                try:
                    roi = estimate_roi_from_point(row.geometry, src_dict, coeffs, params)
                    record.update({
                        "ROI_Radius_m": roi["radius_m"],
                        "Pixel_Count": roi["pixel_count"],
                        "Core_Pixel_Count": roi["core_pixel_count"],
                        "Status": roi["status"],
                    })
                    summary_rows.append(record)

                    if save_polygons and roi["pixel_count"] > 0:
                        polygon = roi_mask_to_polygon(roi["mask"], roi["transform"])
                        if polygon is not None:
                            poly_record = record.copy()
                            poly_record["geometry"] = polygon
                            polygon_rows.append(poly_record)
                except Exception as exc:  # keep processing other samples
                    record.update({
                        "ROI_Radius_m": np.nan,
                        "Pixel_Count": 0,
                        "Core_Pixel_Count": 0,
                        "Status": f"Error: {exc}",
                    })
                    summary_rows.append(record)
        finally:
            close_band_sources(src_dict)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "dynamic_roi_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved ROI summary: {summary_path}")

    if save_polygons and polygon_rows:
        roi_gdf = gpd.GeoDataFrame(polygon_rows, geometry="geometry", crs=imagery_cfg.iloc[0].get("CRS", None))
        # Use the CRS of the first image if no explicit CRS was provided in the config.
        first_src = rasterio.open(str(imagery_cfg.iloc[0]["NIR"]))
        roi_gdf.set_crs(first_src.crs, inplace=True, allow_override=True)
        first_src.close()
        polygon_path = output_dir / "dynamic_roi_polygons.gpkg"
        roi_gdf.to_file(polygon_path, driver="GPKG")
        print(f"Saved ROI polygons: {polygon_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic ROI extraction for UAV multispectral imagery.")
    parser.add_argument("--samples", default=SAMPLE_FILE, help="Sample tree table: CSV or XLSX.")
    parser.add_argument("--imagery-config", default=IMAGERY_CONFIG_FILE, help="Imagery configuration CSV.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--sample-crs", default=SAMPLE_CRS, help="CRS of sample coordinates, e.g., EPSG:4546.")
    parser.add_argument("--no-polygons", action="store_true", help="Do not export ROI polygons.")
    parser.add_argument("--q", type=float, default=70.0, help="Local spectral purity percentile Q.")
    parser.add_argument("--p", type=float, default=90.0, help="Spatial distance percentile P.")
    parser.add_argument("--explore-radius", type=float, default=2.0, help="Initial exploration radius in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ROIParameters(
        exploration_radius_m=args.explore_radius,
        q_percentile=args.q,
        p_percentile=args.p,
    )
    run_dynamic_roi_extraction(
        sample_file=args.samples,
        imagery_config_file=args.imagery_config,
        output_dir=args.output_dir,
        sample_crs=args.sample_crs,
        params=params,
        save_polygons=not args.no_polygons,
    )


if __name__ == "__main__":
    main()
