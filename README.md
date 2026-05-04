# Dynamic ROI-based single-plant LAI estimation in trellis-trained kiwifruit

This repository contains reproducible Python scripts for the UAV multispectral LAI workflow used in the manuscript. The scripts are cleaned public versions: local computer paths and project-specific temporary files have been removed.

## Scripts

| Script | Purpose |
|---|---|
| `01_dynamic_roi_extraction.py` | Extract dynamic single-plant ROIs using RTK/tree center anchors, local spectral purity threshold Q, and spatial distance percentile P. |
| `02_extract_vi_texture_features.py` | Extract band reflectance statistics, 24 vegetation indices, and 8 averaged NIR GLCM texture features within the dynamic ROI. |
| `03_temporal_window_ablation.py` | Compare May, August, September, and combined temporal windows using LOOCV. |
| `04_feature_selection_topk.py` | Rank features and generate the Top-K feature learning curve. |
| `05_roi_strategy_ablation.py` | Compare model performance among multiple ROI strategies from precomputed feature tables. |
| `06_model_benchmark_groupcv.py` | Compare GBDT, RandomForest, ExtraTrees, XGBoost, KNN, SVR, PLSR, and Ridge using grouped or random cross-validation. |
| `07_orchard_lai_mapping.py` | Train/load a GBDT model and generate an orchard-scale LAI GeoTIFF map. |

## Input data

### Sample tree table

A CSV/XLSX table with at least:

- `Tree_ID` or `ID`: sample-tree identifier
- `X`/`Y` or `E`/`N`: projected coordinates of the tree center
- `Date`: image date for the observation
- `LAI_Observed`: measured LAI, optional for ROI extraction but required for modeling

See `example_samples.csv`.

### Imagery configuration

A CSV table with:

- `Date`
- `Green`, `Red`, `RedEdge`, `NIR`: paths to four co-registered single-band GeoTIFFs
- optional calibration coefficients: `Green_a`, `Green_b`, ..., `NIR_a`, `NIR_b`

Reflectance is computed as `reflectance = a * DN + b`. If images are already reflectance products, use `a = 1` and `b = 0`.

See `example_imagery_config.csv`.

## Typical workflow

```bash
python 01_dynamic_roi_extraction.py \
  --samples example_samples.csv \
  --imagery-config example_imagery_config.csv \
  --output-dir outputs/dynamic_roi \
  --sample-crs EPSG:4546

python 02_extract_vi_texture_features.py \
  --samples example_samples.csv \
  --imagery-config example_imagery_config.csv \
  --output outputs/features/single_plant_lai_features.csv \
  --sample-crs EPSG:4546

python 03_temporal_window_ablation.py \
  --input outputs/features/single_plant_lai_features.csv \
  --output-dir outputs/temporal_ablation

python 04_feature_selection_topk.py \
  --input outputs/features/single_plant_lai_features.csv \
  --output-dir outputs/feature_selection \
  --optimal-k 29

python 06_model_benchmark_groupcv.py \
  --input outputs/features/single_plant_lai_features.csv \
  --features outputs/feature_selection/selected_features_topK.txt \
  --output-dir outputs/model_benchmark

python 07_orchard_lai_mapping.py \
  --feature-table outputs/features/single_plant_lai_features.csv \
  --features outputs/feature_selection/selected_features_topK.txt \
  --imagery-config example_imagery_config.csv \
  --date 2024-09-28 \
  --output-dir outputs/lai_mapping
```
## UAVdata
Due to the large size of the drone file, approximately 6GB, we uploaded it to Quark Cloud, which contains orthophoto images of the kiwifruit orchard in Dinghe Town for May, August, and September. Each month's folder includes pre spliced multispectral images of the green, red, red edge, and near red outer bands collected using DJI M3M：
I have shared 'UAV-data. zip' with you using Quark Cloud. Click on the link or copy the entire content, and open the Quark APP to access it.
/~88853YQDz7~:/
Link: https://pan.quark.cn/s/da72cf8f8f82


## Notes

- The dynamic ROI script assumes that tree centers are available from RTK measurement, a tree inventory map, or prior orchard-row mapping. It does not perform fully unsupervised tree detection from imagery.
- The default dynamic ROI parameters match the manuscript settings: `P = 90`, `Q = 70`, initial exploration radius `2.0 m`, radius bounds `0.2-2.6 m`, and fallback radius `1.5 m`.
- For the final public dataset, it is recommended to include anonymized `Tree_ID` and `Date` columns so grouped cross-validation can be reproduced.
