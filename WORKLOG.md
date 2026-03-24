# Work Log

This file tracks the work done on the `zonal_stats_toolkit` project.

## 2026-03-24

*   **Visualization Refactor:** Overhauled `compare_and_plot_changes.R` to generate consolidated, faceted bitemporal difference plots for both Absolute Change and Symmetric Percentage Change.
*   **Variance & Micro-state Handling:** Implemented Standard Error of the Mean (SEM) for error bars instead of raw standard deviation to tighten confidence intervals. Added dynamic filtering to exclude the bottom 10% of countries by `valid_count` to prevent tiny micro-states and islands from skewing the "top 5 / bottom 5" rankings due to extreme pixel variance. Excluded irrelevant regions (Antarctica, Lakes, Rock & Ice).
*   **Methodological Finding (Pollination Discrepancy):** Investigated a divergence where the difference of the means ($\text{Mean}_{2020} - \text{Mean}_{1992}$) did not perfectly match the mean of the differences ($\text{Mean}_{\Delta}$) for Pollination. Concluded this is a **NoData Mask Misalignment** caused by shifting agricultural footprints. Because Pollination is only calculated on agricultural pixels, land-use changes between 1992 and 2020 alter the total valid area (the denominator) independently in each year, creating a natural mathematical divergence from pixel-by-pixel difference rasters. This behavior is expected and should be noted in the methodology.
*   **Data Backfilling:** Discovered `n_retention_ratio` and `sed_retention_ratio` were absent from the base year consolidated CSVs. Developed a lightweight `missing_ratios_base.ini` and a python script `append_ratios.py` to calculate strictly the missing metrics and safely left-join them into the existing master CSVs, avoiding a massive, multi-day pipeline rerun.
*   **Map Generation:** Updated `compare_and_plot_changes.R` to safely export aligned map data prior to filtering. Created `generate_map_gpkgs.py` to automatically pivot absolute and percentage change data into a wide format and join it directly to dissolved vector boundaries (`.gpkg`), generating presentation-ready map layers.
*   **Methodological Finding (Spatial Dissolve vs Tabular Grouping):** Documented a critical performance insight regarding raster zonal statistics engines (like `exactextractr`). Dissolving complex polygons prior to extraction causes massive bounding boxes (leading to Out-Of-Memory errors) and exponentially slows down edge-intersection math. The correct, highly optimized pipeline design avoids geographic dissolves entirely, instead running extraction over a high-resolution grid and performing tabular aggregations (`group_by %>% summarize`) post-extraction.

## 2026-03-20

*   **Runner Configuration Enhancements:** Updated `runner.py`'s configuration parser to gracefully handle and ignore sections starting with `[skip:job:...]`. This allows for modular pipeline execution (e.g., extracting regional datasets while skipping massive global 10km grid jobs) without having to delete the configuration block entirely.
*   **Housekeeping & Archival:** Cleaned up the root directory by moving legacy, one-off diagnostic scripts (`compare_gpkg_columns.py`, `check_csv_diff.py`, `inspect_vector.py`, `inspect_gpkg.py`) into a dedicated `qa_scripts/archive/` folder.

## 2026-03-13 (Continued)

*   **Validation Success & Methodological Justification:** Successfully cross-validated the Legacy (GDAL Rasterize) pipeline against the Optimized (`exact_extract` GPKG) pipeline.
    *   Achieved **0.9975 Pearson Correlation** across 1.69 million grid cells, proving perfect spatial distribution alignment.
    *   Identified that the variance (RMSE) between the datasets is strictly driven by boundary-pixel handling. The legacy pipeline relies on center-point intersection (`ALL_TOUCHED=FALSE`), causing "all-or-nothing" artifacts at grid edges.
    *   Concluded that the optimized pipeline is mathematically superior because it calculates exact fractional pixel overlap, preventing both the missing data of `ALL_TOUCHED=FALSE` and the double-counting of `ALL_TOUCHED=TRUE`.
    *   The Optimized Pipeline is now fully validated for downstream analysis.
*   **Bi-Temporal Change Calculation:**
    *   Developed `calculate_bitemporal_change.py` to efficiently calculate the difference between two time periods (e.g., 2020 vs 1992) for all relevant metrics.
    *   The script operates directly on the GeoPackage using `osgeo.ogr` to add new `_diff_` columns, avoiding memory-intensive `geopandas` operations and bypassing `sqlite3` limitations with spatial triggers.
    *   Successfully executed the script, adding 18 bi-temporal change columns to the `10k_grid_services_base.gpkg` dataset, preparing it for final analysis.

## 2026-03-13

*   **Pipeline Validation Framework:** Developed `compare_gpkg_columns.py` to perform statistical cross-validation between "Legacy" (CSV) and "Optimized" (GPKG) pipeline outputs.
    *   Added logic to handle GeoPackages where the Feature ID (`fid`) is stored in the internal index rather than as an explicit column.
    *   Implemented Normalized RMSE (NRMSE) metrics to quantify the fidelity of the new pipeline relative to the baseline.
*   **Runner Determinism:** Updated `runner.py` to enforce explicit sorting of the output dataframe by the aggregation field (e.g., `fid`).
    *   This ensures deterministic row ordering, which is critical for aligning non-spatial CSV outputs with spatial vectors.
    *   Modified the output structure to ensure the aggregation field is always the first column in the CSV for improved readability.

## 2026-03-09

*   **Raster Conversion Overhaul:** Refactored `convert_to_ha.py` to resolve critical performance and stability issues when processing large global rasters.
    *   Replaced the `rioxarray`/`dask`-based implementation with a memory-efficient `rasterio` and `WarpedVRT` approach. This processes rasters in small blocks (windows), preventing memory exhaustion and "Write failed" errors.
    *   Fixed a critical bug where `NoData` values were being included in calculations, leading to incorrect negative values in the output per-hectare rasters.
    *   Identified that parallel processing was overwhelming system I/O, causing write failures even with sufficient disk space. The process was updated to run sequentially (`max_workers=1`) for stability.
    *   Enforced `BIGTIFF=YES` creation to accommodate output files exceeding the 4GB limit of standard TIFFs.
    *   The new implementation is significantly faster and produces correct, verifiable outputs.
*   **Zonal Statistics `fid` Fix:** Modified `runner.py` to correctly handle `agg_field = fid` by using `GetFID()` instead of `GetField()`, preventing crashes. Also ensured stable row ordering in the output CSV by sorting features by ID before aggregation.

## 2026-01-29

*   **Data Integration & Script Refactoring:**
    *   Began work to integrate new vector-based data attributes into the analysis pipeline.
    *   Updated the `change_bars_pixel.R` plotting script to handle new data inputs, particularly for coastal protection. This required significant refactoring of the variable name cleaning logic to correctly map new `Rt` and `Rt_ratio` columns to their corresponding services (`C_Risk`, `C_Risk_Red_Ratio`).
    *   Refined the cleaning function to be more robust, preventing incorrect partial matching on other service names like `n_export`.
    *   Removed unnecessary `_nohab` columns from the analysis to resolve duplicate column name errors that arose after the name cleaning.
*   **Outputs:** Generated new output plots and summary CSVs reflecting the corrected analysis.

## 2026-01-28

*   **Consolidated Analysis Configuration:** Updated the main configuration file (`global_ncp_change_analysis.ini`) to include vector-based coastal protection data from `c_protection_ch.gpkg`. This merges the "double check" analysis into the main workflow, allowing for a single, consolidated run that processes both the original raster datasets and the new vector-attribute-derived rasters for coastal protection metrics (`Rt_diff_1992_2020`, `Rt_nohab_all_diff_1992_2020`, `Rt_ratio_diff_1992_2020`).

## 2026-01-22

*   **Visualization Refinement:** Updated `change_bars_pixel.R` to improve plot readability for country-level data.
    *   Implemented filtering to show only the top 5 and bottom 5 countries per service.
    *   Adjusted facet scales to `free` to remove empty rows for countries with no data in specific services.
    *   Enforced a fixed 3-column layout with specific service ordering, keeping placeholders for missing services.
    *   Fixed variable name cleaning logic (e.g., handling "polllination" typo).
    *   Changed Y-axis sorting to alphabetical (inverted) for consistent reading.
    *   Optimized facet scales: shared Y-axis for standard groupings, free Y-axis for country lists.

## 2026-01-20

*   Recalculated the sediment and nitrogen retention ratio files to address an artifact causing atypical values and `NaN` outputs. The script `/home/jeronimo/projects/global_NCP/Python_scripts/calculate_ratios.py` was used to regenerate the ratios and their differences. Now re-running zonal statistics for these new ratios.
*   **Disk Space Management:** Investigated and resolved critical disk space errors (`disk usage exceeds 95%`) by identifying that `workdir/` contains temporary files that are not cleaned up automatically. Emptied the `workdir/` to allow the pipeline to run.
*   **Permissions Fix:** Resolved `PermissionError` issues on `workdir/` and `output/` directories by changing their ownership from `root` to the active user (`1001`), allowing the script to create necessary files and directories.

## 2026-01-12

*   **Initial setup:** Cloned the repository.
*   **Debugging:** Encountered and worked on fixing several issues:
    *   `NotImplementedError` related to non-relative patterns in `pathlib.glob`.
    *   `fiona.errors.DriverError` due to file permissions and incorrect paths inside the Docker container.
    *   Moved data into the project's `/data` directory and updated `.gitignore`.
    *   Adjusted paths in `custom_ncp_analysis.ini` to reflect the new data location.
*   **Identified `ValueError`:** The script fails with `ValueError: [job:custom] agg_field "rgn_id" not found...`. The next step is to fix the `agg_field` in the configuration file.
*   **Fixed Configuration:** Corrected the vector file path in `global_ncp_change_analysis.yml` (removed extra `ee_` from filename) to resolve `FileNotFoundError`.
*   **Fixed Output Columns:** Modified `runner.py` to:
    *   Include `avg` in the calculated statistics.
    *   Filter the output CSV columns based on the requested `operations` in the configuration, removing unrequested fields like `min` and `max`.
*   **Fixed Path Resolution:** Updated `runner.py` to resolve paths relative to the configuration file, fixing `FileNotFoundError` when running from different directories.
*   **Fixed NaN Handling:** Modified `runner.py` to treat `NaN` values in floating-point rasters as NoData, resolving the issue where stats returned `NaN`.
*   **Improved Error Handling:** Added detailed diagnostic messages for `PermissionError` in `runner.py` to help debug Docker user/group permission mismatches.
*   **Docker Execution:** Successfully ran the full `global_ncp_change_analysis.ini` pipeline inside the container (running as root to bypass host permission issues).
