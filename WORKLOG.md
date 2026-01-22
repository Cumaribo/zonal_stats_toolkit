# Work Log

This file tracks the work done on the `zonal_stats_toolkit` project.

## 2026-01-22

*   **Visualization Refinement:** Updated `change_bars_pixel.R` to improve plot readability for country-level data.
    *   Implemented filtering to show only the top 5 and bottom 5 countries per service.
    *   Adjusted facet scales to `free` to remove empty rows for countries with no data in specific services.
    *   Enforced a fixed 3-column layout with specific service ordering, keeping placeholders for missing services.
    *   Fixed variable name cleaning logic (e.g., handling "polllination" typo).

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
