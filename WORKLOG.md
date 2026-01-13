# Work Log

This file tracks the work done on the `zonal_stats_toolkit` project.

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
