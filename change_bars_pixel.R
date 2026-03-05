# ==============================================================================
# PIPELINE: Zonal Stats Consolidation, Cleaning, and Visualization
# Project: Global NCP Analysis
# ==============================================================================

library(dplyr)
library(readr)
library(stringr)
library(sf)

# Automatically set working directory to the script's location in RStudio
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  script_path <- rstudioapi::getActiveDocumentContext()$path
  if (script_path != "") {
    setwd(dirname(script_path))
    message("Working directory set to: ", getwd())
  }
}

if (file.exists("Paths.R")) source("Paths.R") else source("paths.R")

# --- 1. DATA PREPARATION ---
# Read the raw output from the runner.py script. This file contains the zonal
# statistics for all services for both 1992 and 2020.
input_file <- project_dir("output_raw_fid", "grid_10km_20260219_224417.csv")
output_csv <- project_dir("output_raw_fid", "grid_10km_fid_filtered.csv")

# Define the path to the input grid GPKG
grid_gpkg_path <- project_dir("data", "vector_basedata", "AOOGrid_10x10km_land_4326_clean.gpkg")
output_gpkg <- project_dir("output_raw_fid", "grid_10km_with_stats.gpkg")

message("Reading raw data from: ", input_file)
raw_data <- read_csv(input_file, show_col_types = FALSE)

# Following the pattern from the original script, we select only the 'mean', 
# 'stdev', and 'valid_count' columns, which are needed for analysis. We also
# keep the 'fid' identifier. We also filter out fid = 0 as it is likely invalid.
message("Filtering columns and removing invalid fid=0...")
filtered_data <- raw_data %>%
  select(fid, starts_with(c("mean_", "stdev_", "valid_count_"))) %>%
  filter(fid != 0)

message("Writing filtered CSV to: ", output_csv)
write_csv(filtered_data, output_csv)

message("Reading input grid from: ", grid_gpkg_path)
grid_poly <- st_read(grid_gpkg_path, quiet = TRUE)

# Ensure we have an 'fid' column to join on. 
# If the GPKG doesn't have an explicit 'fid' column, we assume row numbers correspond to FIDs (1-based).
if (!"fid" %in% names(grid_poly)) {
  message("No 'fid' column found in GPKG. Creating 'fid' from row numbers (assuming 1-based index)...")
  grid_poly$fid <- 1:nrow(grid_poly)
}

message("Joining stats to grid geometry...")
grid_joined <- grid_poly %>%
  inner_join(filtered_data, by = "fid")

message("Cleaning column names for clarity...")
grid_cleaned <- grid_joined %>%
  rename_with(
    .cols = starts_with(c("mean_", "stdev_", "valid_count_")),
    .fn = ~ .x %>%
      # This regex is the key. It captures the stat, the service stem, and the year,
      # and discards all the junk that comes after the year.
      str_replace(
        pattern = "^(mean|stdev|valid_count)_(.*?)(1992|2020).*",
        replacement = "\\1_\\2\\3"
      ) %>%
      # Now, clean up the captured service part into canonical names.
      str_replace("Rt_ratio_", "C_Risk_Red_Ratio_") %>%
      str_replace("Rt_", "C_Risk_") %>%
      str_replace("global_n_export_tnc_esa", "N_export_") %>%
      str_replace("global_n_retention_ESAmar_", "N_retention_") %>%
      str_replace("global_sed_export_marine_mod_ESA_", "Sed_export_") %>%
      str_replace("global_usle_marine_mod_ESA_", "USLE_") %>%
      str_replace("nature_access_lspop2019_ESA", "Nature_Access_") %>%
      str_replace("realized_pollination_on_ag_ESAmar_", "Pollination_")
  ) %>%
  select(-fid) # Drop the temporary fid column before writing

message("Joined and cleaned ", nrow(grid_cleaned), " features. Writing joined GPKG to: ", output_gpkg)

# Explicitly delete the file if it exists to ensure a clean overwrite
if (file.exists(output_gpkg)) unlink(output_gpkg)

st_write(grid_cleaned, output_gpkg, delete_dsn = TRUE, quiet = TRUE)

message("Step 1 (Data Preparation & Join) complete.")

# --- ANALYSIS & VISUALIZATION (DISABLED) ---
# The following sections for change calculation, analysis, and plotting are
# intentionally disabled for this data preparation step. They are preserved
# here as a reference for future adaptation.
if (FALSE) {

  # ... The original analysis and plotting code was here ...

}

message("Script finished. Plotting sections were skipped as requested.")
