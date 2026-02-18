library(tidyverse)
library(here)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to the output directory defined in your INI file
raw_output_dir <- here("output_raw")

# Path where calculated change tables will be saved
final_output_dir <- here("output_raw", "calculated_change")
if (!dir.exists(final_output_dir)) dir.create(final_output_dir, recursive = TRUE)

# List of jobs (must match the [job:TAG] names in your INI file)
jobs <- c("country", "region_wb", "income_grp", "biome", "grid_10km")

# =============================================================================
# FUNCTIONS
# =============================================================================

process_job <- function(job_name) {
  csv_path <- file.path(raw_output_dir, paste0(job_name, ".csv"))
  
  if (!file.exists(csv_path)) {
    message(paste("Skipping missing file:", csv_path))
    return(NULL)
  }
  
  message(paste("Processing:", job_name, "from", csv_path))
  
  # Read CSV
  df <- read_csv(csv_path, show_col_types = FALSE)
  
  # Identify the ID column (assumed to be the first column)
  id_col <- names(df)[1]
  message(paste("  ID Column:", id_col))
  
  # 1. Pivot Longer: Convert wide columns into rows to extract metadata
  # Columns look like: mean_n_export_1992, sum_n_export_2020, etc.
  df_long <- df %>%
    pivot_longer(
      cols = -all_of(id_col),
      names_to = "col_name",
      values_to = "value"
    ) %>%
    # Extract Year (1992 or 2020) at the end of the string
    mutate(
      year = str_extract(col_name, "(1992|2020)$"),
      rest = str_remove(col_name, "_(1992|2020)$")
    ) %>%
    filter(!is.na(year)) # Drop columns that don't match the year pattern
  
  # 2. Extract Operation and Service Name
  # Known operations from runner.py
  ops <- c("mean", "stdev", "sum", "min", "max", "valid_count", "total_count", "p5", "p95")
  ops_pattern <- paste0("^(", paste(ops, collapse="|"), ")_")
  
  df_parsed <- df_long %>%
    mutate(
      operation = str_extract(rest, ops_pattern) %>% str_remove("_$"),
      service = str_remove(rest, ops_pattern)
    ) %>%
    select(-rest, -col_name)
  
  # 3. Pivot Wider to get 1992 and 2020 side-by-side
  df_paired <- df_parsed %>%
    pivot_wider(
      names_from = year,
      values_from = value,
      names_prefix = "y"
    )
  
  # 4. Calculate Change Metrics
  # SPC Formula: (y2 - y1) / (0.5 * (abs(y1) + abs(y2))) * 100
  # Range: -200% to +200%
  df_calc <- df_paired %>%
    mutate(
      abs_change = y2020 - y1992,
      spc = case_when(
        (abs(y1992) + abs(y2020)) == 0 ~ 0, # Avoid division by zero
        TRUE ~ (y2020 - y1992) / (0.5 * (abs(y1992) + abs(y2020))) * 100
      )
    ) %>%
    # Reorder columns for readability
    select(all_of(id_col), service, operation, y1992, y2020, abs_change, spc)
  
  # 5. Save Result
  out_path <- file.path(final_output_dir, paste0(job_name, "_change.csv"))
  write_csv(df_calc, out_path)
  message(paste("  Saved:", out_path))
}

# =============================================================================
# EXECUTION
# =============================================================================

walk(jobs, process_job)

message("Done! Check output_raw/calculated_change/ for results.")