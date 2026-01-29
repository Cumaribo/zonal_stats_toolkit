# ==============================================================================
# PIPELINE: Zonal Stats Consolidation, Cleaning, and Visualization
# Project: Global NCP Analysis
# ==============================================================================

library(tidyverse)

# Automatically set working directory to the script's location in RStudio
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  script_path <- rstudioapi::getActiveDocumentContext()$path
  if (script_path != "") {
    setwd(dirname(script_path))
    message("Working directory set to: ", getwd())
  }
}

if (file.exists("Paths.R")) source("Paths.R") else source("paths.R")

# --- 0. CONFIGURATION & HELPERS ---
# Define a single source of truth for name cleaning to use throughout the script
clean_service_names <- function(column_names) {
  column_names %>%
    tolower() %>%
    # Remove common suffixes to simplify matching
    str_remove_all("_diff.*|_change.*|_1992_2020.*") %>%
    # Remove specific suffixes found in data
    str_remove_all("_esa") %>%
    
    # Map to desired service names (Longer specific matches first)
    str_replace_all("coastal_risk_reduction_ratio", "C_Risk_Red_Ratio") %>%
    str_replace_all("n_retention_ratio", "N_Ret_Ratio") %>%
    str_replace_all("sediment_retention_ratio|sed_retention_ratio", "Sed_Ret_Ratio") %>%
    
    str_replace_all("realized_polllination_on_ag|realized_pollination_on_ag", "Pollination") %>%
    
    str_replace_all("coastal_risk", "C_Risk") %>%
    str_replace_all("n_export", "N_export") %>%
    str_replace_all("sediment_export|sed_export", "Sed_export") %>%
    str_replace_all("polllination|pollination", "Pollination") %>%
    str_replace_all("nature_access", "Nature_Access") %>%
    
    str_replace_all("__+", "_") %>%
    str_replace_all("_$", "")
}

# --- 1. DATA INGESTION & STACKING ---
data_dir_zonal <- project_dir("output")

message("Using project output directory: ", data_dir_zonal)

if (!dir.exists(data_dir_zonal)) stop("Data directory not found: ", data_dir_zonal)

file_list <- list.files(path = data_dir_zonal, pattern = "\\.csv$", full.names = TRUE) %>% 
  .[!str_detect(., "data[._](combined|filtered|final|change|ES)")]

tt_combined <- map_df(file_list, ~{
  df <- read_csv(.x, show_col_types = FALSE)
  # Extract filename from full path and remove timestamp/extension
  grp_name <- basename(.x) %>% str_remove("_[0-9]{8}_[0-9]{6}\\.csv$") %>% str_remove("\\.csv$")
  
  # Robust renaming of the first column to 'unit'
  colnames(df)[colnames(df) == "unit"] <- "unit_original"
  colnames(df)[1] <- "unit"
  
  # Rename avg to mean to match script expectations
  colnames(df) <- gsub("^avg_", "mean_", colnames(df))
  
  df %>%
    mutate(grouping = grp_name, unit = as.character(unit)) %>%
    select(grouping, unit, everything())
})

# write_csv(tt_combined, "data.combined.csv")

# --- 2. FILTERING & CLEANING ---
# Keep only relevant summary stats and change variables
tt_filtered <- tt_combined %>% 
  select(grouping, unit, starts_with(c("mean_", "stdev_", "valid_count_"))) %>%
  filter(unit != "Antarctica") %>% filter(unit!= "Seven seas (open ocean)") # Remove Antarctica and Open Ocean
  

# TODO: Future QAQC - Check if valid pixel counts are consistent between years.
# We previously implemented a check for >5% difference in valid_count between 1992 and 2020.
# This flagged some units (e.g. coastal areas) where the mask changed significantly.
# We decided to disable this for now as the ratio variables might behave differently, 
# but it is a relevant test for future iterations to ensure "apples to apples" comparison.

# Select only the "change" or "diff" columns
tt_ch <- tt_filtered %>% 
  select(grouping, unit, contains("change"), contains("diff"))

# Apply the unified cleaning function # Lecacy also, the name cleaning has been sorted of. 
names(tt_ch) <- clean_service_names(names(tt_ch))

write_csv(tt_ch, "change_variables_cleaned.csv")

# --- 3. ANALYSIS (Standard Error Calculation) ---
# Reshape to long format to calculate SE across all services
tt_analysis <- tt_ch %>%
  select(-matches("X1|unnamed|source_file")) %>%
  pivot_longer(
    cols = -c(grouping, unit),
    names_to = c(".value", "service"),
    names_pattern = "(mean|stdev|valid_count)_(.*)"
  ) %>%
  mutate(se = stdev / sqrt(valid_count)) %>%
  filter(!is.na(mean), mean != 0) %>% 
  filter(!str_detect(service, "usle|n_retention")) # Focus on meaningful changes

# Save wide version for Becky/Rich (Mean and SE columns)
tt_final_wide <- tt_analysis %>%
  select(grouping, unit, service, mean, se) %>%
  pivot_wider(
    names_from = service, 
    values_from = c(mean, se),
    names_glue = "{.value}_{service}"
  )

write_csv(tt_final_wide, paste(data_dir_zonal, "final_ES_change_analysis.csv", sep="/"))


# --- 4. VISUALIZATION ---

# TODO: Add flexibility to filter which 'grouping' or 'service' is plotted (e.g. via function arguments or a config list).
# TODO: Define a canonical order for 'service' factor levels so they appear consistently across plots (not just alphabetical).
# TODO: Implement per-facet ordering for 'unit' so that bars are sorted descending by mean within each service panel (currently reorder() sorts globally).

# Define the specific order for the facets
svc_order <- c(
  "C_Risk", "N_export", "Sed_export",
  "C_Risk_Red_Ratio", "N_Ret_Ratio", "Sed_Ret_Ratio",
  "Pollination", "Nature_Access"
)

# We use a more specific name 'target_group' to avoid any confusion with column names
generate_es_plot <- function(target_group) {
  
  # Use .env$ to explicitly pull 'target_group' from the function argument
  data_subset <- tt_analysis %>% 
    filter(grouping == .env$target_group)
  
  # Filter top 5 and bottom 5 countries per service if grouping contains "country"
  if (str_detect(target_group, "country")) {
    message("Filtering top/bottom 5 units for: ", target_group)
    data_subset <- data_subset %>%
      group_by(service) %>%
      arrange(service, desc(mean)) %>%
      filter(row_number() <= 5 | row_number() > (n() - 5)) %>%
      ungroup()
  }

  # Skip if no data for this group
  if(nrow(data_subset) == 0) {
    message(paste("No data found for", target_group))
    return(NULL)
  }
  
  # Calculate height based on number of units (0.22 inches per unit)
  # This ensures long lists like Countries are readable
  num_units <- length(unique(data_subset$unit))
  calc_height <- max(6, num_units * 0.22)
  
  message(paste("Generating plot for:", target_group, "(Units:", num_units, ")"))
  
  # Ensure service is a factor with the correct order
  data_subset <- data_subset %>%
    mutate(service = factor(service, levels = svc_order))
  
  # Use free scales for countries (different units per facet), free_x for others (shared Y axis)
  facet_scales <- if (str_detect(target_group, "country")) "free" else "free_x"
  
  p <- ggplot(data_subset, aes(x = mean, y = unit, fill = service)) +
    geom_col(alpha = 0.7, show.legend = FALSE) +
    geom_errorbar(aes(xmin = mean - se, xmax = mean + se), width = 0.3, color = "grey30", na.rm = TRUE) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    facet_wrap(~service, scales = facet_scales, ncol = 3, drop = FALSE) +
    scale_y_discrete(limits = rev) +
    theme_minimal() +
    labs(
      title = paste("Ecosystem Service Change:", target_group),
      subtitle = "Bars = Mean Change | Error Bars = +/- 1 SE (1992-2020)",
      x = "Mean Change Value", 
      y = NULL
    ) +
    theme(
      strip.text = element_text(face = "bold", size = 11),
      axis.text.y = element_text(size = 7)
    )
  
  # Save with limitsize = FALSE to handle the long country plots
  ggsave(
    filename = paste0("ES_plot_", target_group, ".png"), 
    plot = p, 
    width = 14, 
    height = calc_height, 
    limitsize = FALSE
  )
}

# 5. EXECUTION
# Get the list of groupings and run the function for each
unique_groupings <- unique(tt_analysis$grouping)

# walk() is like a loop that doesn't print messy output
walk(unique_groupings, generate_es_plot)
