# ==============================================================================
# PIPELINE: Bitemporal Difference Visualization
# ==============================================================================

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(ggplot2)

# --- 1. CONFIGURATION ---
dir_diff <- "output_diff_consolidated"
dir_base <- "output_consolidated"
out_dir <- "output_plots_diff"

unlink(file.path(out_dir, "*"), recursive = TRUE) # Clear existing plots for a fresh run
dir.create(out_dir, showWarnings = FALSE)

groups <- c("country", "region_wb", "income_grp", "biome")

# Helper to find latest file for a given group
get_latest_file <- function(d, g) {
  files <- list.files(d, pattern = paste0("^", g, ".*\\.csv$"), full.names = TRUE)
  if(length(files) == 0) return(NULL)
  sort(files, decreasing = TRUE)[1]
}

# Helper to map base columns to service & year
map_base_cols <- function(cols) {
  svc <- rep(NA_character_, length(cols))
  
  yr <- str_extract(cols, "(1992|2020)")
  
  svc[str_detect(cols, "Rt_ratio")] <- "C_Risk_Red_Ratio"
  svc[str_detect(cols, "mean_Rt_(1992|2020)")] <- "C_Risk"
  svc[str_detect(cols, "n_export")] <- "N_export"
  svc[str_detect(cols, "n_retention_ratio")] <- "N_Ret_Ratio"
  svc[str_detect(cols, "n_retention_ESAmar")] <- "N_retention"
  svc[str_detect(cols, "sed_export")] <- "Sed_export"
  svc[str_detect(cols, "sed_retention_ratio")] <- "Sed_Ret_Ratio"
  svc[str_detect(cols, "usle")] <- "USLE"
  svc[str_detect(cols, "nature_access")] <- "Nature_Access"
  svc[str_detect(cols, "pollination")] <- "Pollination"
  
  tibble(col_name = cols, service = svc, year = yr) %>% filter(!is.na(service), !is.na(year))
}

# Helper to map diff columns to service and metric
map_diff_cols <- function(cols) {
  metric <- rep(NA_character_, length(cols))
  metric[str_detect(cols, "^mean_")] <- "mean_val"
  metric[str_detect(cols, "^stdev_")] <- "stdev_val"
  metric[str_detect(cols, "^valid_count_")] <- "valid_count"
  
  svc <- rep(NA_character_, length(cols))
  
  svc[str_detect(cols, "Rt_ratio")] <- "C_Risk_Red_Ratio"
  svc[str_detect(cols, "_Rt_diff")] <- "C_Risk"
  svc[str_detect(cols, "n_export")] <- "N_export"
  svc[str_detect(cols, "n_retention_diff")] <- "N_retention"
  svc[str_detect(cols, "n_retention_ratio")] <- "N_Ret_Ratio"
  svc[str_detect(cols, "sed_export")] <- "Sed_export"
  svc[str_detect(cols, "sed_retention_ratio")] <- "Sed_Ret_Ratio"
  svc[str_detect(cols, "usle")] <- "USLE"
  svc[str_detect(cols, "nature_access")] <- "Nature_Access"
  svc[str_detect(cols, "pollination")] <- "Pollination"
  
  tibble(col_name = cols, service = svc, metric = metric) %>% filter(!is.na(service), !is.na(metric))
}

for (grp in groups) {
  f_diff <- get_latest_file(dir_diff, grp)
  f_base <- get_latest_file(dir_base, grp)
  if (is.null(f_diff) || is.null(f_base)) next
  
  message("\n=======================================================")
  message("Processing group: ", grp)
  
  df_diff <- read_csv(f_diff, show_col_types = FALSE)
  df_base <- read_csv(f_base, show_col_types = FALSE)
  
  grp_col <- names(df_diff)[1]
  
  # Select mean and stdev columns
  df_diff_stat <- df_diff %>% select(all_of(grp_col), starts_with(c("mean_", "stdev_", "valid_count_")))
  
  diff_map <- map_diff_cols(names(df_diff_stat))
  
  df_long <- df_diff_stat %>%
    pivot_longer(cols = -all_of(grp_col), names_to = "col_name", values_to = "val") %>%
    inner_join(diff_map, by = "col_name") %>%
    select(-col_name) %>%
    pivot_wider(names_from = metric, values_from = val) %>%
    filter(!is.na(!!sym(grp_col)), !is.na(mean_val)) %>%
    mutate(
      # Align the sign convention to (2020 - 1992)
      mean_val = if_else(str_detect(service, "^C_Risk"), -mean_val, mean_val),
      # Calculate Standard Error of the Mean
      se_val = stdev_val / sqrt(valid_count)
    )
      
  # Calculate SPC from base years
  df_base_mean <- df_base %>% select(all_of(grp_col), starts_with("mean_"))
  base_map <- map_base_cols(names(df_base_mean))
  
  df_base_long <- df_base_mean %>%
    pivot_longer(cols = -all_of(grp_col), names_to = "col_name", values_to = "mean_val") %>%
    inner_join(base_map, by = "col_name") %>%
    select(-col_name) %>%
    pivot_wider(names_from = year, values_from = mean_val, names_prefix = "mean_") %>%
    mutate(
      sym_pct_change = 200 * (mean_2020 - mean_1992) / (abs(mean_2020) + abs(mean_1992))
    ) %>%
    filter(!is.na(!!sym(grp_col))) %>%
    select(all_of(grp_col), service, sym_pct_change)
    
  df_long <- df_long %>% left_join(df_base_long, by = c(grp_col, "service"))

  # Filter out N_retention globally so it is removed from all plots
  df_long <- df_long %>% filter(service != "N_retention")

  if (grp == "biome") {
    df_long <- df_long %>% filter(!(!!sym(grp_col) %in% c("Lakes", "Rock & Ice")))
  }
  if (grp == "region_wb") {
    df_long <- df_long %>% filter(!!sym(grp_col) != "Antarctica")
  }

  # Filter logic: Top/Bottom 5 for countries, all for others
  if (grp == "country") {
    df_plot_abs <- df_long %>%
      group_by(service) %>%
      # Exclude the bottom 10% smallest countries by valid area to avoid pixel variance bias
      filter(valid_count >= quantile(valid_count, 0.10, na.rm = TRUE)) %>%
      arrange(service, mean_val) %>%
      filter(row_number() <= 5 | row_number() >= n() - 4) %>%
      ungroup() %>%
      # Create unique per-facet labels sorted by mean_val for independent sorting
      mutate(plot_label = reorder(paste(!!sym(grp_col), service, sep = "__"), mean_val))
      
    df_plot_pct <- df_long %>%
      group_by(service) %>%
      filter(!is.na(sym_pct_change)) %>%
      filter(valid_count >= quantile(valid_count, 0.10, na.rm = TRUE)) %>%
      arrange(service, sym_pct_change) %>%
      filter(row_number() <= 5 | row_number() >= n() - 4) %>%
      ungroup() %>%
      mutate(plot_label = reorder(paste(!!sym(grp_col), service, sep = "__"), sym_pct_change))

    facet_scales <- "free"
    x_var <- "plot_label"
  } else {
    df_plot_abs <- df_long
    df_plot_pct <- df_long %>% filter(!is.na(sym_pct_change))
    
    # Ensure alphabetical ordering of y-axis (descending order for factors so A is at top after coord_flip)
    group_levels <- sort(unique(df_long[[grp_col]]), decreasing = TRUE)
    df_plot_abs[[grp_col]] <- factor(df_plot_abs[[grp_col]], levels = group_levels)
    df_plot_pct[[grp_col]] <- factor(df_plot_pct[[grp_col]], levels = group_levels)
    
    facet_scales <- "free_x"
    x_var <- grp_col
  }
  
  # Generate Faceted Plot
  p_abs <- ggplot(df_plot_abs, aes(x = !!sym(x_var), y = mean_val, fill = mean_val > 0)) +
    geom_col() +
    geom_linerange(aes(ymin = mean_val - se_val, ymax = mean_val + se_val), alpha = 0.5, linewidth = 0.6) +
    coord_flip() +
    facet_wrap(~ service, scales = facet_scales, ncol = 3) +
    scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato")) +
    scale_x_discrete(labels = function(x) gsub("__.*$", "", x)) +
    labs(title = paste("Absolute Change by", grp), 
         subtitle = if(grp == "country") "Top 5 & Bottom 5 (Bottom 10% by valid area excluded)" else "All records",
         x = NULL, 
         y = "Absolute Difference (2020-1992)") +
    theme_minimal() + 
    theme(legend.position = "none",
          strip.text = element_text(face = "bold"))
          
  ggsave(file.path(out_dir, paste0(grp, "_faceted_abs_change.png")), p_abs, width = 14, height = 10, bg="white")

  # Generate Faceted Plot PCT
  p_pct <- ggplot(df_plot_pct, aes(x = !!sym(x_var), y = sym_pct_change, fill = sym_pct_change > 0)) +
    geom_col() +
    coord_flip() +
    facet_wrap(~ service, scales = facet_scales, ncol = 3) +
    scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato")) +
    scale_x_discrete(labels = function(x) gsub("__.*$", "", x)) +
    labs(title = paste("Symmetric % Change by", grp), 
         subtitle = if(grp == "country") "Top 5 & Bottom 5 (Bottom 10% by valid area excluded)" else "All records",
         x = NULL, 
         y = "Symmetric % Change (2020-1992)") +
    theme_minimal() + 
    theme(legend.position = "none",
          strip.text = element_text(face = "bold"))
          
  ggsave(file.path(out_dir, paste0(grp, "_faceted_pct_change.png")), p_pct, width = 14, height = 10, bg="white")
}

message("\nAnalysis complete! Plots saved to ", out_dir)