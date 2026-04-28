# ==============================================================================
# PIPELINE: Bitemporal Difference Visualization
# ==============================================================================

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(ggplot2)
library(patchwork)

# --- 1. CONFIGURATION ---
dir_diff <- "output_diff_consolidated"
dir_base <- "output_consolidated"
out_dir <- "output_plots_diff"

unlink(file.path(out_dir, "*"), recursive = TRUE) # Clear existing plots for a fresh run
dir.create(out_dir, showWarnings = FALSE)

groups <- c("country", "region_wb", "income_grp", "biome")

# Universal Palettes for Regional Groupings
group_palettes <- list(
  biome = c(
    'Tropical & Subtropical Moist Broadleaf Forests' = '#319D00',
    'Tropical & Subtropical Dry Broadleaf Forests' = '#7ABD1B',
    'Tropical & Subtropical Coniferous Forests' = '#556E19',
    'Temperate Broadleaf & Mixed Forests' = '#207433',
    'Temperate Coniferous Forests' = '#3E8D62',
    'Boreal Forests/Taiga' = '#496FF3',
    'Tropical & Subtropical Grasslands, Savannas & Shrublands' = '#D6F392',
    'Temperate Grasslands, Savannas & Shrublands' = '#D1E614',
    'Flooded Grasslands & Savannas' = '#75D0D5',
    'Montane Grasslands & Shrublands' = '#98E600',
    'Tundra' = '#C7DEFF',
    'Mediterranean Forests, Woodlands & Scrub' = '#AF963C',
    'Deserts & Xeric Shrublands' = '#C55C5C',
    'Mangroves' = '#FE04BC'
  ),
  income_grp = c('1. High income: OECD' = '#004D33', '2. High income: nonOECD' = '#1d7355', '3. Upper middle income' = '#4b9e80', '4. Lower middle income' = '#8bc5af', '5. Low income' = '#cde9df'),
  region_wb = c('East Asia & Pacific' = '#2E5A88', 'Europe & Central Asia' = '#D86018', 'Latin America & Caribbean' = '#7A3F91', 'Middle East & North Africa' = '#B38F00', 'North America' = '#1D8A99', 'South Asia' = '#6B8E23', 'Sub-Saharan Africa' = '#8B0000')
)
group_palettes$WWF_biome <- group_palettes$biome

# Short-name labels specifically for the dense Biome legend
biome_labels <- c(
  'Tropical & Subtropical Moist Broadleaf Forests' = 'Trop/Subtrop Moist Broadleaf',
  'Tropical & Subtropical Dry Broadleaf Forests' = 'Trop/Subtrop Dry Broadleaf',
  'Tropical & Subtropical Coniferous Forests' = 'Trop/Subtrop Coniferous',
  'Temperate Broadleaf & Mixed Forests' = 'Temp Broadleaf/Mixed',
  'Temperate Coniferous Forests' = 'Temp Coniferous',
  'Boreal Forests/Taiga' = 'Boreal/Taiga',
  'Tropical & Subtropical Grasslands, Savannas & Shrublands' = 'Trop/Subtrop Grass/Sav/Shrub',
  'Temperate Grasslands, Savannas & Shrublands' = 'Temp Grass/Sav/Shrub',
  'Flooded Grasslands & Savannas' = 'Flooded Grass/Savannas',
  'Montane Grasslands & Shrublands' = 'Montane Grass/Shrub',
  'Tundra' = 'Tundra',
  'Mediterranean Forests, Woodlands & Scrub' = 'Mediterranean',
  'Deserts & Xeric Shrublands' = 'Deserts & Xeric Shrub',
  'Mangroves' = 'Mangroves'
)

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
    
  df_long <- df_long %>% left_join(df_base_long, by = c(grp_col, "service")) %>%
    mutate(
      status_abs = case_when(
        service %in% c("Nature_Access","Pollination","N_Ret_Ratio","Sed_Ret_Ratio","C_Risk_Red_Ratio") & mean_val > 0 ~ "Good",
        service %in% c("Nature_Access","Pollination","N_Ret_Ratio","Sed_Ret_Ratio","C_Risk_Red_Ratio") & mean_val < 0 ~ "Bad",
        service %in% c("Sed_export","N_export","C_Risk") & mean_val < 0 ~ "Good",
        service %in% c("Sed_export","N_export","C_Risk") & mean_val > 0 ~ "Bad",
        TRUE ~ "Neutral"
      ),
      status_pct = case_when(
        service %in% c("Nature_Access","Pollination","N_Ret_Ratio","Sed_Ret_Ratio","C_Risk_Red_Ratio") & sym_pct_change > 0 ~ "Good",
        service %in% c("Nature_Access","Pollination","N_Ret_Ratio","Sed_Ret_Ratio","C_Risk_Red_Ratio") & sym_pct_change < 0 ~ "Bad",
        service %in% c("Sed_export","N_export","C_Risk") & sym_pct_change < 0 ~ "Good",
        service %in% c("Sed_export","N_export","C_Risk") & sym_pct_change > 0 ~ "Bad",
        TRUE ~ "Neutral"
      )
    )

  # Filter out N_retention and USLE globally so they are removed from all plots
  df_long <- df_long %>% filter(!(service %in% c("N_retention", "USLE")))

  # Enforce canonical order for facets (8 services, leaving bottom-right empty in a 3x3 grid)
  canonical_order <- c("C_Risk", "N_export", "Sed_export",
                       "C_Risk_Red_Ratio", "N_Ret_Ratio", "Sed_Ret_Ratio",
                       "Pollination", "Nature_Access")
  df_long$service <- factor(df_long$service, levels = canonical_order)

  if (grp == "biome") {
    df_long <- df_long %>% filter(!(!!sym(grp_col) %in% c("Lakes", "Rock & Ice")))
  }
  if (grp == "region_wb") {
    df_long <- df_long %>% filter(!!sym(grp_col) != "Antarctica")
  }
  
  # Export the fully aligned data for map generation before filtering
  write_csv(df_long, file.path(out_dir, paste0(grp, "_map_data.csv")))

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
  
  # Calculate global weighted averages per service for the reference lines
  df_global_avg <- df_long %>%
    group_by(service) %>%
    summarise(
      global_avg_abs = sum(mean_val * valid_count, na.rm = TRUE) / sum(valid_count, na.rm = TRUE),
      global_avg_pct = sum(sym_pct_change * valid_count, na.rm = TRUE) / sum(valid_count, na.rm = TRUE)
    )

  grp_name <- switch(grp,
    country = "Country",
    region_wb = "World Bank Region",
    income_grp = "Income Group",
    biome = "Biome"
  )

  custom_palette <- group_palettes[[grp]]
  legend_cols <- if (grp == "country") 10 else if (grp == "biome") 5 else 4
  status_colors <- c("Good" = "#007930", "Bad" = "#E83737", "Neutral" = "gray70")

  # Unified plot formatting for all groups
  p_abs <- ggplot(df_plot_abs, aes(x = .data[[x_var]], y = mean_val)) +
    geom_col(aes(fill = status_abs)) +
    geom_linerange(aes(ymin = mean_val - se_val, ymax = mean_val + se_val), alpha = 0.5, linewidth = 0.6) +
    geom_hline(data = df_global_avg, aes(yintercept = global_avg_abs), linetype = "dashed", color = "gray20", linewidth = 0.6, alpha = 0.8) +
    geom_point(aes(color = .data[[grp_col]]), y = -Inf, shape = 15, size = 3.5) +
    coord_flip(clip = "off") +
    facet_wrap(~ service, scales = facet_scales, ncol = 3) +
    scale_x_discrete(labels = function(x) gsub("__.*$", "", x)) +
    labs(title = "Absolute Change", 
         subtitle = NULL,
         caption = NULL,
         x = NULL, 
         y = "Absolute Difference (2020-1992)") +
    theme_minimal(base_size = 13) + 
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
          strip.text = element_text(face = "bold", size = 14))
          
  p_pct <- ggplot(df_plot_pct, aes(x = .data[[x_var]], y = sym_pct_change)) +
    geom_col(aes(fill = status_pct)) +
    geom_hline(data = df_global_avg, aes(yintercept = global_avg_pct), linetype = "dashed", color = "gray20", linewidth = 0.6, alpha = 0.8) +
    geom_point(aes(color = .data[[grp_col]]), y = -Inf, shape = 15, size = 3.5) +
    coord_flip(clip = "off") +
    facet_wrap(~ service, scales = facet_scales, ncol = 3) +
    scale_x_discrete(labels = function(x) gsub("__.*$", "", x)) +
    labs(title = "Percentage Change (%)", 
         subtitle = NULL,
         caption = NULL,
         x = NULL, 
         y = "Symmetric % Change (2020-1992)") +
    theme_minimal(base_size = 13) + 
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
          strip.text = element_text(face = "bold", size = 14))

  p_abs <- p_abs + scale_fill_manual(values = status_colors, guide = "none")
  p_pct <- p_pct + scale_fill_manual(values = status_colors, guide = "none")

  if (!is.null(custom_palette)) {
    if (grp == "biome") {
      p_abs <- p_abs + scale_color_manual(values = custom_palette, labels = biome_labels, na.value = "gray50", name = NULL)
      p_pct <- p_pct + scale_color_manual(values = custom_palette, labels = biome_labels, na.value = "gray50", name = NULL)
    } else {
      p_abs <- p_abs + scale_color_manual(values = custom_palette, na.value = "gray50", name = NULL)
      p_pct <- p_pct + scale_color_manual(values = custom_palette, na.value = "gray50", name = NULL)
    }
  } else {
    p_abs <- p_abs + scale_color_discrete(name = NULL)
    p_pct <- p_pct + scale_color_discrete(name = NULL)
  }

  # Stitch them together side-by-side using Patchwork
  combined_plot <- (p_abs | p_pct) + 
    plot_annotation(title = grp_name,
                    theme = theme(plot.title = element_text(size = 22, face = "bold", hjust = 0.5))) +
    plot_layout(guides = "collect") & 
    theme(legend.position = "bottom",
          legend.title = element_blank(),
          legend.text = element_text(size = 8),
          legend.key.size = unit(0.3, "cm")) &
    guides(color = guide_legend(ncol = legend_cols, override.aes = list(size = 4), reverse = TRUE))

  ggsave(file.path(out_dir, paste0(grp, "_combined_diffs.png")), combined_plot, width = 16, height = 9, bg="white", dpi=300)
}

message("\nAnalysis complete! Plots saved to ", out_dir)