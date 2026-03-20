library(dplyr)
library(tidyr)
library(ggplot2)
library(sf)

# Groupings to process
groupings <- c(
  "region_wb" = "region_wb",
  "income_grp" = "income_grp",
  "biome" = "WWF_biome"
)

out_dir <- "outputs/plots/method_comparison"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (job in names(groupings)) {
  key <- groupings[[job]]
  
  file_b <- file.path("output_consolidated", paste0(job, "_services_change.gpkg"))
  file_c <- file.path("output_diff_consolidated", paste0(job, "_services_diff.gpkg"))
  
  if (!file.exists(file_b) || !file.exists(file_c)) {
    message("Skipping ", job, " - missing one of the GPKGs.")
    next
  }
  
  message("Comparing ", job, "...")
  
  df_b <- st_read(file_b, quiet = TRUE) %>% st_drop_geometry()
  df_c <- st_read(file_c, quiet = TRUE) %>% st_drop_geometry()
  
  # Select only absolute change columns and pivot
  cols_b <- c(key, grep("_abs_chg$", names(df_b), value = TRUE))
  cols_c <- c(key, grep("_abs_chg$", names(df_c), value = TRUE))
  
  long_b <- df_b %>% select(all_of(cols_b)) %>% pivot_longer(-!!sym(key), names_to = "service", values_to = "abs_chg") %>% mutate(method = "Path B (Diff of Base Means)")
  long_c <- df_c %>% select(all_of(cols_c)) %>% pivot_longer(-!!sym(key), names_to = "service", values_to = "abs_chg") %>% mutate(method = "Path C (Mean of Diff Rasters)")
  
  # Combine and clean service names
  df_combined <- bind_rows(long_b, long_c) %>%
    mutate(service = sub("_abs_chg", "", service)) %>%
    filter(!is.na(!!sym(key)))
    
  # Calculate exact mathematical divergence
  df_diff <- df_combined %>%
    pivot_wider(names_from = method, values_from = abs_chg) %>%
    mutate(absolute_discrepancy = `Path B (Diff of Base Means)` - `Path C (Mean of Diff Rasters)`)
    
  write.csv(df_diff, file.path(out_dir, paste0(job, "_method_variance.csv")), row.names = FALSE)
  
  # Plot side-by-side grouped bars
  p <- ggplot(df_combined, aes(x = !!sym(key), y = abs_chg, fill = method)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    facet_wrap(~ service, scales = "free_x") +
    coord_flip() +
    scale_fill_manual(values = c("Path B (Diff of Base Means)" = "#1f77b4", "Path C (Mean of Diff Rasters)" = "#ff7f0e")) +
    labs(title = paste("Method Comparison: Absolute Change by", job), x = NULL, y = "Absolute Change", fill = "Calculation Pathway") +
    theme_minimal() + theme(legend.position = "bottom", strip.background = element_rect(fill = "#f0f0f0", color = NA))
    
  ggsave(file.path(out_dir, paste0(job, "_method_comparison.png")), p, width = 14, height = 10, bg = "white")
}