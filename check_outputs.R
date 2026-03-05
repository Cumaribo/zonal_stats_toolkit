library(sf)

# Load path helper
if (file.exists("Paths.R")) source("Paths.R") else source("paths.R")

# Path to the generated GPKG
gpkg_path <- project_dir("output_raw_fid", "grid_10km_with_stats.gpkg")

if (file.exists(gpkg_path)) {
  message("Reading metadata from: ", gpkg_path)
  
  # Get layer info
  layer_info <- st_layers(gpkg_path)
  layer_name <- layer_info$name[1]
  
  # Read just the header/first row to get column names
  # We use a query to limit data transfer
  sample_data <- st_read(gpkg_path, query = paste("SELECT * FROM", layer_name, "LIMIT 1"), quiet = TRUE)
  
  message("\n--- GPKG INFO ---")
  message("Layer: ", layer_name)
  message("Features: ", layer_info$features[1])
  message("Columns: ", ncol(sample_data))
  message("\n--- COLUMN NAMES ---")
  print(colnames(sample_data))
  
} else {
  stop("File not found: ", gpkg_path)
}