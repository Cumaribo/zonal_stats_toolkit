# R/paths.R
# Helper to resolve paths relative to the project root

project_dir <- function(...) {
  # Use getwd() directly to avoid issues when .Rproj is missing
  file.path(getwd(), ...)
}