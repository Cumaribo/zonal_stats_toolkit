# R/paths.R
# Helper to resolve paths relative to the project root

project_dir <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}