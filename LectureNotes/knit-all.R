# Remove all .knit.md files
knit_files <- list.files(".", pattern = "\\.knit\\.md$", recursive = TRUE, full.names = TRUE)
file.remove(knit_files)

# Get all directories
dirs <- list.dirs(".", recursive = FALSE, full.names = FALSE)
dirs <- dirs[grepl("^\\d", dirs)]  # Only directories starting with numbers

# Knit each one
for(dir in dirs) {
  # Extract the number from directory name
  num <- stringr::str_extract(dir, "^\\d+[a-z]*")
  rmd_file <- file.path(dir, paste0(num, "slides.Rmd"))
  
  if(file.exists(rmd_file)) {
    cat("Knitting:", rmd_file, "\n")
    rmarkdown::render(rmd_file)
    Sys.sleep(3)
  } else {
    cat("File not found:", rmd_file, "\n")
  }
}