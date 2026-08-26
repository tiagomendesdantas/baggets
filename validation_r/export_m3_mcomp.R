# Export the M3 competition data from the Mcomp R package to CSV.
#
# Mcomp is the authoritative M3 source (and what the 2018 paper used). The
# datasetsforecast Python loader points at a dead forecasters.org URL, so the
# Python side loads these exports instead (see src/baggets/datasets.py).
#
# Output: data/m3/m3_{yearly,quarterly,monthly,other}.csv with columns
#   unique_id, idx (1-based within series), split (train|test), y
# plus data/m3/meta.csv with unique_id, group, m, n_train, h.
#
# Run from the repo root:  Rscript validation_r/export_m3_mcomp.R

suppressPackageStartupMessages(library(Mcomp))

out_dir <- file.path("data", "m3")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

groups <- list(
  yearly    = "YEARLY",
  quarterly = "QUARTERLY",
  monthly   = "MONTHLY",
  other     = "OTHER"
)

meta_rows <- list()

for (gname in names(groups)) {
  period <- groups[[gname]]
  items <- Filter(function(s) s$period == period, M3)
  rows <- vector("list", length(items))
  for (i in seq_along(items)) {
    s <- items[[i]]
    y <- c(as.numeric(s$x), as.numeric(s$xx))
    n_train <- length(s$x)
    rows[[i]] <- data.frame(
      unique_id = s$sn,
      idx = seq_along(y),
      split = c(rep("train", n_train), rep("test", length(s$xx))),
      y = y
    )
    meta_rows[[length(meta_rows) + 1]] <- data.frame(
      unique_id = s$sn,
      group = gname,
      m = stats::frequency(s$x),
      n_train = n_train,
      h = length(s$xx)
    )
  }
  df <- do.call(rbind, rows)
  path <- file.path(out_dir, paste0("m3_", gname, ".csv"))
  write.csv(df, path, row.names = FALSE, quote = FALSE)
  cat(sprintf("%s: %d series, %d rows -> %s\n", gname, length(items), nrow(df), path))
}

meta <- do.call(rbind, meta_rows)
write.csv(meta, file.path(out_dir, "meta.csv"), row.names = FALSE, quote = FALSE)
cat(sprintf("meta: %d series -> %s\n", nrow(meta), file.path(out_dir, "meta.csv")))
