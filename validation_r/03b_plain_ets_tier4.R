# Plain forecast::ets on the Tier-4 series — the engine-attribution covariate
# for experiments/tier4_compare.py (written after Tier-4 diagnosis: per-series
# ensemble divergence is dominated by engine differences, so the comparison
# report carries the plain-engine delta alongside the pipeline delta).
#
# Writes tests/fixtures/r_e2e/plain_ets.csv (sid, method, step, value)
# Run from repo root:  Rscript validation_r/03b_plain_ets_tier4.R
suppressPackageStartupMessages(library(forecast))

input_dir <- file.path("tests", "fixtures", "input")
manifest <- read.csv(file.path(input_dir, "manifest.csv"), stringsAsFactors = FALSE)
h_pseudo_of <- function(m) ifelse(m > 1, 2 * m, 10)
runnable <- manifest[manifest$n_train - h_pseudo_of(manifest$m) > h_pseudo_of(manifest$m) &
                     manifest$sid != "synthetic_ar1", ]

rows <- list()
for (i in seq_len(nrow(runnable))) {
  sid <- runnable$sid[i]; m <- runnable$m[i]; h <- runnable$h[i]
  y <- read.csv(file.path(input_dir, paste0(sid, ".csv")))$y
  fit <- ets(ts(y, frequency = m))
  fc <- as.numeric(forecast(fit, h = h, PI = FALSE)$mean)
  rows[[sid]] <- data.frame(sid = sid, method = fit$method, step = seq_len(h), value = fc)
  cat(sprintf("%s: %s\n", sid, fit$method))
}
write.csv(do.call(rbind, rows), file.path("tests", "fixtures", "r_e2e", "plain_ets.csv"),
          row.names = FALSE)
