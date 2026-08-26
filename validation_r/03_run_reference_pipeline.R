# Tier-4 reference: run the ORIGINAL 2018 baggedClusterETS.R end-to-end.
#
# The original file is sourced verbatim except for its `library(TSclust)` line
# (TSclust is not installed); its only use, diss(X, "EUCL"), is plain Euclidean
# distance between the mts columns, shimmed below with identical labels
# ("Series k") that the original code's name-parsing relies on.
#
# Series: the M3 fixture series whose training part survives the original
# code's short-series branch (it references undefined variables and crashes
# when length(y) - h_pseudo <= h_pseudo — e.g. 48-obs monthly series).
#
# Writes tests/fixtures/r_e2e/forecasts.csv  (sid, variant, step, median, mean)
#        tests/fixtures/r_e2e/ensembles.csv  (sid, variant, member, step, value)
#        tests/fixtures/r_e2e/info.csv       (sid, variant, k, n_members)
#
# Run from the repo root (~15-25 min):  Rscript validation_r/03_run_reference_pipeline.R

suppressPackageStartupMessages({
  library(forecast)
  library(parallel)
  library(cluster)
})

# --- shim for TSclust::diss, preserving mts column labels ("Series k") ------
diss <- function(SERIES, METHOD) dist(t(as.matrix(SERIES)))

# --- source the original implementation, skipping only the TSclust line -----
orig <- readLines(file.path("validation_r", "original", "baggedClusterETS.R"))
orig <- orig[!grepl("^\\s*library\\(TSclust\\)", orig)]
eval(parse(text = orig))

input_dir <- file.path("tests", "fixtures", "input")
manifest <- read.csv(file.path(input_dir, "manifest.csv"), stringsAsFactors = FALSE)

# survives the original short-series branch: n_train - h_pseudo > h_pseudo
h_pseudo_of <- function(m) ifelse(m > 1, 2 * m, 10)
runnable <- manifest[manifest$n_train - h_pseudo_of(manifest$m) > h_pseudo_of(manifest$m) &
                     manifest$sid != "synthetic_ar1", ]
cat("runnable series:", paste(runnable$sid, collapse = ", "), "\n")

# silhouette (automatic k) variant only for two series — it is slow in R
auto_sids <- c("monthly_N1861", "monthly_N2501")

fc_rows <- list()
ens_rows <- list()
info_rows <- list()

for (i in seq_len(nrow(runnable))) {
  sid <- runnable$sid[i]
  m <- runnable$m[i]
  h <- runnable$h[i]
  y <- read.csv(file.path(input_dir, paste0(sid, ".csv")))$y
  x <- ts(y, frequency = m)

  variants <- if (sid %in% auto_sids) c("k5", "auto") else "k5"
  for (variant in variants) {
    t0 <- Sys.time()
    set.seed(42)
    fit <- baggedClusterETS(x, cores = 6, nclusters = 5,
                            silhouette = (variant == "auto"), boot_samples = 1000)
    fc <- forecast.baggedClusterETS(fit, cores = 6, h = h)
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    fc_rows[[paste(sid, variant)]] <- data.frame(
      sid = sid, variant = variant, step = seq_len(h),
      median = as.numeric(fc$median), mean = as.numeric(fc$mean)
    )
    ens <- fc$forecasts_boot  # h x n_members
    ens_rows[[paste(sid, variant)]] <- do.call(rbind, lapply(seq_len(ncol(ens)), function(j) {
      data.frame(sid = sid, variant = variant, member = j, step = seq_len(h),
                 value = ens[, j])
    }))
    info_rows[[paste(sid, variant)]] <- data.frame(
      sid = sid, variant = variant, k = fit$k, n_members = ncol(ens)
    )
    cat(sprintf("%s [%s]: k=%s, members=%d, %.0fs\n",
                sid, variant, fit$k, ncol(ens), elapsed))
  }
}

out_dir <- file.path("tests", "fixtures", "r_e2e")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, fc_rows), file.path(out_dir, "forecasts.csv"), row.names = FALSE)
write.csv(do.call(rbind, ens_rows), file.path(out_dir, "ensembles.csv"), row.names = FALSE)
write.csv(do.call(rbind, info_rows), file.path(out_dir, "info.csv"), row.names = FALSE)
cat("done\n")
