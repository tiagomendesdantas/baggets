# Tier-1 reference fixtures from R for the deterministic bootstrap components:
# Guerrero Box-Cox lambda, the Box-Cox-transformed series, STL/loess
# decomposition, and reference MBB draw distributions.
#
# Reads  tests/fixtures/input/*.csv (written by 00_write_input_series.py)
# Writes tests/fixtures/r_components/{sid}.csv   (x_bc, trend, seasonal, remainder)
#        tests/fixtures/r_components/lambdas.csv (sid, m, n, lambda)
#        tests/fixtures/r_mbb/{sid}_perdraw_sd.csv, {sid}_pooled.csv
#
# Run from the repo root:  Rscript validation_r/01_export_component_fixtures.R

suppressPackageStartupMessages(library(forecast))

input_dir <- file.path("tests", "fixtures", "input")
comp_dir <- file.path("tests", "fixtures", "r_components")
mbb_dir <- file.path("tests", "fixtures", "r_mbb")
dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(mbb_dir, recursive = TRUE, showWarnings = FALSE)

# The original MBB implementation, verbatim (validation_r/original/utils.R)
source(file.path("validation_r", "original", "utils.R"))

manifest <- read.csv(file.path(input_dir, "manifest.csv"), stringsAsFactors = FALSE)

# Series whose remainder gets the MBB reference-draw treatment
mbb_sids <- c("monthly_N1861", "synthetic_ar1")

lambda_rows <- list()

for (i in seq_len(nrow(manifest))) {
  sid <- manifest$sid[i]
  m <- manifest$m[i]
  y <- read.csv(file.path(input_dir, paste0(sid, ".csv")))$y
  x <- ts(y, frequency = m)

  lambda <- BoxCox.lambda(x, lower = 0, upper = 1)
  x.bc <- BoxCox(x, lambda)

  # Decomposition exactly as in bld.mbb.bootstrap (Bergmeir et al. 2016 /
  # validation_r/original/baggedClusterETS.R)
  if (m > 1) {
    x.stl <- stl(ts(x.bc, frequency = m), "per")$time.series
    seasonal <- as.numeric(x.stl[, 1])
    trend <- as.numeric(x.stl[, 2])
    remainder <- as.numeric(x.stl[, 3])
  } else {
    tt <- 1:length(x)
    suppressWarnings(x.loess <- loess(x.bc ~ tt, span = 6 / length(x), degree = 1))
    seasonal <- rep(0, length(x))
    trend <- as.numeric(fitted(x.loess))
    remainder <- as.numeric(residuals(x.loess))
  }

  write.csv(
    data.frame(x_bc = as.numeric(x.bc), trend = trend, seasonal = seasonal,
               remainder = remainder),
    file.path(comp_dir, paste0(sid, ".csv")),
    row.names = FALSE, quote = FALSE
  )
  lambda_rows[[i]] <- data.frame(sid = sid, m = m, n = length(y), lambda = lambda)

  if (sid %in% mbb_sids) {
    block_size <- if (m > 1) 2 * m else 8
    set.seed(42)
    n_draws <- 2000
    per_draw_sd <- numeric(n_draws)
    pooled <- list()
    for (d in 1:n_draws) {
      draw <- MBB(remainder, block_size)
      per_draw_sd[d] <- sd(draw)
      if (d <= 50) pooled[[d]] <- draw
    }
    write.csv(data.frame(sd = per_draw_sd),
              file.path(mbb_dir, paste0(sid, "_perdraw_sd.csv")),
              row.names = FALSE, quote = FALSE)
    write.csv(data.frame(value = unlist(pooled)),
              file.path(mbb_dir, paste0(sid, "_pooled.csv")),
              row.names = FALSE, quote = FALSE)
    cat(sprintf("%s: MBB reference draws written (block=%d)\n", sid, block_size))
  }
  cat(sprintf("%s: lambda=%.6f, n=%d, m=%d\n", sid, lambda, length(y), m))
}

write.csv(do.call(rbind, lambda_rows),
          file.path(comp_dir, "lambdas.csv"), row.names = FALSE, quote = FALSE)
cat("done\n")
