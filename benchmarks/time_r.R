# Time the ORIGINAL 2018 R implementation on the benchmark series.
#
# Conditions match benchmarks/time_python.py: same series (committed fixture
# training parts), same B, k fixed at 5 (no silhouette sweep), 100 members,
# one core, one warm-up ets() fit excluded from the timing. The original file
# is sourced verbatim except its library(TSclust) line, exactly as in
# validation_r/03_run_reference_pipeline.R.
#
# Run from the repo root (B=1000 takes about 12 minutes on one core):
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     Rscript benchmarks/time_r.R 1000

suppressPackageStartupMessages({
  library(forecast)
  library(parallel)
  library(cluster)
})

# shim for TSclust::diss (plain Euclidean distance, same column labels)
diss <- function(SERIES, METHOD) dist(t(as.matrix(SERIES)))
orig <- readLines(file.path("validation_r", "original", "baggedClusterETS.R"))
orig <- orig[!grepl("^\\s*library\\(TSclust\\)", orig)]
eval(parse(text = orig))

args <- commandArgs(trailingOnly = TRUE)
B <- if (length(args) >= 1) as.integer(args[1]) else 1000L
series <- c("monthly_N1603", "monthly_N2244", "monthly_N1908")

input_dir <- file.path("tests", "fixtures", "input")
manifest <- read.csv(file.path(input_dir, "manifest.csv"), stringsAsFactors = FALSE)
load_y <- function(sid) read.csv(file.path(input_dir, paste0(sid, ".csv")))$y

invisible(ets(ts(load_y(series[1]), frequency = 12)))  # warm-up, not timed

rows <- list()
for (sid in series) {
  row <- manifest[manifest$sid == sid, ]
  x <- ts(load_y(sid), frequency = row$m)
  t0 <- proc.time()
  set.seed(42)
  fit <- baggedClusterETS(x, cores = 1, nclusters = 5, silhouette = FALSE,
                          boot_samples = B)
  fc <- forecast.baggedClusterETS(fit, cores = 1, h = row$h)
  d <- proc.time() - t0
  wall <- d[["elapsed"]]
  cpu <- d[["user.self"]] + d[["sys.self"]]
  rows[[sid]] <- data.frame(sid = sid, n_train = length(x), B = B,
                            n_members = ncol(fc$forecasts_boot),
                            wall_seconds = wall, cpu_seconds = cpu)
  cat(sprintf("%s: n=%d, B=%d, %d members, wall %.1fs, cpu %.1fs\n",
              sid, length(x), B, ncol(fc$forecasts_boot), wall, cpu))
}

out <- do.call(rbind, rows)
path <- file.path("benchmarks", sprintf("timings_r_B%d.csv", B))
write.csv(out, path, row.names = FALSE)
cat(sprintf("total wall %.1fs, cpu %.1fs -> %s\n",
            sum(out$wall_seconds), sum(out$cpu_seconds), path))
cat(sprintf("%s, forecast %s, cluster %s\n", R.version.string,
            packageVersion("forecast"), packageVersion("cluster")))
