# Tier-2 reference: R forecast::ets automatic fits on 30 M3 monthly series.
#
# Reads  data/m3/ (Mcomp export) and tests/fixtures/input/ets_reference_ids.csv
# Writes tests/fixtures/r_ets/reference.csv (uid, method, step, value)
#
# Run from the repo root:  Rscript validation_r/02_ets_reference.R

suppressPackageStartupMessages(library(forecast))

ids <- read.csv(file.path("tests", "fixtures", "input", "ets_reference_ids.csv"),
                stringsAsFactors = FALSE)$uid
values <- read.csv(file.path("data", "m3", "m3_monthly.csv"), stringsAsFactors = FALSE)
meta <- read.csv(file.path("data", "m3", "meta.csv"), stringsAsFactors = FALSE)

out_dir <- file.path("tests", "fixtures", "r_ets")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rows <- list()
for (uid in ids) {
  n_train <- meta$n_train[meta$unique_id == uid]
  h <- meta$h[meta$unique_id == uid]
  sv <- values[values$unique_id == uid, ]
  sv <- sv[order(sv$idx), ]
  y_train <- sv$y[sv$split == "train"]
  stopifnot(length(y_train) == n_train)

  x <- ts(y_train, frequency = 12)
  fit <- ets(x)
  fc <- as.numeric(forecast(fit, h = h, PI = FALSE)$mean)
  rows[[uid]] <- data.frame(uid = uid, method = fit$method, step = seq_len(h), value = fc)
  cat(sprintf("%s: %s\n", uid, fit$method))
}

# NOTE: methods like "ETS(M,A,N)" contain commas — keep quoting on
write.csv(do.call(rbind, rows), file.path(out_dir, "reference.csv"),
          row.names = FALSE)
cat("done\n")
