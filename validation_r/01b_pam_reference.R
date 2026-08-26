# Tier-3 reference: cluster::pam partitions and silhouette widths on the fixed
# distance matrix written by 00_write_input_series.py.
#
# Reads  tests/fixtures/input/pam_distance.csv
# Writes tests/fixtures/r_pam/labels.csv      (k, point, label) for k in {2,5,20,50}
#        tests/fixtures/r_pam/silhouette.csv  (k, avg_width)    for k in 2..60
#
# Run from the repo root:  Rscript validation_r/01b_pam_reference.R

suppressPackageStartupMessages(library(cluster))

D <- as.matrix(read.csv(file.path("tests", "fixtures", "input", "pam_distance.csv"),
                        header = FALSE))
D <- as.dist(D)

out_dir <- file.path("tests", "fixtures", "r_pam")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

label_rows <- list()
for (k in c(2, 5, 20, 50)) {
  fit <- pam(D, k = k)
  label_rows[[as.character(k)]] <- data.frame(
    k = k, point = seq_along(fit$clustering), label = as.integer(fit$clustering)
  )
}
write.csv(do.call(rbind, label_rows), file.path(out_dir, "labels.csv"),
          row.names = FALSE, quote = FALSE)

sil_rows <- list()
for (k in 2:60) {
  fit <- pam(D, k = k)
  sil_rows[[as.character(k)]] <- data.frame(k = k, avg_width = fit$silinfo$avg.width)
}
write.csv(do.call(rbind, sil_rows), file.path(out_dir, "silhouette.csv"),
          row.names = FALSE, quote = FALSE)
cat("done\n")
