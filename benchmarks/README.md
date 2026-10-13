# Timing against the original R code

How long the full method takes in baggets and in the original 2018 R code
(`validation_r/original/baggedClusterETS.R`), under matched conditions.

## Conditions

- **Same series**: three committed fixture series (training parts of M3
  monthly N1603, N2244 and N1908, 51 to 126 points), forecast horizon 18.
- **Same configuration**: B = 1,000 bootstraps, k fixed at 5 (no silhouette
  sweep), about 100 members, the paper's constants (`ClusterSelection.paper(n_clusters=5)`
  in Python; `baggedClusterETS(..., nclusters = 5, silhouette = FALSE)` in R).
- **One core on each side**: `cores = 1` in R, `n_jobs = 1` in Python, BLAS and
  numba threads set to 1.
- **Timed**: fit plus forecast for each series, after one untimed warm-up ETS
  fit (which pays numba's compile or cache load in Python).
- **Machine**: Apple M2 (8 cores), macOS 26.2, on AC power, sleep prevented with
  `caffeinate -i`. A desktop session, not an idle machine. R 4.3.3 with
  forecast 8.22.0 and cluster 2.1.6; Python 3.12.3 with baggets 0.1.0,
  statsforecast 2.1.1 and numpy 2.5.2. Python ran first, then R.

## Result

Wall-clock seconds per series (CPU seconds are in the CSVs and within 2% of these):

| series | points | original R | baggets | R / baggets |
|---|---|---|---|---|
| N1603 | 51 | 224.3 | 69.4 | 3.2 |
| N2244 | 116 | 262.4 | 168.8 | 1.6 |
| N1908 | 126 | 252.2 | 183.1 | 1.4 |
| total | | 738.9 | 421.3 | 1.8 |

On these three series baggets took between 1.4 and 3.2 times less time than the
original R code. Nearly all of it goes to about 1,100 automatic ETS fits per
series (1,000 on validation windows, 100 final). baggets runs statsforecast's
optimizer with a tighter tolerance and a higher iteration cap than its default,
which `src/baggets/engine.py` reports as about 3 times the cost per fit; the
default settings are faster but did not match R's model choices.

Three series on one machine is a small sample. Treat the ratio as a rough
indication, not a constant.

## Reproduce

From the repo root (R with `forecast` and `cluster`; about 7 minutes for Python
and 12 for R on the machine above):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run python benchmarks/time_python.py 1000   # writes benchmarks/timings_python_B1000.csv
Rscript benchmarks/time_r.R 1000               # writes benchmarks/timings_r_B1000.csv
```

Below B = 299 the two are not comparable: the R code then selects far
fewer than 100 members (its allocation divides by a fixed 299), while baggets
keeps 100.
