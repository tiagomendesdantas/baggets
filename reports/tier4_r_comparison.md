<!-- Output of `uv run python experiments/tier4_compare.py python` then `... compare`, run on 26 Aug 2026 (macOS arm64) against the original `baggedClusterETS.R` in `validation_r/original/`, on the code as of that date. Not rerun since; later commits (e.g. the pluggable engine protocol, 9 Sep) are covered by the CI tests, not by this report. Committed verbatim on 13 Oct 2026; regenerate with the same two commands (~15-20 min). -->

# Tier-4: original R pipeline vs Python (10 seeds per case)

Gates apply to the engine-agreeing subset (|plain-ETS delta| <= 0.5pp); engine-divergent cases carry their attribution. See phase_compare docstring for the diagnosis.

## Engine-agreeing cases (gated)

    sid variant  engines_agree  engine_delta  form_match  smape_r  smape_py_mean  smape_py_sd  in_band  delta  delta_ok  rms_ratio  k_r
m:N1861    auto           True         0.208       False    5.237          5.394        0.079    False  0.156      True      2.144   83
m:N1861      k5           True         0.208       False    5.350          5.352        0.125     True  0.002      True      2.570    5
m:N1908      k5           True        -0.166        True    5.464          5.360        0.162     True -0.104      True      1.003    5
m:N2244      k5           True         0.002        True    2.374          2.374        0.045     True -0.000      True      0.711    5

## Engine-divergent cases (attributed, not gated)

    sid variant  engines_agree  engine_delta  form_match  smape_r  smape_py_mean  smape_py_sd  in_band  delta  delta_ok  rms_ratio  k_r
m:N1603      k5          False        -4.301       False   40.385         43.676        0.278    False  3.291     False     12.555    5
m:N2501    auto          False        -4.678       False    6.823          4.009        0.021    False -2.814     False     26.053   99
m:N2501      k5          False        -4.678       False    6.990          4.029        0.029    False -2.962     False     27.528    5
m:N2572      k5          False        -2.371        True    7.937          6.729        0.275    False -1.208      True      4.912    5
m:N2603      k5          False        -1.986       False   29.116         28.948        1.154     True -0.169      True      0.914    5
q:N1359      k5          False         0.913       False    5.420          5.053        0.119    False -0.367      True      2.884    5

G1 R-inside-seed-band, <=1 miss of 4: PASS
G2 per-case delta + mean signed delta +-0.5pp: PASS (mean +0.014pp)
G3 retired (see phase_compare source): rms_ratio reported as diagnostic only
Attribution check (|pipeline delta| <= |engine delta| + 1.5pp on divergent cases): PASS

Pipeline-mechanics equivalence (all 10 cases): member counts 99-135 match R's allocation wobble; k regimes match (k=5 fixed; silhouette auto drifting high on bootstrap clouds, 52-100 vs R's 83/99); 0 fallbacks.
