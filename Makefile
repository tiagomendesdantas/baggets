# Development targets. Fixture regeneration needs R (see validation_r/README.md).

.PHONY: test lint fixtures calibrate

test:
	uv run pytest -n auto

lint:
	uv run ruff check src tests experiments validation_r/00_write_input_series.py

# Regenerate all R-reference fixtures (R 4.3+ with forecast, cluster, Mcomp).
# Order matters; 03 takes ~15-25 min.
fixtures:
	Rscript validation_r/export_m3_mcomp.R
	uv run python validation_r/00_write_input_series.py
	Rscript validation_r/01_export_component_fixtures.R
	Rscript validation_r/01b_pam_reference.R
	Rscript validation_r/02_ets_reference.R
	Rscript validation_r/03_run_reference_pipeline.R

# M2 engine gate: plain AutoETS on full M3 monthly vs the paper's Table 5 anchor
calibrate:
	uv run python experiments/calibrate_engine.py
