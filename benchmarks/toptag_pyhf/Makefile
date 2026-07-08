PY ?= python
E79_RUN ?= runs/20260703-121216-e79-toptag-score-template-export

.PHONY: summarize templates workspace stress

summarize:
	$(PY) scripts/summarize_likelihood_results.py

templates:
	$(PY) scripts/make_toptag_templates.py --seed 42 --shard shard000

workspace:
	$(PY) scripts/build_pyhf_workspace.py --e79-run-dir $(E79_RUN)

stress:
	$(PY) scripts/fit_pyhf_workspace.py --e79-run-dir $(E79_RUN)
