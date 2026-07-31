# Event Representation Reliability Review

Anonymous review package for reproducing the representation-reliability checks reported in the paper.

Large ROOT files, TopTag HDF5 files, derived tensors, checkpoints and run directories are not tracked in git. The repository keeps the code, small manifests, source-data CSV files and benchmark summaries used by the reported checks. A citable archive of the derived tensors and checkpoints, with file-level checksums, is still pending; until it is deposited, this repository alone does not provide end-to-end reproduction of trained-model results.

## Contents

- `scripts/`: experiment and figure-generation entry points.
- `benchmarks/toptag_pyhf/`: TopTag score-template and `pyhf` benchmark package.
- `data/`: public-data and external-asset manifests.
- `figures/source_data_*.csv`: source data for manuscript figures; `figures/source_data_manifest.sha256` records their file hashes.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU execution is recommended for training runs. Figure generation and small smoke checks can run on CPU.

## CMS H4l Checks

Download the public CMS H4l ROOT inputs:

```bash
scripts/download_cms_h4l_open_data.sh all
```

The files are written under:

```text
data/cms_h4l_2012_reduced_nanoaod/root_files/
```

Regenerate the H4l tensor and main split-branch checks:

```bash
python scripts/e65_cms_h4l_event_tensor_export.py \
  --output-name cms_h4l_mc_candidates_e65.npz

python scripts/e71_cms_h4l_evenet_bridge_feasibility.py \
  --tensor-npz data_processed/cms_h4l_e65/cms_h4l_mc_candidates_e65.npz \
  --teacher-ckpt data/checkpoints/teachers/evenet_public.ckpt

python scripts/e72_cms_h4l_evenet_embedding_split_branch.py \
  --tensor-npz data_processed/cms_h4l_e65/cms_h4l_mc_candidates_e65.npz \
  --teacher-ckpt data/checkpoints/teachers/evenet_public.ckpt \
  --max-events 0 --epochs 5 --probe-epochs 20

for seed in 41 42 43; do
  python scripts/e73_cms_h4l_evenet_domain_design_repeat.py \
    --domain-set mixed_visible \
    --tensor-npz data_processed/cms_h4l_e65/cms_h4l_mc_candidates_e65.npz \
    --teacher-ckpt data/checkpoints/teachers/evenet_public.ckpt \
    --max-events 0 --epochs 5 --probe-epochs 20 \
    --adv-lambda 0.5 --orth-lambda 0.5 --seed "$seed"
done

python scripts/e69b_cms_h4l_observed_data_sanity.py
```

The EveNet checkout defaults to `external/EveNet_Public`, and the public EveNet checkpoint defaults to `data/checkpoints/teachers/evenet_public.ckpt`. Override them with `EVENET_REPO` and `EVENET_TEACHER_CKPT` if needed.

## TopTag Checks

The TopTag workflow uses selected shards from CERN Open Data record `80030` for systematic-transfer checks.

```bash
python scripts/e75a_toptag_record80030_preflight.py
python scripts/e75b_toptag_record80030_schema_smoke.py

for seed in 41 42 43; do
  python scripts/e76b_toptag_constituent_encoder.py \
    --max-events-per-domain 100000 --epochs 6 --probe-epochs 8 \
    --orth-lambda 0.25 --seed "$seed"
done

python scripts/e77_toptag_heldout_systematic_audit.py \
  --max-events-per-domain 100000 --epochs 6 --probe-epochs 8

python scripts/e78_toptag_reference_calibration.py \
  --train-events 300000 --val-events 150000 --epochs 8
```

The likelihood-facing TopTag benchmark is packaged separately. Inspect the
packaged cross-shard summary first:

```bash
python benchmarks/toptag_pyhf/scripts/summarize_likelihood_results.py
```

Workspace regeneration requires an E79 run directory containing
`score_templates.csv`. That run directory is part of the pending derived-artifact
archive and is not tracked here. After the archive is deposited and unpacked,
use its documented concrete run path as follows:

```bash
python benchmarks/toptag_pyhf/scripts/build_pyhf_workspace.py --e79-run-dir "$E79_RUN_DIR"
python benchmarks/toptag_pyhf/scripts/fit_pyhf_workspace.py --e79-run-dir "$E79_RUN_DIR"
```

Set `E79_RUN_DIR` to the exact archive path recorded in the archive manifest.
The command is intentionally not presented as currently runnable because no
archive identifier or manifest path has yet been released.

`benchmarks/toptag_pyhf/outputs/` contains the E91d summary tables used for the current interpretation. `benchmarks/toptag_pyhf/workspaces/` contains small example `pyhf` workspaces for reviewer inspection.

## Figures

Regenerate the manuscript panels from the included source-data CSV files:

```bash
python scripts/make_submission_figures.py
shasum -a 256 -c figures/source_data_manifest.sha256
```

The script writes PDF, SVG and TIFF panels under `figures/`. Generated figure files are ignored by git; the tracked files are the source-data CSV tables.

## Reproducibility Notes

The package supports inspection of representation-level and likelihood-facing reliability checks for the CMS H4l and TopTag workflows. Inputs are public CMS H4l files and CERN Open Data TopTag shards. Figure files can be regenerated from the tracked source CSV files. Generated tensors and model checkpoints require either regeneration from the public inputs or the pending citable archive and are ignored by git.
