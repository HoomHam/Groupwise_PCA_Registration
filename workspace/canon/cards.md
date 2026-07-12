# PCA Registration — Cards

One card per load-bearing script. Origin: H/C/A/R(etro-inferred).
Retro-filled 2026-07-12; all origins R.

## Quick table

| ID | Script | Branch | Status |
|----|--------|--------|--------|
| C1 | register_groupwise_PCA_step.py (root) | stem | WORKS (F1 F2 F3 live) |
| C2 | workspace/helpers/scripts/register_hierarchical_gasonly.py | gas-only | WORKS |
| C3 | workspace/helpers/tests/register_elastix_oneshot.py | jacobian | WORKS |
| C4 | workspace/helpers/tests/register_ants_oneshot.py | jacobian | WORKS |
| C5 | workspace/helpers/scripts/register_elastix_oneshot_gas.py | gas-only | WORKS |
| C6 | workspace/helpers/scripts/make_video_grid.py | figures | WORKS (canonical) |
| C7 | workspace/helpers/analysis/*.m (MATLAB metrics suite) | analysis | WORKS |

## Cards

### C1 · register_groupwise_PCA_step.py (root, 2047 lines, ~half commented legacy)
- why: THE pipeline — 7-phase hierarchical groupwise PCA registration of dynamic lung frames (Logic.md: 4 → 7 → 14 → midpoint → apply-back → assemble 16 → final global 16)
- origin: R · branch: stem · facts: F1 F2 F3 F6 F8 F9 F12 F13
- in: 4 SimpleITK 4D channels via `read_files()` (:44) — gas, dp_rbc, dp_mem, clahe; config hardcoded :16-22 (`save_dir` on /Volumes/HoomHamExt, `endinhale=6`, `threshold=0.04`)
- out: registered 4D volumes + transform params on the external drive
- key funcs: `register_groupwise()` (:389, PCAMetric2 + BSplineStackTransform + ASGD, 4 resolutions, 1024 samples, RandomSparseMask) · `orchestrate_registration_workflow()` (:642, 4 stages gated by run_flags) · `_process_seven_block()` (:1261) + `step_groupwise_registration()` (:1389, phases 1&2 in ProcessPoolExecutor(max_workers=2)) · `register_EI_ants()` (:1914) · `main()` (:1989)
- status: WORKS — but F1 (cyclic dead), F2 (no SetOutputDirectory), F3 (typo'd grid-spacing key) are live defects

### C2 · workspace/helpers/scripts/register_hierarchical_gasonly.py
- why: bare-bones gas-only reproduction of the 4→7→14→16 hierarchy (9 Elastix registrations); imports helpers from C1 via importlib
- origin: R · branch: gas-only · facts: F10
- status: WORKS

### C3 · workspace/helpers/tests/register_elastix_oneshot.py
- why: gas-only single-pass Elastix with Jacobian outputs (`jac_det`, `jac_cons`, `jac_from_EE`)
- origin: R · branch: jacobian · facts: F4 F5 F11
- status: WORKS (CLAUDE.md code map still points at a root path that no longer exists — F11)

### C4 · workspace/helpers/tests/register_ants_oneshot.py
- why: ANTs counterpart for the Jacobian comparison; reference frame = EI bin 6 (NOT groupwise mean)
- origin: R · branch: jacobian · facts: F4 F12
- status: WORKS

### C5 · workspace/helpers/scripts/register_elastix_oneshot_gas.py
- why: gas-only oneshot variant against the new single-file recon format
- origin: R · branch: gas-only · facts: F10
- status: WORKS

### C6 · workspace/helpers/scripts/make_video_grid.py
- why: canonical grid-video renderer for registered series
- origin: R · branch: figures
- status: WORKS (other make_video_* / orientation_montage_* variants superseded)

### C7 · workspace/helpers/analysis/*.m
- why: MATLAB metric suite — `comprehensive_metrics.m`, `compare_registrations{,_500}.m`, `computePCAError.m`, `pca_error_{comparison,sorted}.m`, `image_difference_HH.m`, `print_{all_metrics,pca_stats}.m`, `CompReg.m`/`run_CompReg.m`
- origin: R · branch: analysis
- status: WORKS

## Scratch / reference (uncarded)
`archive/register_group_step_version_*.py` (20+ legacy pipelines — DO NOT EDIT, F15) ·
superseded video/diagnostic scripts (`make_video_23*.py`, `orientation_montage_23.py`) ·
`workspace/helpers/tests/test_v15_5iter.py` · `sonnetz_elastix_oneshots.txt` · `result.0.nii`.
