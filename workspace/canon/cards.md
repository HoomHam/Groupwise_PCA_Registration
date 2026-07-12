# PCA Registration — Cards

One card per load-bearing script. Origin: H/C/A/R(etro-inferred).
Retro-filled 2026-07-12; all origins R.

## Quick table

| ID | Script | Branch | Status |
|----|--------|--------|--------|
| C1 | register_groupwise_PCA_step.py (root) | trunk | WORKS — carries F12 F13 F41 defects |
| C2 | workspace/helpers/tests/register_elastix_oneshot.py | jacobian | WORKS (3.2 min, 025JC) |
| C3 | workspace/helpers/tests/register_ants_oneshot.py | jacobian | WORKS for CC (78.6 min); MI path never run (F18) |
| C4 | workspace/helpers/scripts/run_elastix_oneshot_23.py | new-format | WORKS (2.5 min) |
| C5 | workspace/helpers/scripts/register_elastix_oneshot_gas.py | new-format | WORKS (2.4 min, on 49.mat) |
| C6 | workspace/helpers/scripts/register_hierarchical_gasonly.py | new-format | WORKS (~10 min, on 23.mat) |
| C7 | workspace/helpers/scripts/make_video_grid.py | QC | WORKS (canonical) |
| C8 | workspace/helpers/analysis/*.m | analysis | UNKNOWN (dormant, predates 2026-07) |
| C9 | verify_env.py | preflight | UNKNOWN |

## Cards

### C1 · register_groupwise_PCA_step.py (root, 2047 lines, ~half commented legacy)
- why: THE canonical hierarchical 4→7→14→16 groupwise Elastix/PCAMetric2 pipeline; source of truth for every splice/transform helper the other scripts import
- origin: R · branch: trunk · facts: F12 F13 F22 F23 F30 F31 F32 F33 F36 F41
- in: `rspace*/cspace*/dspace*.mat` triplet dir (`input_files`, hardcoded :16-22); `run_flags`, `pad`, `iterations: List[str]`
- out: 4 aligned 4D images (gas, dp_rbc, dp_mem, clahe) → `save_dir` on /Volumes/HoomHamExt/; plus ANTs EI final step
- key funcs: `read_files()` :44 · `register_groupwise()` :389 (PCAMetric2 + BSplineStackTransform + ASGD, 4 resolutions, 1024 samples, RandomSparseMask, CheckNumberOfSamples=true :479) · `orchestrate_registration_workflow()` :642 · `_process_seven_block()` :1261 + `step_groupwise_registration()` :1389 (phases 1&2 in ProcessPoolExecutor(max_workers=2)) · `register_EI_ants()` :1914 · `main()` :1989
- status: WORKS, untouched by recent sessions. THREE live defects: F12 (cyclic dead), F13 (Elastix writes to CWD), F41 (grid-spacing key typo — silently ignored in every run to date)

### C2 · workspace/helpers/tests/register_elastix_oneshot.py
- why: single-pass gas-only Elastix groupwise reference implementation, with the three Jacobian outputs
- origin: R · branch: jacobian · facts: F8 F16 F20 F21
- in: rspace/cspace/dspace .mat triplet; `run_flags=(0,0,0)`; 500 iterations
- out: gas/rbc/mem .nii+.mat, mask, `jac_det.mat` (16,100³), `jac_cons.mat` (15,…), `jac_from_EE.mat` (16,…)
- status: WORKS (3.2 min verified on 025JC; also driven unmodified by C4)

### C3 · workspace/helpers/tests/register_ants_oneshot.py
- why: ANTs pairwise (frame→EI bin 6) counterpart for the Elastix-vs-ANTs comparison; interactive CC/MI metric chooser
- origin: R · branch: jacobian · facts: F15 F16 F17 F18
- out: `ants_oneshot_gas_CC/` or `_MI/` with jac_det/jac_cons/jac_from_EE (referenced to EI bin 6, NOT groupwise mean)
- status: WORKS for CC (78.6 min run done); MI path UNKNOWN — never run

### C4 · workspace/helpers/scripts/run_elastix_oneshot_23.py
- why: adapter letting the new single-file recon format drive the UNMODIFIED C2
- origin: R · branch: new-format · facts: F24 F34
- in: `workspace/data/recon/23.mat` (T-first, no CLAHE); fills the clahe4d slot with a duplicate of gas_phase
- out: `workspace/outputs/registered/23_oneshot/`
- status: WORKS (2.5 min, ran once)

### C5 · workspace/helpers/scripts/register_elastix_oneshot_gas.py
- why: bare-bones gas-only oneshot — no run_flags/pad/stage machinery, no ANTs, no rbc/mem/clahe
- origin: R · branch: new-format · facts: F24 F36
- in: hardcoded INPUT_MAT/SAVE_DIR at top — CURRENTLY pointing at `49.mat` / `49_gasonly/`; must edit to reuse
- out: gas, mask, jac_det/cons/from_EE. 1 Elastix pass, 500 iter, PCAMetric2, threshold 0.04, time_threshold 5, cyclic=True
- status: WORKS (2.4 min, ran once on 49.mat)

### C6 · workspace/helpers/scripts/register_hierarchical_gasonly.py
- why: bare-bones gas-only reproduction of the full 4→7→14→16 hierarchy, importing the root pipeline's helpers verbatim via importlib rather than reimplementing splice semantics
- origin: R · branch: new-format · facts: F12 F13 F33 F34 F40
- in: hardcoded INPUT_MAT/SAVE_DIR — CURRENTLY `23.mat` / `23_hierarchical_gasonly/`
- out: gas.nii/gas.mat + mask.mat only (no Jacobians); 9 Elastix registrations per run
- status: WORKS (~10 min, ran once on 23.mat). Faithfully preserves the dead-cyclic quirk (effective cyclic=False); works around the CWD-leak bug (F13) with os.chdir into a scratch dir

### C7 · workspace/helpers/scripts/make_video_grid.py
- why: canonical generalized QC renderer — the visual verification loop for every run
- origin: R · branch: QC · facts: F27 F29
- in: `python make_video_grid.py <reg_dir>` → reads `<reg_dir>/gas.mat`
- out: `<reg_dir>/gas_grid_sag_cor.mp4` — 2 rows (sagittal/coronal) × 6 cols, slices [10,26,42,57,73,89] from `pick_slices()` central-80% default, across all 16 timepoints
- status: WORKS — fully supersedes the three earlier video scripts

### C8 · workspace/helpers/analysis/*.m (MATLAB suite)
- why: post-hoc quantitative registration comparison — PCA error, difference images, metric tables
- files: CompReg.m, run_CompReg.m, compare_registrations{,_500}.m, comprehensive_metrics.m, computePCAError.m, image_difference_HH.m, pca_error_{comparison,sorted}.m, print_{all_metrics,pca_stats}.m
- origin: R · branch: analysis
- status: UNKNOWN — dormant, not referenced in either handoff; predates the 2026-07 work

### C9 · verify_env.py
- why: environment preflight (SimpleElastix wheel present, drive mounted) — matches CLAUDE.md's pre-flight checklist
- origin: R · branch: preflight · facts: F1 F10
- status: UNKNOWN (not mentioned in either handoff; shows as deleted in git status)

## Scratch / superseded (uncarded, safe to delete)
`make_video_23.py`, `make_video_23_sagittal.py`, `make_video_23_grid.py` (superseded by C7) ·
`orientation_montage_23.py` (one-off diagnostic that produced the axis ground truth, F27) ·
`workspace/helpers/tests/`: run_SyrT1_1k_enhpad.py, run_SyrT1_54321k.py, run_ants_on_gw.py,
run_comparison.py, test_v15_5iter.py (older ablation/test drivers, status UNKNOWN) ·
`workspace/analysis/` is empty · `archive/register_group_step_version_*.py` (20+ legacy — DO NOT EDIT, F37).

No underscore-prefixed scratch scripts exist in this repo.
