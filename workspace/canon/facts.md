# PCA Registration — Facts (belief table)

Flip Status, never delete rows. VALID / RETRACTED / SUSPECT / UNKNOWN.
Retro-filled 2026-07-12 from CLAUDE.md + Logic.md + handoff-report.md (root, 06-03/07-06)
+ workspace/handoff-report.md (07-06/07).

## Environment / build

| ID | Claim | Status | Since | Note |
|----|-------|--------|-------|------|
| F1 | Plain SimpleITK has no `ElastixImageFilter` on macOS arm64 — must install the `SimpleITK-SimpleElastix` wheel | VALID | — | CLAUDE.md |
| F2 | "`CheckNumberOfSamples` is `['false']`" (old CLAUDE.md claim) | RETRACTED | 2026-06-03 | it is `['true']` in BOTH register_groupwise_PCA_step.py:479 and register_elastix_oneshot.py; doc corrected |
| F3 | `CheckNumberOfSamples=true` can crash PCAMetric2 with "samples outside buffer" on small/masked lung volumes → flip to false if it happens | SUSPECT | 2026-06-03 | predicted, never observed |
| F4 | The predicted CheckNumberOfSamples crash did NOT occur — 100³ volumes fine across every run (oneshot + 9-registration hierarchical) | VALID | 2026-07-07 | supersedes F3 in practice |
| F5 | `sitk.Multiply` is unsupported for 4D images in this build | VALID | 2026-06-03 | |
| F6 | `CopyInformation` / `SetDirection` on 4D images fails in this build | VALID | 2026-06-03 | |
| F7 | `ComputeJacobianDeterminantOn()` is the WRONG method name in this SimpleElastix build | RETRACTED | 2026-06-03 | use `ComputeDeterminantOfSpatialJacobianOn()` (which writes no file) |
| F8 | LAW (from F5–F7): keep the Jacobian entirely as numpy `(T,Z,Y,X)`, convert to SimpleITK only at save time via `JoinSeries` of per-frame 3D images | VALID | 2026-06-03 | settled pattern |
| F9 | Iterations must be passed as `List[str]` (`['5000','4000']`), never ints — Elastix type-errors or silently mis-parses | VALID | — | law |
| F10 | External drive `/Volumes/HoomHamExt/` must be mounted before `main()` — no runtime check until first write | VALID | — | pre-flight |
| F11 | Elastix temp dirs accumulate to GB — `cleanup_elastix_temp_files(phase_dir)` must run after each phase | VALID | — | |

## Pipeline defects (live, unfixed)

| ID | Claim | Status | Since | Note |
|----|-------|--------|-------|------|
| F12 | `cyclic: bool` arg of `orchestrate_registration_workflow()` is DEAD — every internal `register_groupwise()` call hardcodes `cyclic=False`. Hierarchical phases are NOT using cyclic transforms despite intent. | VALID | 2026-07-07 | confirmed by grep; unfixed |
| F13 | `register_groupwise()` never calls `SetOutputDirectory()` — Elastix writes working/log files into the process CWD, not the `savedir` passed to it | VALID | 2026-07-07 | worked around only in register_hierarchical_gasonly.py via os.chdir |
| F14 | 500 iterations were used for ALL runs so far — production needs `['5000','4000','3000','2000']`. Current results are NOT production-grade. | VALID | 2026-07 | open caveat |

## Method / semantics

| ID | Claim | Status | Since | Note |
|----|-------|--------|-------|------|
| F15 | ANTs is PAIRWISE only (each frame → EI reference), NOT groupwise; `antsMultivariateTemplateConstruction2` is iterative pairwise, not simultaneous. Elastix + PCAMetric2 remains the only practical true-groupwise option. | VALID | 2026-06-03 | settled |
| F16 | Elastix `jac_det` is relative to the implicit GROUPWISE MEAN (mid-breath); ANTs `jac_det` is relative to the EI frame (bin 6). NOT directly comparable without normalization. | VALID | 2026-06-03 | law |
| F17 | MI (`SyN`) is theoretically preferable to CC (`SyNCC`) for xenon lung — CC assumes a local affine intensity relationship in each ~9³ window, violated at ventilation-defect boundaries and airway/parenchyma transitions | VALID | 2026-06-03 | belief, UNTESTED — MI run never done (see F18) |
| F18 | Only the CC (SyNCC) ANTs run was completed for 025JC; the MI (SyN) run was never done despite MI being preferred | VALID | 2026-06-03 | open gap |
| F19 | Jacobian determinant = local volume ratio, a property of the DEFORMATION FIELD, not anatomy; trachea/large-airway Jacobian artifacts come from rigid structure + BSpline compromise | VALID | 2026-06-03 | |
| F20 | Consecutive Jacobian/displacement (`jac_cons`, `disp_cons ≈ disp[t+1]−disp[t]`) is an APPROXIMATION valid only for small respiratory deformations; exact version requires composing transforms | VALID | 2026-06-03 | caveat |
| F21 | Composed Jacobian across stages via element-wise numpy multiply is likewise an approximation valid only for small deformations | VALID | 2026-06-03 | caveat |
| F22 | Direct 16-way groupwise registration is unstable — hence the hierarchical 4→7→14→16 | VALID | — | settled; rejected alternative |
| F23 | PCAMetric2 chosen over MSE (better convergence on low-SNR xenon); NormalizedMutualInformation rejected as too slow | VALID | — | settled |

## Data / format / orientation

| ID | Claim | Status | Since | Note |
|----|-------|--------|-------|------|
| F24 | New recon format (`workspace/data/recon/{23,49}.mat`) = single file, keys `gas_phase`/`dissolved_phase_real`/`dissolved_phase_imag`/`dissolved_phase_magnitude`, shape `(16,100,100,100)` T-FIRST, no CLAHE channel, no orientation header — INCOMPATIBLE with `read_files()`'s rspace/cspace/dspace triplet | VALID | 2026-07-07 | |
| F25 | `dspace` .mat key varies by dataset (some use `'image'`) — `read_files()` tries both | VALID | — | |
| F26 | Saved `.mat` outputs must be transposed `(3,2,1,0)` so on-disk arrays are `(X,Y,Z,T)` MATLAB order | VALID | 2026-07-07 | convention law |
| F27 | Axis identity for `gas.mat` `(X,Y,Z,T)`: X = sagittal (L-R), Y = coronal (A-P), Z = axial — derived empirically from `orientation_montage_23.py`, no header available | VALID | 2026-07-07 | |
| F28 | Which physical side (patient L vs R) maps to low-X vs high-X | UNKNOWN | 2026-07-07 | no metadata exists; matters only if L/R labels reported downstream |
| F29 | Mask-threshold-based slice localization (threshold 0.04) | RETRACTED | 2026-07-07 | not a clean lung segmentation — stays non-trivial across nearly the whole FOV, per-X/Y profiles too noisy. Replaced by fixed central-80%-of-FOV slice picking. |
| F30 | All 4 channels (gas, dp_rbc, dp_mem, clahe) must stay spatially aligned — `CopyInformation(image4d)` on dissolved channels after load. CLAHE is registration-guidance ONLY, never a science channel. | VALID | — | contract |
| F31 | `TransformationPosition.BEGINNING` = slot 0, `END` = slot 3 of the 4-image group | VALID | — | contract |

## Reuse / conventions

| ID | Claim | Status | Since | Note |
|----|-------|--------|-------|------|
| F32 | With `run_flags=(0,0,0), pad=False`, `orchestrate_registration_workflow()` collapses to exactly ONE `register_groupwise()` call per stage (the "final" stage always runs regardless of flags) | VALID | 2026-07-07 | key simplification |
| F33 | Pipeline helpers (`process_images`, `update_images*`, `combine_*`, `apply_transformations_in_sequence`, `extract_4d_subregion`, `average_4d_image_stack`, `register_groupwise`, `TransformationPosition`) are channel-agnostic generic-4D — safe to import and reuse rather than reimplement | VALID | 2026-07-07 | |
| F34 | Import the root pipeline via `importlib.util.spec_from_file_location`, NOT sys.path+import, to avoid polluting global import state. Importing does not execute `main()` (guarded), so its `/Volumes/HoomHamExt/` constants are never touched. | VALID | 2026-07-07 | pattern |
| F35 | Root pipeline files were NOT modified in the 2026-07 session — only read/imported | VALID | 2026-07-07 | |
| F36 | Repo convention: hardcoded `INPUT_MAT`/`SAVE_DIR` constants at file top, not CLI args — research pipeline, not a library; ablations run by commenting/uncommenting rows in `runs` in `main()` | VALID | — | accepted |
| F37 | Do NOT edit `workspace/archive/register_group_step_version_*.py` — versioned old scripts | VALID | — | law |
| F38 | Do NOT add an ANTs intermediate step before the final ANTs call — transform composition is handled explicitly | VALID | — | law |
| F39 | The 3D quiver-plot task queued in the root handoff (2026-06-03) is STILL NOT STARTED as of 2026-07-06 | VALID | 2026-07-06 | |
| F40 | Two harmless empty dirs (`01_phase1_first7/`, `02_phase2_last7/`) persist after `register_hierarchical_gasonly.py` runs — cosmetic only | VALID | 2026-07-07 | |
| F41 | `register_groupwise_PCA_step.py:455` has a typo'd key `parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']` (stray leading paren) — the key is silently ignored by Elastix, so B-spline grid spacing has been running at Elastix's default, never 6, in every run to date | VALID | 2026-07-12 | found + verified against live file during canon adoption |
