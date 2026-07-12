# PCA Registration — Facts (belief table)

Flip Status, never delete rows. VALID / RETRACTED / SUSPECT.
Retro-filled 2026-07-12 from CLAUDE.md + handoff-report.md (root + workspace) + Logic.md.

| ID | Claim | Status | Since | Note |
|----|-------|--------|-------|------|
| F1 | `cyclic` is a DEAD parameter in `orchestrate_registration_workflow()` — callers pass `cyclic=True` for 14-frame/midpoint/final-16 phases, but all four internal `register_groupwise()` calls hardcode `cyclic=False` (:698, :714, :729, :738). Hierarchical phases never use cyclic transforms despite intent. | VALID | 2026-07-06 | workspace handoff; unfixed in root file |
| F2 | `register_groupwise()` never calls `SetOutputDirectory()` — Elastix writes logs/working files into the process CWD, not the `savedir` it is passed | VALID | 2026-07-06 | workspace handoff |
| F3 | `register_groupwise_PCA_step.py:455` has a typo'd key `parameterMap['(FinalGridSpacingInPhysicalUnits']` (stray leading paren) — that grid-spacing setting is SILENTLY IGNORED by Elastix | VALID | 2026-07-12 | found during canon adoption |
| F4 | Elastix and ANTs use DIFFERENT reference frames: Elastix groupwise = groupwise mean; ANTs pairwise = EI bin 6. Jacobians/displacements are not directly comparable across the two. | VALID | 2026-07-06 | root handoff |
| F5 | Use MI (not CC) for xenon registration — intensity relationship across dynamic gas frames is not linear | VALID | 2026-07-06 | root handoff |
| F6 | PCAMetric2 crashes unless `CheckNumberOfSamples=['true']` is set in the param map | VALID | — | CLAUDE.md pitfalls |
| F7 | SimpleElastix wheel is REQUIRED (`SimpleITK-SimpleElastix>=3.0.0a1`) — stock SimpleITK has no groupwise/PCAMetric2 | VALID | — | CLAUDE.md, requirements.txt |
| F8 | The external drive (`/Volumes/HoomHamExt/…`) must be mounted before any run — `save_dir`/`input_files` are hardcoded at `register_groupwise_PCA_step.py:16-22` | VALID | — | CLAUDE.md pre-flight |
| F9 | The `dspace` key varies between datasets — never assume it | VALID | — | CLAUDE.md pitfalls |
| F10 | Recon output moved to a NEW single-file format: `gas_phase` / `dissolved_phase_real` / `dissolved_phase_imag`, T-first axis order | VALID | 2026-07-06 | workspace handoff |
| F11 | Root-level `register_elastix_oneshot.py` (referenced in CLAUDE.md code map) NO LONGER EXISTS at root — it lives at `workspace/helpers/tests/register_elastix_oneshot.py`; only a stale `__pycache__` entry remains at root. CLAUDE.md code map is stale on this line. | VALID | 2026-07-12 | found during canon adoption |
| F12 | `register_EI_ants()` (:1914) uses `type_of_transform='SyN'` — `'SyNCC'` is present but COMMENTED OUT | VALID | — | code |
| F13 | `main()` (:1989) ablation list is mostly commented — only `((0,0,0), False, 'npad_image')` is live | VALID | — | code |
| F14 | Elastix leaves temp dirs behind — clean up between runs | VALID | — | CLAUDE.md pitfalls |
| F15 | `archive/` holds 20+ versioned older pipelines (`register_group_step_version_*.py`) — reference only, DO NOT EDIT | VALID | — | CLAUDE.md |
