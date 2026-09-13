# ourpca2 — native one-shot groupwise PCA registration (copied from XeCS)

Copied 2026-09-12 from `~/Hooman/Work/Codes/2026_XeCS_Recon` on Hooman's
order, after that repo's live session confirmed this is its working version.
It is **numpy + scipy only** — no Elastix. Elastix/PCAMetric2 was XeCS's
cross-check oracle; this functional (Huizinga D_PCA2 = Σ_j j·λ_j) is the
native analog and was the arm Hooman called validated on 2026-07-07. Used on
31 sessions (RT/EBV/LTX/LTX2 dirty runs, XeCS F206: passed 31/31 —
temporal-CV ×0.55–0.90, det J > 0, |u| max ≤ 18 vox).

## Files

| Here | Origin (XeCS) | Commit | Changes made here |
|---|---|---|---|
| `registration_groupwise.py` | `recon/registration_groupwise.py` | root `997ed74` 2026-07-07 | one import line: `registration_physics` → `registration_penalty` |
| `registration_penalty.py` | `_pen_grad` in `recon/registration_physics.py` | root `997ed74` | extracted verbatim into own module |
| `dirtyrun_register.py` | `workspace/helpers/dirtyrun_register.py` | ws `b8d78d9` 2026-09-10 | hardcoded `AIkill_Dynamic/<study>/d/recon.mat` + `outputs/physreg` → CLI `mats` + `--out` |
| `dirtyrun_regcheck.py` | `workspace/helpers/dirtyrun_regcheck.py` | ws `b8d78d9` | `OUT` → `--out` CLI arg; default tag `pca` |
| `XeCS_DirtyRun_Pipeline.md` | `workspace/reference/DirtyRun_Pipeline.md` | ws `b8d78d9` | verbatim, for context (row 1 = this step; rows 2–5 are XeCS physics steps, NOT copied) |

Not copied (per XeCS): `register_elastix_piston_oracle.py` (phantom-only
Elastix oracle), `_ourpca_run.py` (scratch arm script), rest of
`registration_physics.py` (physics-metric arm), `recon/warp.py` WarpBins
(only needed inside a recon chain), gradient selftests
`recon/selftest_fusion.py` #9/#10 + `selftest_rtr.py` #7 (copy if the metric
is ever touched).

## Validated settings ("ourpca2")

```python
register_groupwise(S.astype(np.float64), mask, L=None,
                   mesh=(5, 7, 9, 17), shrink=(4, 2, 1, 1),
                   iters=(150, 120, 100, 80), max_disp=12.0,
                   lam_space=1e-3, lam_phase=0.0)
```

- `L=None` → D_PCA2 (index-weighted, all eigenvalues). Rank-protected `L=3`
  under-registered real lungs 0.36× (protects motion variance too); right only
  on the phantom whose generative rank is 3.
- Smoothness comes from the **mesh schedule** (control pitch coarse→fine), not
  a bending penalty. Fine mesh at coarse pyramid level seeds high-frequency
  coefficient noise (measured |u| tails 18–22 vox). `lam_space=1e-3` = light
  lattice second-difference penalty; Elastix-faithful default is 0.
- Gauge: across-bin control-point mean projected out inside the cost
  (= Elastix `UseZeroAverageDisplacementConstraint`). Do not remove.
- `u`, weights kept float64 inside the metric gradient; float32 quantizes the
  gradient and FD checks fail. Output cast to float32 at the end only.
- Gradient undefined exactly on trilinear cell boundaries, zero where sample
  clamps at the volume edge.
- `max_disp` clamps each control coefficient (12 vox at N=100, 3.5 mm vox).
- Mask matters: metric on masked voxels only; tight mask under-registers
  base/apex. Union mask recipe: ≥ 4 % of stack max in ≥ 5/16 bins, largest CC.

## I/O contract

- Input cine `(B, N, N, N)` float, **B=16 bins first**; `gas_phase`
  `(16,100,100,100)` straight from `scipy.io.loadmat(...)['gas_phase']`
  (magnitude taken inside if complex). Our `workspace/data/recon/{23,49}.mat`
  already have this key/shape.
- Mask `(N,N,N)` bool.
- Output `u` `(B, 3, N, N, N)` float32, ref-anchored (`u[0] = 0`);
  convention `warped_b(x) = stack_b(x + u_b(x))`, trilinear, nearest clamp.
  Registered stack = `warp_cine(S, u)`.
- Per study on disk: `stack.npy`, `mask_union.npy`, `u_ourpca2.npy`,
  `registered_ourpca2.npy` (~320 MB total → external drive by the 250 MB rule;
  default out = `/Volumes/HoomHamExt/Work/Codes/2026_PCA_registration/ourpca2/`).
- Runtime ~5–6 min per 16×100³ study on the laptop.

## Run

```bash
cd ~/Hooman/Work/Codes/2026_PCA_registration/ourpca2
python dirtyrun_register.py ../workspace/data/recon/49.mat
python dirtyrun_regcheck.py --tag pca 49
```

## How it differs from this repo's Elastix pipeline

| | `register_groupwise_PCA_step.py` (ours) | `ourpca2` |
|---|---|---|
| Engine | Elastix `PCAMetric2`, `BSplineStackTransform`, ASGD | numpy D_PCA2, trilinear control lattice, L-BFGS-B, analytic gradient |
| Strategy | hierarchical 4→7→14→16, 4 channels | one shot, all 16 bins, gas only |
| Smoothness | B-spline order 3 + `GridSpacingSchedule` | mesh schedule (5,7,9,17) + lattice penalty |
| Transform out | Elastix TransformParameters | dense `u` (16,3,N,N,N) |
| Dependency | SimpleITK-SimpleElastix wheel | numpy, scipy |
