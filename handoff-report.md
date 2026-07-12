# Handoff — 2026-06-03
## Project: 2026_PCA_registration — Xenon-129 Lung MRI Registration Pipeline

---

## Update — 2026-07-06 session (see `workspace/handoff-report.md` for full detail)

Quiver-plot task below is still **not started** — this session went a different
direction: adapted the oneshot/hierarchical registration scripts to a newer
single-file recon `.mat` format (`workspace/data/recon/{23,49}.mat`, keys
`gas_phase`/`dissolved_phase_real`/`dissolved_phase_imag`, T-first axis, no
CLAHE channel), wrote a bare-bones gas-only oneshot script, wrote a bare-bones
gas-only version of the full 4→7→14→16 hierarchical pipeline, and built a
reusable sagittal/coronal grid-video renderer. All new scripts and outputs
live under `workspace/` (gitignored) — none of the root-level pipeline files
(`register_groupwise_PCA_step.py`, `register_elastix_oneshot.py`) were
modified, only imported/read.

Full session narrative, decisions, file index, and next-step suggestions:
`workspace/handoff-report.md`.

---

## What Was Done This Session

### 1. Verified parameter parity between stepwise and oneshot Elastix scripts
- `register_elastix_oneshot.py` parameters match `register_groupwise_PCA_step.py` exactly
- Only intentional diff: `WriteResultImage` (`true` in stepwise, `false` in oneshot)
- CLAUDE.md corrected: `CheckNumberOfSamples` is `['true']` in both scripts (doc had claimed `['false']`)

---

### 2. Rewrote `register_elastix_oneshot.py` — gas-only single-pass + Jacobians

**Location**: `workspace/helpers/tests/register_elastix_oneshot.py`

**Changes from original:**
- `run_flags = (0, 0, 0)` — CLAHE removed, single registration pass (stage 3 only)
- `save_dir` → local: `/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/elastix_oneshot_gas/`
- Added `_jac_from_defField(df_4d)` helper — computes Jacobian per frame via deformation field (`ComputeDeformationFieldOn` → `sitk.DisplacementFieldJacobianDeterminant` per 3D frame)
- Jacobian threading: `run_elastix_groupwise(..., prev_jac_arr)` accumulates composed Jacobian as numpy array across stages (element-wise multiply approximation — valid for small respiratory deformations)
- `orchestrate_elastix_staged` converts final numpy jac → 4D SimpleITK via `JoinSeries` for saving
- Added `_save_mat` helper for .mat-only saves
- Saves registration mask (3D, `.nii` + `.mat`)

**Three Jacobian outputs (all (T,Z,Y,X) or (T-1,Z,Y,X) numpy in MATLAB X,Y,Z,T order):**

| File | Shape | Meaning |
|------|-------|---------|
| `jac_det.mat` / `.nii` | (16,100,100,100) | det(J) of deformation from implicit groupwise mean to each bin |
| `jac_cons.mat` | (15,100,100,100) | bin t → bin t+1: `jac_arr[t+1] / jac_arr[t]` |
| `jac_from_EE.mat` | (16,100,100,100) | relative to end-expiration bin 0: `jac_arr / jac_arr[0:1]` |

**Key implementation note — SimpleElastix 4D limitations:**
- `ComputeJacobianDeterminantOn()` → wrong name in this build → use `ComputeDeterminantOfSpatialJacobianOn()` but it doesn't write a file
- `sitk.Multiply` → unsupported for 4D in this build
- `CopyInformation` / `SetDirection` on 4D images → fails
- **Solution**: keep Jacobian entirely as numpy (T,Z,Y,X), convert to SimpleITK only at save via `JoinSeries` of per-frame 3D images

**Run completed**: 3.2 min, output verified at `elastix_oneshot_gas/`

---

### 3. Rewrote `register_ants_oneshot.py` — gas-only single-pass + Jacobians

**Location**: `workspace/helpers/tests/register_ants_oneshot.py`

**Key design:**
- ANTs is **pairwise** (each frame → EI), NOT groupwise. EI = fixed reference (detected by max signal sum)
- Single-stage `ants.registration(..., type_of_transform=...)` — no CLAHE, no multi-stage
- Jacobian from ANTs warp field: `ants.create_jacobian_determinant_image(domain=fixed_ants, tx=fwd[0], geom=True)` computed BEFORE `cleanup_ants_files`
- EI frame: Jacobian = `np.ones(shape)` (identity)

**Metric chooser** — script prompts at launch:
```
[1] CC  (SyNCC) — local cross-correlation, sharper, assumes local linearity
[2] MI  (SyN)   — Mattes mutual information, no linearity assumption, better at defect boundaries
```
Output goes to `ants_oneshot_gas_CC/` or `ants_oneshot_gas_MI/` accordingly.

**Three Jacobian outputs** (same naming/semantics as Elastix):

| File | Shape | Reference |
|------|-------|-----------|
| `jac_det.mat` | (16,Z,Y,X) | vs EI frame (bin 6); EI frame = 1s |
| `jac_cons.mat` | (15,Z,Y,X) | consecutive: `jac[t+1]/jac[t]` |
| `jac_from_EE.mat` | (16,Z,Y,X) | relative to EE bin 0 |

**Important distinction from Elastix:**
- Elastix `jac_det` = deformation vs **implicit groupwise mean** (mid-breath)
- ANTs `jac_det` = deformation vs **EI frame** (end-inhale, bin 6)
- These are NOT directly comparable without normalization

**Why MI over CC for xenon lung:**
- CC assumes local affine intensity relationship within each ~9³ window
- Violated at ventilation defect boundaries and airway/parenchyma transitions
- MI makes no linearity assumption — only requires statistical dependency
- ANTs `SyN` (MI) is theoretically correct for heterogeneous xenon ventilation

**ANTs groupwise status (researched this session):** ANTs still pairwise-only as of 2026. `antsMultivariateTemplateConstruction2` is iterative pairwise, not simultaneous groupwise. Elastix + PCAMetric2 remains the only practical true groupwise option.

**Run completed (CC run):** 78.6 min for 15 frames × SyNCC

---

### 4. Completed runs for subject `2024-11-13_025JC`

| Run folder | Method | Duration |
|------------|--------|----------|
| `reg/elastix_oneshot_jac/` | Elastix CLAHE+final, 500 iter | 7 min |
| `reg/elastix_oneshot_gas/` | Elastix gas-only, 500 iter | 3.2 min |
| `reg/ants_oneshot_gas/` → `ants_oneshot_gas_CC/` | ANTs SyNCC | 78.6 min |

---

### 5. Conceptual clarifications documented

- Jacobian det = local volume ratio, deformation field property NOT anatomy
- Trachea/large airway Jacobian artifacts explained (rigid structure + BSpline compromise)
- Jacobian reference frames: groupwise mean (Elastix) vs EI (ANTs)
- `jac_from_EE` maps to regional Specific Ventilation (standard clinical reference)

---

## Next Task: 3D Quiver Plot from Deformation Fields

The next agent should implement a **3D quiver plot** of the deformation/displacement vector field, in the same three variants as the Jacobian:

1. **All-to-target** (`disp_det`): displacement of each bin relative to the registration reference (groupwise mean for Elastix, EI for ANTs)
2. **Consecutive** (`disp_cons`): displacement from bin t to bin t+1
3. **All-to-EE** (`disp_from_EE`): displacement relative to end-expiration bin 0

### How to get the displacement fields

**For Elastix oneshot** (`register_elastix_oneshot.py`):
- Already computes deformation field per stage in `run_elastix_groupwise()` via `tf_df.ComputeDeformationFieldOn()` → `tf_df.GetDeformationField()`
- The deformation field is (T, Z, Y, X, 4) numpy — first 3 components are spatial (x, y, z displacements in mm), 4th is time (always ~0)
- Currently only used to compute Jacobian then discarded — **save the displacement field too**

**For ANTs oneshot** (`register_ants_oneshot.py`):
- `reg['fwdtransforms'][0]` is the `.nii.gz` warp file (displacement field)
- Can read it back: `ants.image_read(fwd[0])` gives the 3D vector displacement field in mm
- Or use `ants.deformation_gradient(...)` 

### Quiver plot design

The quiver plot should subsample the 3D field (e.g., every 5-10 voxels in each direction) and display arrows showing displacement direction and magnitude. Suggested approach:

```python
# Subsample grid
step = 8  # voxels
xs, ys, zs = np.mgrid[0:X:step, 0:Y:step, 0:Z:step]
us = disp_x[::step, ::step, ::step]  # x-component
vs = disp_y[::step, ::step, ::step]  # y-component
ws = disp_z[::step, ::step, ::step]  # z-component

# 3D quiver via matplotlib or plotly
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(xs, ys, zs, us, vs, ws, length=scale, normalize=False)
```

Color-code by displacement magnitude `sqrt(u²+v²+w²)`.

### Suggested output structure (parallel to Jacobian)

| File | Contents |
|------|----------|
| `disp_det.mat` | (T, Z, Y, X, 3) float32 — displacement vs reference, mm |
| `disp_cons.mat` | (T-1, Z, Y, X, 3) — consecutive displacement |
| `disp_from_EE.mat` | (T, Z, Y, X, 3) — displacement vs EE bin 0 |
| `disp_quiver_*.png` | Rendered quiver plots (per bin or as animation) |

Note: Consecutive displacement `disp_cons[t]` ≈ `disp[t+1] - disp[t]` is an approximation (valid for small deformations). Exact version requires composing transforms.

---

## Key File Locations

| File | Purpose |
|------|---------|
| `workspace/helpers/tests/register_elastix_oneshot.py` | Elastix groupwise oneshot (gas-only, with Jacobians) |
| `workspace/helpers/tests/register_ants_oneshot.py` | ANTs pairwise oneshot (gas-only, metric choice, with Jacobians) |
| `register_groupwise_PCA_step.py` | Main stepwise hierarchical pipeline (unchanged) |
| `CLAUDE.md` | Project context, code map, pitfalls |
| `/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/` | All run outputs for subject 025JC |

---

## Suggested Skills for Next Session

- `/handoff` — update this doc when quiver work is done
- Use `cavecrew-builder` for targeted edits to the oneshot scripts
- Use `cavecrew-investigator` to locate deformation field handling code before modifying

---

## Open Questions / Notes

- 500 iterations used for all runs so far — production runs should use `[5000,4000,3000,2000]`
- Elastix `CheckNumberOfSamples = ['true']` — flip to `['false']` if crash on small lung volumes
- ANTs MI (SyN) run NOT yet completed for 025JC — still only CC run done. Should run MI for comparison.
- Disk space on internal SSD: ~28 GB free — tight for full ablation but sufficient for individual oneshot runs
