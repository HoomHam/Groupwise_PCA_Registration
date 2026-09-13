"""ourpca2 groupwise registration straight from Steve's Tyger recon.mat
(AIkill_Dynamic dumps) — no physics constants, no ledger. Registration only.

Per study: load d/recon.mat gas_phase (16,100,100,100), build the one-shot
union mask (>=4% of max in >=5/16 frames, largest CC — physreg_inspect
recipe), run register_groupwise with the validated ourpca2 settings
(_ourpca_run.py: L=None, elastix-matched schedule, lattice penalty 1e-3),
save stack / mask_union / u_ourpca2 / registered stack.

Copied 2026-09-12 from 2026_XeCS_Recon/workspace/helpers/dirtyrun_register.py
(workspace commit b8d78d9). ONLY change: input/output paths are CLI args
instead of the hardcoded AIkill_Dynamic/<study>/d/recon.mat + outputs/physreg.

Usage: python dirtyrun_register.py [--out OUTROOT] path/to/49.mat [more.mat ...]
  each .mat must hold key 'gas_phase' (16,N,N,N), T-first; results land in
  OUTROOT/<mat-stem>/. Default OUTROOT = mirror path on the external drive
  (outputs ~320 MB/study > 250 MB laptop rule).
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import scipy.io as sio
from scipy.ndimage import label, map_coordinates

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from registration_groupwise import register_groupwise

DEFAULT_OUT = pathlib.Path(
    '/Volumes/HoomHamExt/Work/Codes/2026_PCA_registration/ourpca2')


def load_gas(mat_path):
    m = sio.loadmat(mat_path, variable_names=['gas_phase'])
    return np.asarray(m['gas_phase'], dtype=np.float32)  # (16,100,100,100)


def union_mask(S):
    m3 = ((S >= 0.04 * S.max()).sum(axis=0) >= 5)
    lab, _ = label(m3)
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    return lab == sz.argmax()


def warp_cine(S, u):
    """WarpBins convention: warped_b(x) = S_b(x + u_b(x)), trilinear."""
    B, N = S.shape[0], S.shape[1]
    idx = np.indices((N, N, N), dtype=np.float64)
    out = np.empty_like(S, dtype=np.float64)
    for b in range(B):
        out[b] = map_coordinates(S[b].astype(np.float64), idx + u[b],
                                 order=1, mode='nearest')
    return out


ap = argparse.ArgumentParser()
ap.add_argument('mats', nargs='+', help='recon .mat files with gas_phase')
ap.add_argument('--out', type=pathlib.Path, default=DEFAULT_OUT)
ap.add_argument('--max-disp', type=float, default=12.0,
                help='box bound on control coefficients, finest-voxel units (XeCS canonical 12)')
ap.add_argument('--iters', default='150,120,100,80',
                help='L-BFGS-B maxiter per pyramid level (XeCS canonical 150,120,100,80)')
args = ap.parse_args()
iters = tuple(int(v) for v in args.iters.split(','))
assert len(iters) == 4, 'need 4 iteration counts for shrink (4,2,1,1)'
if not args.out.parent.exists():
    sys.exit(f'output root parent missing (drive unmounted?): {args.out.parent}')

for mat in args.mats:
    mat = pathlib.Path(mat)
    study = mat.stem
    if study == 'recon':                      # Tyger layout <study>/d/recon.mat
        study = mat.resolve().parents[1].name.split('_')[-1]
    out = args.out / study
    out.mkdir(parents=True, exist_ok=True)
    S = load_gas(mat)
    mask = union_mask(S)
    print(f'{study}: stack {S.shape}, mask {int(mask.sum())} vox '
          f'({100 * mask.mean():.1f}%)', flush=True)
    np.save(out / 'stack.npy', S)
    np.save(out / 'mask_union.npy', mask)

    t0 = time.time()
    u, info = register_groupwise(S.astype(np.float64), mask, L=None,
                                 mesh=(5, 7, 9, 17),
                                 shrink=(4, 2, 1, 1),
                                 iters=iters,
                                 max_disp=args.max_disp,
                                 lam_space=1e-3, lam_phase=0.0)
    print(f'{study} ourpca2: {time.time() - t0:.0f} s', flush=True)
    np.save(out / 'u_ourpca2.npy', u.astype(np.float32))
    np.save(out / 'registered_ourpca2.npy',
            warp_cine(S, u).astype(np.float32))
    print(f'{study}: saved u_ourpca2 + registered_ourpca2 -> {out}',
          flush=True)
