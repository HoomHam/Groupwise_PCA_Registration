"""Registration validation for dirty-run studies (ourpca2 groupwise).

Per study, from outputs/physreg/<study>/{stack,mask_union,u_ourpca2,
registered_ourpca2}.npy:
  * temporal std across the 16 bins inside the mask, unregistered vs
    registered — ratio < 1 = motion removed (the cine should still keep its
    intensity breathing, so the ratio is NOT expected to reach 0)
  * displacement field: median / p95 / max |u|, fraction > 8 vox, and the
    Jacobian determinant of x + u_b(x) per bin (min, fraction < 0 = folding)
  * mask-edge motion: per-bin support (>=4 % max) centroid z-excursion,
    before vs after
  * figure: 3 coronal slices × [EE unreg | EI unreg | EI reg | tstd unreg |
    tstd reg | |u| EI], EE/EI = argmin/argmax of the masked signal sum.

Copied 2026-09-12 from 2026_XeCS_Recon/workspace/helpers/dirtyrun_regcheck.py
(workspace commit b8d78d9). ONLY change: output root is a CLI arg (--out),
matching dirtyrun_register.py here; <study> = the .mat stem.

Usage: python dirtyrun_regcheck.py [--out OUTROOT] --tag pca <study> [...]
Writes OUTROOT/<tag>_regcheck_<study>.png + <tag>_regcheck.json
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass

DEFAULT_OUT = pathlib.Path(
    '/Volumes/HoomHamExt/Work/Codes/2026_PCA_registration/ourpca2')
OUT = DEFAULT_OUT  # reassigned from --out in __main__


def jacobian_det(u):
    """u: (3, N, N, N) displacement in voxels; det(I + grad u)."""
    g = [[np.gradient(u[i], axis=j) for j in range(3)] for i in range(3)]
    J = np.empty(u.shape[1:] + (3, 3))
    for i in range(3):
        for j in range(3):
            J[..., i, j] = g[i][j] + (1.0 if i == j else 0.0)
    return np.linalg.det(J)


def disp(vol, y):
    return np.fliplr(np.rot90(vol[:, y, :].T, k=-1))


def check(study, tag):
    d = OUT / study
    S = np.load(d / 'stack.npy').astype(np.float32)
    R = np.load(d / 'registered_ourpca2.npy').astype(np.float32)
    mask = np.load(d / 'mask_union.npy')
    u = np.load(d / 'u_ourpca2.npy').astype(np.float32)  # (16,3,N,N,N)
    Msum = S[:, mask].sum(axis=1)
    ee, ei = int(Msum.argmin()), int(Msum.argmax())

    # normalise each bin to its masked sum so intensity breathing does not
    # masquerade as motion in the temporal std
    Sn = S / S[:, mask].sum(axis=1)[:, None, None, None]
    Rn = R / R[:, mask].sum(axis=1)[:, None, None, None]
    ts0, ts1 = Sn.std(axis=0), Rn.std(axis=0)
    mean0 = Sn.mean(axis=0)
    cv0 = float(ts0[mask].mean() / mean0[mask].mean())
    cv1 = float(ts1[mask].mean() / mean0[mask].mean())

    mag = np.sqrt((u ** 2).sum(axis=1))          # (16,N,N,N)
    magm = mag[:, mask]
    detJ = np.stack([jacobian_det(u[b]) for b in range(u.shape[0])])
    detm = detJ[:, mask]

    thr = 0.04 * S.max()
    cz0 = [center_of_mass(S[b] >= thr)[2] for b in range(S.shape[0])]
    cz1 = [center_of_mass(R[b] >= thr)[2] for b in range(R.shape[0])]

    res = dict(ee_bin=ee, ei_bin=ei, mask_vox=int(mask.sum()),
               tstd_cv_unreg=cv0, tstd_cv_reg=cv1, tstd_ratio=cv1 / cv0,
               u_median=float(np.median(magm)), u_p95=float(
                   np.percentile(magm, 95)), u_max=float(magm.max()),
               u_frac_gt8=float((magm > 8).mean()),
               detJ_min=float(detm.min()), detJ_frac_neg=float(
                   (detm <= 0).mean()), detJ_p1=float(np.percentile(detm, 1)),
               detJ_p99=float(np.percentile(detm, 99)),
               support_cz_excursion_unreg=float(max(cz0) - min(cz0)),
               support_cz_excursion_reg=float(max(cz1) - min(cz1)))
    print(f'{study}: tstd CV {cv0:.3f} -> {cv1:.3f} (x{cv1 / cv0:.2f}); |u| '
          f'med {res["u_median"]:.2f} p95 {res["u_p95"]:.2f} max '
          f'{res["u_max"]:.1f} vox, >8: {100 * res["u_frac_gt8"]:.2f}%; detJ '
          f'min {res["detJ_min"]:.2f} neg {100 * res["detJ_frac_neg"]:.3f}%; '
          f'support z-excursion {res["support_cz_excursion_unreg"]:.2f} -> '
          f'{res["support_cz_excursion_reg"]:.2f} vox', flush=True)

    ys = np.where(mask.any(axis=(0, 2)))[0]
    ysel = [int(np.percentile(ys, q)) for q in (25, 50, 75)]
    vmax = float(np.percentile(S[ei][mask], 99))
    tmax = float(np.percentile(ts0[mask], 99))
    rows = np.where(mask.any(axis=(1, 2)))[0]
    cols = np.where(mask.any(axis=(0, 1)))[0]
    fig, axes = plt.subplots(3, 6, figsize=(18, 8.5),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12,
                                              left=0.02, right=0.98,
                                              top=0.9, bottom=0.02))
    titles = [f'EE (bin {ee}) unreg', f'EI (bin {ei}) unreg',
              f'EI (bin {ei}) REG', 'temporal std unreg', 'temporal std REG',
              f'|u| bin {ei} (vox)']
    for r, y in enumerate(ysel):
        panels = [disp(S[ee], y), disp(S[ei], y), disp(R[ei], y),
                  disp(ts0 * mask, y), disp(ts1 * mask, y),
                  disp(mag[ei] * mask, y)]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
                cont = disp(mask.astype(float), y)
                ax.contour(cont, levels=[0.5], colors='r', linewidths=0.4)
            elif c < 5:
                ax.imshow(img, cmap='magma', vmin=0, vmax=tmax)
            else:
                im = ax.imshow(img, cmap='viridis', vmin=0, vmax=12)
            ax.axis('off')
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
            if c == 0:
                ax.text(0.02, 0.95, f'y={y}', color='w', fontsize=8,
                        transform=ax.transAxes, va='top')
    fig.colorbar(im, ax=list(axes[:, 5]), shrink=0.6, pad=0.01)
    fig.suptitle(f'{study} registration check — tstd CV {cv0:.3f}→{cv1:.3f} '
                 f'(×{cv1 / cv0:.2f}), |u| med {res["u_median"]:.1f} / p95 '
                 f'{res["u_p95"]:.1f} / max {res["u_max"]:.0f} vox, detJ min '
                 f'{res["detJ_min"]:.2f} ({100 * res["detJ_frac_neg"]:.2f}% '
                 f'folded), support z-excursion '
                 f'{res["support_cz_excursion_unreg"]:.1f}→'
                 f'{res["support_cz_excursion_reg"]:.1f} vox', fontsize=11)
    p = OUT / f'{tag}_regcheck_{study}.png'
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print(f'wrote {p}', flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('studies', nargs='+')
    ap.add_argument('--tag', default='pca')
    ap.add_argument('--out', type=pathlib.Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    OUT = a.out
    jp = OUT / f'{a.tag}_regcheck.json'
    allres = json.loads(jp.read_text()) if jp.exists() else {}
    for s in a.studies:
        allres[s] = check(s, a.tag)
        jp.write_text(json.dumps(allres, indent=1))
