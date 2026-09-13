"""Control-lattice smoothness penalty for register_groupwise (ourpca2).

VERBATIM extract of `_pen_grad` from
  2026_XeCS_Recon/recon/registration_physics.py  (commit 997ed74, 2026-07-07)
copied 2026-09-12 so that registration_groupwise.py is self-contained here
(the rest of registration_physics.py is the physics-metric arm — not needed).
"""
import numpy as np


def _pen_grad(cP, lam_space, lam_phase):
    """Second-difference smoothness penalty on the control lattice.

    Spatial: free-boundary second differences along the three control axes
    (a bending-energy proxy — unlike elastix's StackTransform, our own
    lattice CAN take an explicit penalty; grid pitch is no longer the only
    smoothness lever). Phase: CYCLIC second difference across bins (the
    stroke is periodic in phase even though each voxel's signal is not).
    Returns (penalty, grad) with grad shaped like cP. Needed because the
    metric alone noise-chases: with per-voxel free (a, g0) plus unpenalized
    u, an L-BFGS solve on noisy data finds costs BELOW the true-alignment
    cost by bending u into the noise (measured on the selftest phantom:
    0.033 vs 0.077 at truth, 1% noise).
    """
    pen, grad = 0.0, np.zeros_like(cP)
    if lam_space > 0:
        for ax in range(2, 5):
            dd = np.diff(cP, 2, axis=ax)
            pen += 0.5 * lam_space * float((dd * dd).sum())
            g = np.zeros_like(cP)
            sl = [slice(None)] * 5
            sl[ax] = slice(None, -2); g[tuple(sl)] += dd
            sl[ax] = slice(1, -1);    g[tuple(sl)] -= 2 * dd
            sl[ax] = slice(2, None);  g[tuple(sl)] += dd
            grad += lam_space * g
    if lam_phase > 0:
        dd = np.roll(cP, -1, axis=0) - 2 * cP + np.roll(cP, 1, axis=0)
        pen += 0.5 * lam_phase * float((dd * dd).sum())
        grad += lam_phase * (np.roll(dd, -1, axis=0) - 2 * dd
                             + np.roll(dd, 1, axis=0))
    return pen, grad
