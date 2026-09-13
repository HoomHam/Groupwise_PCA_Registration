"""Native groupwise D_PCA registration metric — Route A, Phase D Step 3.

The production registrant for the piston 4DCS fusion pipeline (bench phantom:
polymer foam plug in a syringe barrel, mechanical piston stroke, HP Xe-129).
We own this functional so the physics-metric endgame of Method Draft §2.7.1
(swap D_PCA for the gas-dynamics residual) stays a small change; elastix is a
cross-check oracle only (`workspace/helpers/register_elastix_piston_oracle.py`).

Metric (Huizinga D_PCA, the rank-protected form — NOT D_PCA2):

    A  = Casorati matrix (B bins x V masked voxels) of the warped stack,
         each row (phase image over foam) mean-subtracted over voxels
    K  = B x B correlation matrix of the rows
    D_PCA = trace(K) - sum_{j<=L} lambda_j(K) = B - sum of top-L eigenvalues

L = 3 protects the first-order physics manifold (mean bell, temporal
derivative, baseline — §2.7.1). Measured on phantom v3 via the elastix oracle
(plan record 2026-07-05d): the index-weighted D_PCA2 taxes the physics modes
and hallucinates ~2 vox of transverse motion; this rank-protected form drops
RMS-vs-truth from 2.28 to 0.60 with the same transform. CAVEAT: L=3 matches
the phantom's generative rank by construction; on real data pick L from the
eigenvalue scree / §2.9 PCA-consistency check.

Gradients (all analytic, FD-verified in selftest_fusion #9/#10):

    d(sum top-L lambda_j)/dK = sum_j q_j q_j^T   (q_j = eigenvectors; valid
        while lambda_L > lambda_{L+1} — a degenerate crossing is non-smooth)
    chain through the correlation normalization and row mean-subtraction
    -> dcost/dA, scattered to dcost/d(warped stack), then through the
    trilinear warp's dependence on the sample point -> dcost/du.

The warp here mirrors warp._gather_plan / WarpBins EXACTLY (same clamping,
same base-corner rule, same u convention: warped_b(x) = stack_b(x + u_b(x)))
but keeps u and the weights in float64 — the metric gradient wrt u must not
be quantized by float32 (the selftest_rtr #7 lesson). WarpBins remains the
operator used inside recon chains; this module is the registration side.
Derivative wrt u is undefined exactly on trilinear cell boundaries and zero
where the sample point clamps at the volume edge (FD probes must stay
interior, as in selftest_rtr #7).
"""

import numpy as np

_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


# ------------------------------------------------------------------- metric

def dpca_cost_grad(A, L=3, eps=1e-12):
    """D_PCA cost and gradient on a Casorati matrix A (B, V) float64.

    Returns (cost, dcost/dA). L integer: cost = B - sum of the top-L
    eigenvalues of the row-correlation matrix (rank-protected Huizinga
    D_PCA); perfect alignment onto a rank-L manifold drives cost toward
    B - L... 0-ish residual variance outside the manifold.

    L=None: Huizinga D_PCA2, cost = sum_j j*lambda_j with eigenvalues in
    DESCENDING order — the index-weighted all-eigenvalue penalty, the native
    analog of elastix PCAMetric2 (the metric that recovered the real-data COM
    amplitude on MID00049 where rank-3 protection under-registered 0.36x).
    No protected subspace: it taxes every mode, trailing modes hardest.
    """
    A = np.asarray(A, dtype=np.float64)
    B, V = A.shape
    Ac = A - A.mean(axis=1, keepdims=True)
    C = (Ac @ Ac.T) / (V - 1)
    dC = np.clip(np.diag(C), eps, None)
    s = 1.0 / np.sqrt(dC)
    K = C * s[:, None] * s[None, :]
    lam, Q = np.linalg.eigh(K)                     # ascending
    if L is None:                                  # D_PCA2: weight B..1
        wgt = np.arange(B, 0, -1, dtype=np.float64)
        cost = float((wgt * lam).sum())
        W = -(Q * wgt) @ Q.T                       # -dcost/dK (sign below)
    else:
        cost = float(B - lam[-L:].sum())
        W = Q[:, -L:] @ Q[:, -L:].T                # d(sum top-L)/dK
    M = W * s[:, None] * s[None, :]
    F = M.copy()
    F[np.diag_indices(B)] -= (M @ C).diagonal() / dC
    G = (-2.0 / (V - 1)) * (F @ Ac)                # d(cost)/dAc = -d(tr WK)
    G -= G.mean(axis=1, keepdims=True)             # chain: row mean-subtract
    return cost, G


def dpca_stack(stack, mask, L=3):
    """D_PCA over the masked voxels of a (B, N, N, N) stack.

    Returns (cost, grad) with grad shaped like stack, zero off-mask.
    """
    stack = np.asarray(stack, dtype=np.float64)
    B = stack.shape[0]
    mflat = np.asarray(mask, bool).ravel()
    A = stack.reshape(B, -1)[:, mflat]
    cost, GA = dpca_cost_grad(A, L=L)
    G = np.zeros((B, mflat.size))
    G[:, mflat] = GA
    return cost, G.reshape(stack.shape)


# ------------------------------------------------ float64 warp + du chain

def _plan64(u_b, N):
    """float64 mirror of warp._gather_plan (same clamp + base-corner rule)."""
    idx = np.indices((N, N, N), dtype=np.float64)
    p = idx + np.asarray(u_b, dtype=np.float64)
    np.clip(p, 0.0, float(N - 1), out=p)
    p0 = np.minimum(np.floor(p), N - 2)
    f = (p - p0).reshape(3, -1)
    i0 = (p0[0].astype(np.int64) * N + p0[1].astype(np.int64)) * N \
        + p0[2].astype(np.int64)
    return i0.reshape(-1), f


def _warp_grad_bin(src_flat, i0, f, N):
    """Trilinear gather + derivative wrt the sample-point coordinates.

    Returns (out (Nvox,), d (3, Nvox)) where d[c] = d out / d p_c — i.e. the
    interpolated spatial gradient of src at the sample points. Since
    p = x + u_b(x), d out / d u_b,c = d[c].
    """
    fx, fy, fz = f
    nv = i0.size
    out = np.zeros(nv)
    d = np.zeros((3, nv))
    w = {0: (1.0 - fx, 1.0 - fy, 1.0 - fz), 1: (fx, fy, fz)}
    for dx, dy, dz in _CORNERS:
        s = src_flat[i0 + (dx * N + dy) * N + dz]
        wx, wy, wz = w[dx][0], w[dy][1], w[dz][2]
        sx, sy, sz = (1.0 if dx else -1.0), (1.0 if dy else -1.0), \
                     (1.0 if dz else -1.0)
        out += wx * wy * wz * s
        d[0] += sx * wy * wz * s
        d[1] += wx * sy * wz * s
        d[2] += wx * wy * sz * s
    return out, d


def warp_stack(stack, u):
    """Warp a real (B, N, N, N) stack by u (B, 3, N, N, N), float64.

    Same convention as warp.WarpBins: warped_b(x) = stack_b(x + u_b(x)).
    """
    stack = np.asarray(stack, dtype=np.float64)
    B, N = stack.shape[0], stack.shape[1]
    out = np.empty_like(stack)
    for b in range(B):
        i0, f = _plan64(u[b], N)
        out[b] = _warp_grad_bin(stack[b].ravel(), i0, f, N)[0].reshape(N, N, N)
    return out


def dpca_warped(stack, u, mask, L=3):
    """cost = D_PCA(warp(stack, u) over mask) and gradient wrt u.

    Returns (cost, gu) with gu shaped (B, 3, N, N, N) float64. The reference
    bin is the caller's business (Step 4 keeps u[ref] = 0 and applies the
    zero-mean stroke gauge at the control-point level).
    """
    stack = np.asarray(stack, dtype=np.float64)
    B, N = stack.shape[0], stack.shape[1]
    warped = np.empty((B, N * N * N))
    dwdp = np.empty((B, 3, N * N * N))
    for b in range(B):
        i0, f = _plan64(u[b], N)
        warped[b], dwdp[b] = _warp_grad_bin(stack[b].ravel(), i0, f, N)
    cost, G = dpca_stack(warped.reshape(B, N, N, N), mask, L=L)
    gu = G.reshape(B, 1, -1) * dwdp                 # chain rule, per component
    return cost, gu.reshape(B, 3, N, N, N)


# ------------------------------------------------ Step 4: B-spline solver

def _upmat(m, N):
    """Dense (N, m) 1-D linear-interpolation matrix, control grid spanning
    the full axis (g[0]=0, g[-1]=m-1). Trilinear upsampling is separable, so
    three of these give control -> dense; the transpose is the exact adjoint."""
    g = np.linspace(0.0, m - 1.0, N)
    i0 = np.minimum(np.floor(g).astype(np.int64), m - 2)
    f = g - i0
    U = np.zeros((N, m))
    U[np.arange(N), i0] = 1.0 - f
    U[np.arange(N), i0 + 1] = f
    return U


def _up(c, U):
    """(m, m, m) control lattice -> (N, N, N) dense, trilinear."""
    return np.einsum('ai,bj,ck,ijk->abc', U, U, U, c, optimize=True)


def _upT(g, U):
    """Exact adjoint of _up: (N, N, N) -> (m, m, m)."""
    return np.einsum('ai,bj,ck,abc->ijk', U, U, U, g, optimize=True)


def _downsample(cine, mask, s):
    """Block-mean pyramid level (shrink factor s must divide N)."""
    if s == 1:
        return np.asarray(cine, np.float64), np.asarray(mask, bool)
    B, N = cine.shape[0], cine.shape[1]
    Ns = N // s
    cs = np.asarray(cine, np.float64).reshape(
        B, Ns, s, Ns, s, Ns, s).mean(axis=(2, 4, 6))
    ms = np.asarray(mask, float).reshape(
        Ns, s, Ns, s, Ns, s).mean(axis=(1, 3, 5)) > 0.5
    return cs, ms


def register_groupwise(cine, mask, L=3, mesh=8, shrink=(4, 2, 1),
                       iters=(100, 60, 40), ref=0, max_disp=8.0,
                       lam_space=0.0, lam_phase=0.0, verbose=True):
    """Groupwise D_PCA registration of a (B, N, N, N) stroke cine (Route A).

    Transform: per-bin trilinear control lattice mesh^3 (control coefficients
    kept in FINEST-level voxel units, so they carry unchanged across the
    multires pyramid: at shrink s the dense field is up(c)/s in that level's
    voxels). Gauge: the across-bin control-point mean is projected out inside
    the cost (zero-mean stroke gauge — the groupwise analog of elastix's
    UseZeroAverageDisplacementConstraint), so the metric's translation-in-
    common null space never wanders. Optimizer: full-batch L-BFGS-B on the
    analytic gradient (dpca_warped chained through the upsampling adjoint).

    mesh: int (fixed lattice) or per-level tuple matching shrink — the
    GRID-SPACING SCHEDULE, elastix's only smoothness lever for the stack
    transform (BSplineStackTransform cannot take a bending penalty,
    itkStackTransform.h:385; elastix runs control pitch 24->18->12->6 vox).
    A fixed fine mesh at a coarse pyramid level is nearly per-voxel-free
    (mesh 16 on a 25^3 level = 1.7 vox pitch) and seeds high-frequency
    coefficient noise the fine levels inherit (measured 2026-07-06:
    max|u| tail 18-22 vox on real piston data). The lattice is resampled
    exactly between levels (trilinear-in-trilinear via _upmat(m_old, m_new)).

    lam_space/lam_phase: optional control-lattice second-difference penalty
    (reuses registration_physics._pen_grad). Default 0 = elastix-faithful;
    the mesh schedule is the first-line smoother.

    Returns (u, info): u (B, 3, N, N, N) float32, ref-anchored (u[ref] = 0),
    ready for warp.WarpBins; info = per-level cost trajectories.
    """
    from scipy.optimize import minimize

    cine = np.asarray(cine, np.float64)
    if np.iscomplexobj(cine):
        cine = np.abs(cine)
    B, N = cine.shape[0], cine.shape[1]
    meshes = [int(v) for v in np.broadcast_to(np.asarray(mesh, int),
                                              (len(shrink),))]
    m = meshes[0]
    c = np.zeros((B, 3, m, m, m))
    info = {'levels': []}

    for s, it, m_new in zip(shrink, iters, meshes):
        if m_new != m:                       # lattice refinement (exact)
            U12 = _upmat(m, m_new)
            c_new = np.empty((B, 3, m_new, m_new, m_new))
            for b in range(B):
                for k in range(3):
                    c_new[b, k] = _up(c[b, k], U12)
            c, m = c_new, m_new
        S_s, m_s = _downsample(cine, mask, s)
        Ns = S_s.shape[1]
        U = _upmat(m, Ns)
        hist = []

        def cost_grad(cvec):
            cc = cvec.reshape(B, 3, m, m, m)
            cP = cc - cc.mean(axis=0, keepdims=True)      # zero-mean gauge
            u = np.empty((B, 3, Ns, Ns, Ns))
            for b in range(B):
                for k in range(3):
                    u[b, k] = _up(cP[b, k], U) / s
            cost, gu = dpca_warped(S_s, u, m_s, L=L)
            gc = np.empty_like(cc)
            for b in range(B):
                for k in range(3):
                    gc[b, k] = _upT(gu[b, k], U) / s
            if lam_space > 0 or lam_phase > 0:
                from registration_penalty import _pen_grad  # local extract (was registration_physics)
                pen, gpen = _pen_grad(cP, lam_space, lam_phase)
                cost += pen
                gc += gpen
            gc -= gc.mean(axis=0, keepdims=True)          # gauge projection
            hist.append(cost)
            return cost, gc.ravel()

        # box bounds: on noisy CS cines an unbounded solve can run away
        # (measured: |u| up to 220 vox on a 64-cube). max_disp caps each
        # control coefficient in finest-voxel units.
        res = minimize(cost_grad, c.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-max_disp, max_disp)] * c.size,
                       options={'maxiter': int(it), 'maxcor': 20})
        c = res.x.reshape(B, 3, m, m, m)
        info['levels'].append({'shrink': s, 'n_cost': len(hist),
                               'cost0': hist[0], 'cost1': float(res.fun)})
        if verbose:
            print(f'  level /{s}: {Ns}^3, {it} iters  '
                  f'D_PCA {hist[0]:.5f} -> {res.fun:.5f}')

    c -= c.mean(axis=0, keepdims=True)                    # bake in the gauge
    Uf = _upmat(m, N)
    u = np.empty((B, 3, N, N, N), dtype=np.float32)
    for b in range(B):
        for k in range(3):
            u[b, k] = _up(c[b, k], Uf)
    u = u - u[ref:ref + 1]                                # ref-anchor
    u[ref] = 0.0
    return u, info
