"""
Run pairwise ANTs SyNCC on a pre-registered groupwise gas image.
Uses gas as its own registration guide (no CLAHE available for old runs).
Saves gas.mat to <run_dir>/ants/gas.mat.
"""
import os
import sys
import time
import numpy as np
import scipy.io
import ants

# ── Config ───────────────────────────────────────────────────────────────────
RUNS_TO_PROCESS = [
    '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/5000',
]
# ─────────────────────────────────────────────────────────────────────────────


def load_gas(run_dir):
    m = scipy.io.loadmat(os.path.join(run_dir, 'gw', 'gas.mat'), variable_names=['gas'])
    g = m['gas'].astype(np.float32)          # (X, Y, Z, T) MATLAB order
    g = g / g.max()
    return g


def find_ei(gas_xyzT):
    sums = gas_xyzT.reshape(-1, gas_xyzT.shape[3]).sum(axis=0)
    return int(np.argmax(sums))


def cleanup(path_list):
    for p in (path_list or []):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


def sitk_to_ants_xyzT(gas_xyzT, t):
    """Extract frame t from (X,Y,Z,T) array → ANTs image (axes already correct)."""
    arr = gas_xyzT[:, :, :, t].astype(np.float32)
    return ants.from_numpy(arr)


def register_ants(gas_xyzT, ei):
    fixed = sitk_to_ants_xyzT(gas_xyzT, ei)
    out = np.zeros_like(gas_xyzT)

    for t in range(gas_xyzT.shape[3]):
        print(f"  Frame {t+1:2d}/16  (EI={ei})", flush=True)
        if t == ei:
            out[:, :, :, t] = gas_xyzT[:, :, :, t]
            continue

        moving = sitk_to_ants_xyzT(gas_xyzT, t)
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNCC')
        fwd = reg['fwdtransforms']
        warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd)
        cleanup(fwd)
        cleanup(reg.get('invtransforms', []))
        out[:, :, :, t] = warped.numpy()

    return out


def save_gas(gas_xyzT, run_dir):
    out_dir = os.path.join(run_dir, 'ants')
    os.makedirs(out_dir, exist_ok=True)
    scipy.io.savemat(
        os.path.join(out_dir, 'gas.mat'),
        {'gas': gas_xyzT},
        do_compression=True,
    )
    print(f"  Saved → {out_dir}/gas.mat")


def main():
    for run_dir in RUNS_TO_PROCESS:
        print(f"\n{'='*55}")
        print(f"Run: {run_dir}")
        gas = load_gas(run_dir)
        ei  = find_ei(gas)
        print(f"EI frame: {ei}  |  shape: {gas.shape}")
        t0 = time.time()
        reg_gas = register_ants(gas, ei)
        print(f"ANTs done in {(time.time()-t0)/60:.1f} min")
        save_gas(reg_gas, run_dir)


if __name__ == '__main__':
    main()
