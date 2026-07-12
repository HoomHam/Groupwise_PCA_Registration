import os
import sys
import time
import numpy as np
import SimpleITK as sitk
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from register_ants_oneshot import (
    read_files,
    extract_3d_frames,
    join_3d_to_4d,
    find_ei_index,
    register_ants_to_ei_staged,
)
from register_elastix_oneshot import orchestrate_elastix_staged

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILES = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/'
BASE_DIR    = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/'
ITERATIONS  = 500
CYCLIC      = True
# ─────────────────────────────────────────────────────────────────────────────

RUNS = [
    # (method,    run_flags,  pad,   subdir)
    # ── Step 1 & 2: raw image baseline ──
    ('ants',    (0,0,0), False, 'ants_image_only'),
    ('elastix', (0,0,0), False, 'elastix_image_only'),
    # ── Step 3: single-flag runs, pad and nonpad ──
    ('ants',    (1,0,0), False, 'ants_enhance_nonpad'),
    ('ants',    (1,0,0), True,  'ants_enhance_pad'),
    ('ants',    (0,1,0), False, 'ants_denoise_nonpad'),
    ('ants',    (0,1,0), True,  'ants_denoise_pad'),
    ('ants',    (0,0,1), False, 'ants_clahe_nonpad'),
    ('ants',    (0,0,1), True,  'ants_clahe_pad'),
    ('elastix', (1,0,0), False, 'elastix_enhance_nonpad'),
    ('elastix', (1,0,0), True,  'elastix_enhance_pad'),
    ('elastix', (0,1,0), False, 'elastix_denoise_nonpad'),
    ('elastix', (0,1,0), True,  'elastix_denoise_pad'),
    ('elastix', (0,0,1), False, 'elastix_clahe_nonpad'),
    ('elastix', (0,0,1), True,  'elastix_clahe_pad'),
]


def save_gas_mat(run_dir, gas4d):
    """Save registered gas channel as .mat only (MATLAB axis order: X,Y,Z,T)."""
    os.makedirs(run_dir, exist_ok=True)
    arr = sitk.GetArrayFromImage(gas4d)          # (T, Z, Y, X)
    arr = np.transpose(arr, (3, 2, 1, 0))        # → (X, Y, Z, T)
    scipy.io.savemat(
        os.path.join(run_dir, 'gas.mat'),
        {'gas': arr},
        do_compression=True,
    )


def main():
    print("=" * 60)
    print("Loading input data...")
    image4d, dp_rbc4d, dp_mem4d, clahe4d = read_files(INPUT_FILES)

    image3D  = extract_3d_frames(image4d)
    clahe3D  = extract_3d_frames(clahe4d)
    dp_rbc3D = extract_3d_frames(dp_rbc4d)
    dp_mem3D = extract_3d_frames(dp_mem4d)

    print("Detecting EI (used by ANTs runs)...")
    ei = find_ei_index(image3D)

    total_t0 = time.time()
    completed = []
    failed    = []

    for i, (method, flags, pad_flag, subdir) in enumerate(RUNS, 1):
        run_dir = os.path.join(BASE_DIR, subdir)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(RUNS)}]  {subdir}  |  flags={flags}  pad={pad_flag}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            if method == 'ants':
                out_img, _clh, _rbc, _mem = register_ants_to_ei_staged(
                    image3D, clahe3D, dp_rbc3D, dp_mem3D,
                    refno=ei, run_flags=flags, pad=pad_flag,
                )
                gas4d = join_3d_to_4d(out_img)

            else:  # elastix
                gas4d, _rbc, _mem, _clh = orchestrate_elastix_staged(
                    image4d, clahe4d, dp_rbc4d, dp_mem4d,
                    savedir=run_dir,
                    run_flags=flags, pad=pad_flag,
                    cyclic=CYCLIC, iters=ITERATIONS,
                )

            save_gas_mat(run_dir, gas4d)
            elapsed = (time.time() - t0) / 60
            print(f"  ✓ Done in {elapsed:.1f} min  →  {run_dir}/gas.mat")
            completed.append((subdir, elapsed))

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"  ✗ FAILED after {elapsed:.1f} min: {e}")
            failed.append((subdir, str(e)))

    total_elapsed = (time.time() - total_t0) / 60
    print(f"\n{'='*60}")
    print(f"All runs finished in {total_elapsed:.1f} min total.")
    print(f"Completed ({len(completed)}):")
    for name, t in completed:
        print(f"  {name:40s}  {t:.1f} min")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, err in failed:
            print(f"  {name:40s}  {err}")


if __name__ == "__main__":
    main()
