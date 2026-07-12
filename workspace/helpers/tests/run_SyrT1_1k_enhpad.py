"""
Full hierarchical groupwise PCA + ANTs for 2025-05-14_SyrT1
Iterations: [1000]  (single value, all 4 resolution levels)
Enhance + pad, no denoise, no clahe: run_flags=(1,0,0), pad=True
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from register_groupwise_PCA_step import (
    read_files,
    step_groupwise_registration,
    register_EI_ants_hoom,
    extract_image_3d,
    join_image3d_ants,
    _save_channels,
)

# ── Config ────────────────────────────────────────────────────────────────────
input_files       = '/Users/hoomham/Hooman/Work/Analysis/2025-05-14_SyrT1/rec/'
save_dir          = '/Users/hoomham/Hooman/Work/Analysis/2025-05-14_SyrT1/reg/1k_enhance_pad/'
run_flags         = (1, 0, 0)   # enhance only
pad               = True
save_intermediate = False
iterations        = ['1000']
refno             = 9           # EI = frame 9
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("Loading files...")
    image, dp_rbc, dp_mem, clahe = read_files(input_files)
    print(f"Loaded: {image.GetSize()}")

    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGroupwise PCA  iterations={iterations}  flags={run_flags}  pad={pad}")
    t0 = time.time()
    results = step_groupwise_registration(
        image, dp_rbc, dp_mem, clahe,
        save_dir,
        run_flags=run_flags,
        pad=pad,
        save_intermediate=save_intermediate,
        iterations=iterations,
    )
    t_gw = time.time() - t0
    print(f"Groupwise done in {t_gw/60:.1f} min")

    print(f"\nANTs SyNCC  refno={refno}")
    t1 = time.time()
    image3D = extract_image_3d(results['resultImage4d'])
    clahe3D = extract_image_3d(results['resultClahe4d'])
    rbc3D   = extract_image_3d(results['resultDpRbc4d'])
    mem3D   = extract_image_3d(results['resultDpMem4d'])

    reg_img, reg_clh, reg_rbc, reg_mem = register_EI_ants_hoom(
        image3D, clahe3D, rbc3D, mem3D, refno, save_dir
    )
    t_ants = time.time() - t1
    print(f"ANTs done in {t_ants/60:.1f} min")

    _save_channels(os.path.join(save_dir, 'final_ants'),
        gas=join_image3d_ants(reg_img),
        rbc=join_image3d_ants(reg_rbc),
        mem=join_image3d_ants(reg_mem),
        clahe=join_image3d_ants(reg_clh))

    print(f"\nTotal: {(t_gw+t_ants)/60:.1f} min  →  {save_dir}")


if __name__ == '__main__':
    main()
