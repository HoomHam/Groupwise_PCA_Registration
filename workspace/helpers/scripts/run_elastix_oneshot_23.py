"""
Adapter to run register_elastix_oneshot's groupwise PCA on a single-file
recon .mat (T-first axis, gas/dissolved_real/dissolved_imag keys, no clahe
channel) instead of the rspace/cspace/dspace triplet format that
read_files() in workspace/helpers/tests/register_elastix_oneshot.py expects.

gas_phase is duplicated into the clh4d slot -- it's warped like the other
channels internally but never saved, since run_flags=(0,0,0) means no
enhance/denoise/clahe stages run (single final pass only).
"""
import os
import sys
import time
import numpy as np
import SimpleITK as sitk
import scipy.io

sys.path.insert(0, '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/helpers/tests')
import register_elastix_oneshot as oneshot

INPUT_MAT = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/data/recon/23.mat'
SAVE_DIR  = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/outputs/registered/23_oneshot/'

RUN_FLAGS = (0, 0, 0)   # no enhance/denoise/clahe
PAD       = False       # -> single final Elastix pass
ITERATIONS = 500
CYCLIC     = True


def _to_4d(arr):
    """arr: (T, Z, Y, X) float32 -> SimpleITK 4D JoinSeries image."""
    vec = sitk.VectorOfImage()
    for t in range(arr.shape[0]):
        vec.push_back(sitk.GetImageFromArray(arr[t].astype(np.float32)))
    return sitk.JoinSeries(vec)


def read_23(path):
    d = scipy.io.loadmat(path)
    gas = _to_4d(d['gas_phase'])
    rbc = _to_4d(d['dissolved_phase_real'])
    mem = _to_4d(d['dissolved_phase_imag'])
    clh = _to_4d(d['gas_phase'])   # placeholder, discarded on save
    return gas, rbc, mem, clh


def main():
    print("Loading 23.mat...")
    image4d, dp_rbc4d, dp_mem4d, clahe4d = read_23(INPUT_MAT)

    print("Detecting EI...")
    ei = oneshot.find_ei_index(image4d)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Saving registration mask...")
    mask4d = oneshot.create_mask(image4d)
    mask_arr = sitk.GetArrayFromImage(mask4d)
    mask3d_img = sitk.GetImageFromArray(mask_arr[0].astype(np.uint8))
    mask3d_img.SetSpacing(image4d.GetSpacing()[:3])
    mask3d_img.SetOrigin(image4d.GetOrigin()[:3])
    sitk.WriteImage(mask3d_img, os.path.join(SAVE_DIR, 'mask.nii'))
    oneshot._save_mat(SAVE_DIR, 'mask', mask_arr[0].astype(np.uint8))

    print(f"Starting Elastix one-shot groupwise PCA (16 frames, iters={ITERATIONS})...")
    t0 = time.time()
    gas, rbc, mem, clh, jac = oneshot.orchestrate_elastix_staged(
        image4d, clahe4d, dp_rbc4d, dp_mem4d,
        SAVE_DIR, RUN_FLAGS, PAD, CYCLIC, ITERATIONS,
    )
    print(f"Registration done in {(time.time()-t0)/60:.1f} min.")

    jac_arr  = sitk.GetArrayFromImage(jac)
    jac_cons = jac_arr[1:] / jac_arr[:-1]
    jac_EE   = jac_arr / jac_arr[0:1]

    print("Saving...")
    oneshot.save_channels(SAVE_DIR, gas=gas, rbc=rbc, mem=mem, jac_det=jac)
    oneshot._save_mat(SAVE_DIR, 'jac_cons', jac_cons.astype(np.float32))
    oneshot._save_mat(SAVE_DIR, 'jac_from_EE', jac_EE.astype(np.float32))

    print(f"Done -> {SAVE_DIR}")
    print(f"EI frame: {ei}  |  jac_det: {jac_arr.shape}  |  jac_cons: {jac_cons.shape}  |  jac_from_EE: {jac_EE.shape}")


if __name__ == "__main__":
    main()
