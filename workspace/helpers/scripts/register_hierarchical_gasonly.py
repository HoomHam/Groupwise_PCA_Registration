"""
Bare-bones hierarchical groupwise PCA registration, gas phase only.

Reproduces the 4->7->14->16 orchestration from step_groupwise_registration()
in register_groupwise_PCA_step.py, with every filtering stage removed
(no enhance, no BM3D denoise, no CLAHE guide) and no dissolved-phase
channels carried through (rbc/mem/clahe). Each stage that the original
would run through orchestrate_registration_workflow() with run_flags=(0,0,0),
pad=False collapses to exactly one register_groupwise() call (guide=image
itself), so this script calls that single-stage form directly.

The splice/average/transform helpers (process_images, update_images,
update_images_final, combine_transformed_and_registered,
apply_transformations_in_sequence, reverse_last_seven_images,
resort_to_original_order, combine_4d_images, combine_4d_itk_images,
extract_4d_subregion, average_4d_image_stack, register_groupwise,
TransformationPosition) are imported directly from the main pipeline
module -- they're channel-agnostic, so reusing them keeps this faithful
to the real splice semantics instead of re-deriving them.

Note: register_groupwise() never calls SetOutputDirectory, so Elastix
writes its working files into the current directory -- this script chdirs
into a scratch dir around each call and removes it afterward.
"""
import os
import sys
import shutil
import importlib.util
import numpy as np
import SimpleITK as sitk
import scipy.io

REPO_ROOT = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration'
INPUT_MAT = os.path.join(REPO_ROOT, 'workspace/data/recon/23.mat')
SAVE_DIR  = os.path.join(REPO_ROOT, 'workspace/outputs/registered/23_hierarchical_gasonly/')

ITERATIONS = ['500']

spec = importlib.util.spec_from_file_location(
    "register_groupwise_PCA_step",
    os.path.join(REPO_ROOT, "register_groupwise_PCA_step.py"),
)
orch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orch)


def read_gas(path):
    arr = scipy.io.loadmat(path)['gas_phase']   # (T, Z, Y, X)
    vec = sitk.VectorOfImage()
    for t in range(arr.shape[0]):
        vec.push_back(sitk.GetImageFromArray(arr[t].astype(np.float32)))
    return sitk.JoinSeries(vec)


def single_stage(img4d, elx_dir):
    os.makedirs(elx_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(elx_dir)
    try:
        reg_img, mask, _, _, _, T = orch.register_groupwise(
            img4d, img4d, img4d, img4d, img4d,
            elx_dir, time_threshold=1, cyclic=False, iterations=ITERATIONS,
        )
    finally:
        os.chdir(cwd)
    shutil.rmtree(elx_dir, ignore_errors=True)
    return sitk.Cast(reg_img, sitk.sitkFloat64), mask, T


def process_seven_block(img_blk, phase_dir):
    img4 = orch.process_images(img_blk, (0, 4))
    reg4, _, _ = single_stage(img4, os.path.join(phase_dir, 'stageA'))

    avg_img = orch.average_4d_image_stack(reg4)
    up_img = orch.update_images(img_blk, (4, 7), avg_img)
    reg43, _, T43 = single_stage(up_img, os.path.join(phase_dir, 'stageB'))

    t_img = orch.apply_transformations_in_sequence(
        reg4, [T43], phase_dir, orch.TransformationPosition.BEGINNING)
    combined_img = orch.combine_transformed_and_registered(
        t_img, reg43, index_transfer_range=(0, 4), index_register_range=(1, 4))
    final7_img = orch.process_images(combined_img, (0, 7))

    reg7, _, _ = single_stage(final7_img, os.path.join(phase_dir, 'stageD'))
    return reg7


def main():
    print("Loading gas_phase...")
    gas4d = read_gas(INPUT_MAT)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Phase 1: first 7 frames...")
    first7_blk = orch.process_images(gas4d, (0, 7))
    out_first = process_seven_block(first7_blk, os.path.join(SAVE_DIR, '01_phase1_first7'))

    print("Phase 2: last 7 frames (reversed)...")
    last7_blk = orch.reverse_last_seven_images(gas4d)
    out_last = process_seven_block(last7_blk, os.path.join(SAVE_DIR, '02_phase2_last7'))
    out_last = orch.resort_to_original_order(out_last)

    print("Phase 3: combine into 14 frames, register...")
    combined14 = orch.combine_4d_images([out_first], [out_last])
    reg14, _, _ = single_stage(combined14, os.path.join(SAVE_DIR, '03_phase3_combined14'))

    first_seven_14 = orch.extract_4d_subregion(reg14, 0, 7)
    last_seven_14 = orch.extract_4d_subregion(reg14, combined14.GetSize()[3] - 7, combined14.GetSize()[3])
    avg_first = orch.average_4d_image_stack(first_seven_14)
    avg_last = orch.average_4d_image_stack(last_seven_14)

    print("Phase 4: midpoint alignment (R8, R9)...")
    last_up_img = orch.update_images_final(gas4d, (7, 9), avg_first, avg_last)
    reg_mid, _, T_mid = single_stage(last_up_img, os.path.join(SAVE_DIR, '04_phase4_midpoint'))

    print("Phase 5: apply midpoint transform back to first7/last7...")
    phase5_dir = os.path.join(SAVE_DIR, '05_phase5_apply_back')
    transfered_first7 = orch.apply_transformations_in_sequence(
        first_seven_14, [T_mid], phase5_dir, orch.TransformationPosition.BEGINNING)
    transfered_last7 = orch.apply_transformations_in_sequence(
        last_seven_14, [T_mid], phase5_dir, orch.TransformationPosition.END)

    final_combined = orch.combine_transformed_and_registered(
        transfered_first7, reg_mid, index_transfer_range=(0, 7), index_register_range=(1, 3))

    print("Phase 6: assemble 16-frame stack...")
    combined16 = orch.combine_4d_itk_images(final_combined, transfered_last7)

    print("Phase 7: final 16-frame registration...")
    reg16, mask16, _ = single_stage(combined16, os.path.join(SAVE_DIR, '07_phase7_final16'))

    print("Saving...")
    sitk.WriteImage(reg16, os.path.join(SAVE_DIR, 'gas.nii'))
    gas_arr = sitk.GetArrayFromImage(reg16)   # (T, Z, Y, X)
    scipy.io.savemat(
        os.path.join(SAVE_DIR, 'gas.mat'),
        {'gas': np.transpose(gas_arr, (3, 2, 1, 0))},
        do_compression=True,
    )
    mask_arr = sitk.GetArrayFromImage(mask16)[0].astype(np.uint8)
    scipy.io.savemat(
        os.path.join(SAVE_DIR, 'mask.mat'),
        {'mask': np.transpose(mask_arr, (2, 1, 0))},
        do_compression=True,
    )

    print(f"Done -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
