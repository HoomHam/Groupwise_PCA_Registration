import os
import shutil
import time
import numpy as np
import SimpleITK as sitk
import scipy.io
import bm3d as bm3d_lib
from scipy.ndimage import label

# ── Config ───────────────────────────────────────────────────────────────────
input_files = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/'
save_dir    = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/elastix_oneshot_gas/'

# run_flags = (do_enhance, do_denoise, do_clahe)
# pad=False: only flagged stages run + mandatory final stage
# pad=True:  unflagged stages still run but use raw image as guide → always 4 total
run_flags  = (0, 0, 0)
pad        = False
iterations = 500
cyclic     = True   # cyclic transform for respiratory cycle

# Constants
THRESHOLD       = 0.04
TIME_THRESHOLD  = 5
BM3D_NOISE_VAR  = 0.02
ENHANCE_WEIGHT  = 7
# ─────────────────────────────────────────────────────────────────────────────


# ── I/O helpers ──────────────────────────────────────────────────────────────

def find_mat(directory, prefix):
    for fn in os.listdir(directory):
        if fn.startswith(prefix) and fn.endswith('.mat'):
            return os.path.join(directory, fn)
    raise FileNotFoundError(f"No .mat starting with '{prefix}' in {directory}")


def read_files(directory):
    def load_mat(path, key):
        data = scipy.io.loadmat(path)[key]   # (X, Y, Z, T)
        vec = sitk.VectorOfImage()
        for t in range(data.shape[3]):
            arr = np.transpose(data[:, :, :, t], (2, 1, 0)).astype(np.float32)
            vec.push_back(sitk.GetImageFromArray(arr))
        return sitk.JoinSeries(vec)

    image4d = load_mat(find_mat(directory, 'rspace'), 'image')
    clahe4d = load_mat(find_mat(directory, 'cspace'), 'clahe')

    cplx = scipy.io.loadmat(find_mat(directory, 'dspace'))['dspace']
    vr, vm = sitk.VectorOfImage(), sitk.VectorOfImage()
    for t in range(cplx.shape[3]):
        arr = np.transpose(cplx[:, :, :, t], (2, 1, 0))
        vr.push_back(sitk.GetImageFromArray(np.real(arr).astype(np.float32)))
        vm.push_back(sitk.GetImageFromArray(np.imag(arr).astype(np.float32)))
    dp_rbc4d = sitk.JoinSeries(vr)
    dp_mem4d = sitk.JoinSeries(vm)

    return image4d, dp_rbc4d, dp_mem4d, clahe4d


def save_channels(out_dir, **channels):
    os.makedirs(out_dir, exist_ok=True)
    for name, img in channels.items():
        if img is None:
            continue
        sitk.WriteImage(img, os.path.join(out_dir, f"{name}.nii"))
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 4:
            arr = np.transpose(arr, (3, 2, 1, 0))
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        scipy.io.savemat(
            os.path.join(out_dir, f"{name}.mat"),
            {name: arr},
            do_compression=True,
        )


# ── EI detection ─────────────────────────────────────────────────────────────

def find_ei_index(image4d):
    """Return index of the 3D frame with highest total signal sum."""
    arr = sitk.GetArrayFromImage(image4d)   # (T, Z, Y, X)
    sums = arr.reshape(arr.shape[0], -1).sum(axis=1)
    ei = int(np.argmax(sums))
    print(f"  Signal sums: {[f'{s:.1f}' for s in sums]}")
    print(f"  → EI auto-detected: frame {ei}")
    return ei


# ── Preprocessing ────────────────────────────────────────────────────────────

def create_mask(image4d, threshold=THRESHOLD, time_threshold=TIME_THRESHOLD):
    """Binary lung mask: threshold per-timepoint, keep largest connected component."""
    arr = sitk.GetArrayFromImage(image4d)   # (T, Z, Y, X)
    bin4d = (arr >= threshold).astype(np.uint8)
    summed = bin4d.sum(axis=0)
    mask3d = (summed >= time_threshold).astype(np.uint8)

    labels, _ = label(mask3d)
    sizes = np.bincount(labels.ravel())
    if sizes.size > 1:
        largest = np.argmax(sizes[1:]) + 1
        mask3d[labels != largest] = 0

    T = arr.shape[0]
    final = np.repeat(mask3d[np.newaxis], T, axis=0)

    vec = sitk.VectorOfImage()
    for t in range(T):
        m = sitk.GetImageFromArray(final[t].astype(np.uint8))
        m.SetSpacing(image4d.GetSpacing()[:3])
        m.SetOrigin(image4d.GetOrigin())
        vec.push_back(m)
    mask4d = sitk.JoinSeries(vec)
    mask4d.SetSpacing((image4d.GetSpacing()[-1],) + image4d.GetSpacing()[:3])
    return mask4d


def enhance_4d(image4d, target_weight=ENHANCE_WEIGHT):
    """
    Per-frame SNR enhancement over all T frames:
      enhanced_t = (w * frame_t + sum_others) / (w + T - 1)
    """
    arr = sitk.GetArrayFromImage(image4d)   # (T, Z, Y, X)
    T = arr.shape[0]
    total = arr.sum(axis=0)
    denom = float(target_weight + T - 1)
    out = np.empty_like(arr)
    for t in range(T):
        out[t] = (target_weight * arr[t] + (total - arr[t])) / denom

    vec = sitk.VectorOfImage()
    for t in range(T):
        sl = sitk.GetImageFromArray(out[t].astype(np.float32))
        sl.SetSpacing(image4d.GetSpacing()[:3])
        sl.SetOrigin(image4d.GetOrigin())
        vec.push_back(sl)
    enh4d = sitk.JoinSeries(vec)
    enh4d.SetSpacing(image4d.GetSpacing())
    enh4d.SetOrigin(image4d.GetOrigin())
    enh4d.SetDirection(image4d.GetDirection())
    return enh4d


def bm3d_4d(image4d, noise_var=BM3D_NOISE_VAR):
    """Slice-by-slice 2D BM3D denoising applied to every frame of a 4D image."""
    arr = sitk.GetArrayFromImage(image4d)   # (T, Z, Y, X)
    out = np.zeros_like(arr, dtype=np.float32)
    for t in range(arr.shape[0]):
        for z in range(arr.shape[1]):
            out[t, z] = bm3d_lib.bm3d(arr[t, z].astype(np.float64), noise_var).astype(np.float32)

    vec = sitk.VectorOfImage()
    for t in range(arr.shape[0]):
        sl = sitk.GetImageFromArray(out[t])
        sl.SetSpacing(image4d.GetSpacing()[:3])
        sl.SetOrigin(image4d.GetOrigin())
        vec.push_back(sl)
    den4d = sitk.JoinSeries(vec)
    den4d.SetSpacing(image4d.GetSpacing())
    den4d.SetOrigin(image4d.GetOrigin())
    den4d.SetDirection(image4d.GetDirection())
    return den4d


# ── Jacobian helper ──────────────────────────────────────────────────────────

def _jac_from_defField(df_4d):
    """
    Compute Jacobian determinant from a 4D deformation field.
    BSplineStackTransform deforms only the 3 spatial dims; 4th component (time) is dropped.
    Returns numpy array (T, Z, Y, X) float32 — avoids 4D SimpleITK ops unsupported in this build.
    """
    arr = sitk.GetArrayFromImage(df_4d)   # (T, Z, Y, X, 4)
    sp3 = df_4d.GetSpacing()[:3]
    or3 = df_4d.GetOrigin()[:3]
    jac = np.zeros(arr.shape[:4], dtype=np.float32)
    for t in range(arr.shape[0]):
        disp = sitk.GetImageFromArray(arr[t, ..., :3], isVector=True)
        disp.SetSpacing(sp3)
        disp.SetOrigin(or3)
        jac[t] = sitk.GetArrayFromImage(
            sitk.Cast(sitk.DisplacementFieldJacobianDeterminant(disp), sitk.sitkFloat32)
        )
    return jac


# ── Elastix groupwise ─────────────────────────────────────────────────────────

def _make_param_map(cyclic, iters):
    pm = sitk.GetDefaultParameterMap('groupwise')

    pm['UseCyclicTransform'] = ['true'] if cyclic else ['false']

    pm['FixedInternalImagePixelType']  = ['float']
    pm['MovingInternalImagePixelType'] = ['float']
    pm['FixedImageDimension']          = ['4']
    pm['MovingImageDimension']         = ['4']
    pm['UseDirectionCosines']          = ['true']

    pm['Registration']          = ['MultiResolutionRegistration']
    pm['Interpolator']          = ['ReducedDimensionBSplineInterpolator']
    pm['ResampleInterpolator']  = ['FinalReducedDimensionBSplineInterpolator']
    pm['Resampler']             = ['DefaultResampler']
    pm['BSplineInterpolationOrder']  = ['1']
    pm['BSplineTransformSplineOrder'] = ['3']
    pm['FixedImagePyramid']     = ['FixedSmoothingImagePyramid']
    pm['MovingImagePyramid']    = ['MovingSmoothingImagePyramid']
    pm['Optimizer']             = ['AdaptiveStochasticGradientDescent']
    pm['HowToCombineTransforms'] = ['Compose']
    pm['Transform']             = ['BSplineStackTransform']

    pm['Metric']        = ['PCAMetric2']
    pm['SubtractMean']  = ['true']
    pm['MovingImageDerivativeScales']     = ['1', '1', '1', '0']
    pm['(FinalGridSpacingInPhysicalUnits'] = ['6']

    pm['NumberOfResolutions']          = ['4']
    pm['AutomaticParameterEstimation'] = ['true']
    pm['ASGDParameterEstimationMethod'] = ['Original']
    pm['MaximumNumberOfIterations']    = [str(iters)]

    pm['GridSpacingSchedule']    = ['4', '3', '2', '1']
    pm['ImagePyramidSchedule']   = ['8','8','8','0','4','4','4','0','2','2','2','0','1','1','1','0']

    pm['NumberOfSpatialSamples']   = ['1024']
    pm['NewSamplesEveryIteration'] = ['true']
    pm['ImageSampler']             = ['RandomSparseMask']
    pm['CheckNumberOfSamples']     = ['true']

    pm['ErodeMask']       = ['false']
    pm['ErodeFixedMask']  = ['false']
    pm['ErodeMovingMask'] = ['false']

    pm['DefaultPixelValue']     = ['0']
    pm['WriteResultImage']      = ['false']   # result from GetResultImage(), skip disk write
    pm['ResultImagePixelType']  = ['float']

    return pm


def run_elastix_groupwise(cur_img4d, cur_rbc4d, cur_mem4d, cur_clh4d,
                           guide4d, elx_dir, cyclic, iters, prev_jac=None):
    """
    One Elastix groupwise PCA pass using guide4d as the registration driver.
    Transforms are applied to all 4 channels. Elastix output goes to elx_dir
    and is deleted on return.
    """
    os.makedirs(elx_dir, exist_ok=True)
    mask = create_mask(cur_img4d)

    ef = sitk.ElastixImageFilter()
    ef.SetFixedImage(guide4d)
    ef.SetMovingImage(guide4d)
    ef.SetFixedMask(mask)
    ef.SetMovingMask(mask)
    ef.SetOutputDirectory(elx_dir)
    ef.SetParameterMap(_make_param_map(cyclic, iters))
    ef.SetLogToConsole(False)
    ef.Execute()

    tx = ef.GetTransformParameterMap()

    def warp(vol):
        tf = sitk.TransformixImageFilter()
        tf.SetMovingImage(vol)
        tf.SetTransformParameterMap(tx)
        tf.SetOutputDirectory(elx_dir)
        tf.SetLogToConsole(False)
        tf.Execute()
        return tf.GetResultImage()

    reg_img = warp(cur_img4d)
    reg_rbc = warp(cur_rbc4d)
    reg_mem = warp(cur_mem4d)
    reg_clh = warp(cur_clh4d)

    # Jacobian determinant via deformation field (det getter not exposed in this build)
    tf_df = sitk.TransformixImageFilter()
    tf_df.SetMovingImage(sitk.Cast(cur_img4d, sitk.sitkFloat32))
    tf_df.SetTransformParameterMap(tx)
    tf_df.SetOutputDirectory(elx_dir)
    tf_df.SetLogToConsole(False)
    tf_df.ComputeDeformationFieldOn()
    tf_df.Execute()
    stage_jac_arr = _jac_from_defField(tf_df.GetDeformationField())

    # Compose with accumulated Jacobian from prior stages.
    # Element-wise multiply is a valid approximation for small respiratory deformations.
    if prev_jac is not None:
        stage_jac_arr = prev_jac * stage_jac_arr

    # Delete all Elastix output files
    try:
        shutil.rmtree(elx_dir)
    except Exception:
        pass

    return reg_img, reg_rbc, reg_mem, reg_clh, stage_jac_arr


# ── Multi-stage orchestration with flag/pad system ────────────────────────────

def orchestrate_elastix_staged(image4d, clahe4d, rbc4d, mem4d,
                                savedir, run_flags, pad, cyclic, iters):
    """
    Multi-stage Elastix groupwise PCA on the full 4D stack.

    run_flags = (do_enhance, do_denoise, do_clahe)
    pad=False: only flagged stages run + final  (min 1 reg)
    pad=True:  unflagged stages run with raw image guide → always 4 regs

    Stage order:
      0  Enhance   — guide = weighted-avg enhanced image (or raw if pad)
      1  BM3D      — guide = BM3D denoised current image (or raw if pad)
      2  CLAHE     — guide = current CLAHE channel        (or raw if pad)
      3  Final     — guide = current gas image            (always)
    """
    do_enhance, do_denoise, do_clahe = run_flags

    n_stages = sum([
        1 if (do_enhance or pad) else 0,
        1 if (do_denoise or pad) else 0,
        1 if (do_clahe   or pad) else 0,
        1,  # final always
    ])
    print(f"  run_flags={run_flags}  pad={pad}  cyclic={cyclic}  → {n_stages} Elastix pass(es)")

    cur_img, cur_rbc, cur_mem, cur_clh = image4d, rbc4d, mem4d, clahe4d
    cur_jac = None

    elx_base = os.path.join(savedir, '_elx_tmp')

    # Stage 0: Enhance
    if do_enhance or pad:
        tag = "enhance" if do_enhance else "pad_raw_for_enhance"
        print(f"  Stage 0 [{tag}]...")
        if do_enhance:
            print("    Building enhanced 4D guide...")
            guide = enhance_4d(cur_img)
        else:
            guide = cur_img
        cur_img, cur_rbc, cur_mem, cur_clh, cur_jac = run_elastix_groupwise(
            cur_img, cur_rbc, cur_mem, cur_clh, guide,
            os.path.join(elx_base, 'stage0'), cyclic, iters, cur_jac,
        )

    # Stage 1: BM3D Denoise
    if do_denoise or pad:
        tag = "bm3d" if do_denoise else "pad_raw_for_denoise"
        print(f"  Stage 1 [{tag}]...")
        if do_denoise:
            print("    BM3D denoising current 4D image...")
            guide = bm3d_4d(cur_img)
        else:
            guide = cur_img
        cur_img, cur_rbc, cur_mem, cur_clh, cur_jac = run_elastix_groupwise(
            cur_img, cur_rbc, cur_mem, cur_clh, guide,
            os.path.join(elx_base, 'stage1'), cyclic, iters, cur_jac,
        )

    # Stage 2: CLAHE
    if do_clahe or pad:
        tag = "clahe" if do_clahe else "pad_raw_for_clahe"
        print(f"  Stage 2 [{tag}]...")
        guide = cur_clh if do_clahe else cur_img
        cur_img, cur_rbc, cur_mem, cur_clh, cur_jac = run_elastix_groupwise(
            cur_img, cur_rbc, cur_mem, cur_clh, guide,
            os.path.join(elx_base, 'stage2'), cyclic, iters, cur_jac,
        )

    # Stage 3: Final (always)
    print("  Stage 3 [final image]...")
    cur_img, cur_rbc, cur_mem, cur_clh, cur_jac = run_elastix_groupwise(
        cur_img, cur_rbc, cur_mem, cur_clh, cur_img,
        os.path.join(elx_base, 'stage3'), cyclic, iters, cur_jac,
    )

    # Ensure _elx_tmp is fully gone
    try:
        shutil.rmtree(elx_base)
    except Exception:
        pass

    # Convert numpy jac (T,Z,Y,X) to 4D SimpleITK via JoinSeries for saving
    sp3 = cur_img.GetSpacing()[:3]
    or3 = cur_img.GetOrigin()[:3]
    jac_frames = []
    for t in range(cur_jac.shape[0]):
        frame = sitk.GetImageFromArray(cur_jac[t].astype(np.float32))
        frame.SetSpacing(sp3)
        frame.SetOrigin(or3)
        jac_frames.append(frame)
    jac4d = sitk.JoinSeries(jac_frames)
    jac4d.SetSpacing(cur_img.GetSpacing())
    jac4d.SetOrigin(cur_img.GetOrigin())

    return (
        sitk.Cast(cur_img, sitk.sitkFloat64),
        sitk.Cast(cur_rbc, sitk.sitkFloat64),
        sitk.Cast(cur_mem, sitk.sitkFloat64),
        sitk.Cast(cur_clh, sitk.sitkFloat64),
        jac4d,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_mat(out_dir, name, arr):
    if arr.ndim == 4:
        mat_arr = np.transpose(arr, (3, 2, 1, 0))
    elif arr.ndim == 3:
        mat_arr = np.transpose(arr, (2, 1, 0))
    else:
        mat_arr = arr
    scipy.io.savemat(
        os.path.join(out_dir, f"{name}.mat"),
        {name: mat_arr},
        do_compression=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading files...")
    image4d, dp_rbc4d, dp_mem4d, clahe4d = read_files(input_files)

    print("Detecting EI...")
    ei = find_ei_index(image4d)

    os.makedirs(save_dir, exist_ok=True)

    print("Saving registration mask...")
    mask4d = create_mask(image4d)
    mask_arr = sitk.GetArrayFromImage(mask4d)          # (T, Z, Y, X)
    mask3d_img = sitk.GetImageFromArray(mask_arr[0].astype(np.uint8))
    mask3d_img.SetSpacing(image4d.GetSpacing()[:3])
    mask3d_img.SetOrigin(image4d.GetOrigin()[:3])
    sitk.WriteImage(mask3d_img, os.path.join(save_dir, 'mask.nii'))
    _save_mat(save_dir, 'mask', mask_arr[0].astype(np.uint8))

    print(f"Starting Elastix one-shot groupwise PCA (16 frames, iters={iterations})...")
    t0 = time.time()
    gas, rbc, mem, clh, jac = orchestrate_elastix_staged(
        image4d, clahe4d, dp_rbc4d, dp_mem4d,
        save_dir, run_flags, pad, cyclic, iterations,
    )
    print(f"Registration done in {(time.time()-t0)/60:.1f} min.")

    jac_arr    = sitk.GetArrayFromImage(jac)            # (T, Z, Y, X) numpy
    jac_cons   = jac_arr[1:] / jac_arr[:-1]            # (15, Z, Y, X): bin t → t+1
    jac_EE     = jac_arr / jac_arr[0:1]                # (16, Z, Y, X): relative to EE bin 0

    print("Saving...")
    save_channels(save_dir, gas=gas, rbc=rbc, mem=mem, jac_det=jac)
    _save_mat(save_dir, 'jac_cons',    jac_cons.astype(np.float32))
    _save_mat(save_dir, 'jac_from_EE', jac_EE.astype(np.float32))

    print(f"Done → {save_dir}")
    print(f"EI frame: {ei}  |  jac_det: {jac_arr.shape}  |  jac_cons: {jac_cons.shape}  |  jac_from_EE: {jac_EE.shape}")


if __name__ == "__main__":
    main()
