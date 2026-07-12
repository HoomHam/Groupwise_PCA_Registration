import os
import shutil
import time
import numpy as np
import SimpleITK as sitk
import scipy.io
from scipy.ndimage import label

INPUT_MAT = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/data/recon/49.mat'
SAVE_DIR  = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/outputs/registered/49_gasonly/'

CYCLIC     = True
ITERATIONS = 500

THRESHOLD      = 0.04
TIME_THRESHOLD = 5


def read_gas(path):
    arr = scipy.io.loadmat(path)['gas_phase']   # (T, Z, Y, X)
    vec = sitk.VectorOfImage()
    for t in range(arr.shape[0]):
        vec.push_back(sitk.GetImageFromArray(arr[t].astype(np.float32)))
    return sitk.JoinSeries(vec)


def create_mask(image4d, threshold=THRESHOLD, time_threshold=TIME_THRESHOLD):
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


def make_param_map(cyclic, iters):
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
    pm['WriteResultImage']      = ['false']
    pm['ResultImagePixelType']  = ['float']

    return pm


def jac_from_deffield(df_4d):
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


def save_mat(out_dir, name, arr):
    mat_arr = np.transpose(arr, (3, 2, 1, 0)) if arr.ndim == 4 else np.transpose(arr, (2, 1, 0))
    scipy.io.savemat(os.path.join(out_dir, f"{name}.mat"), {name: mat_arr}, do_compression=True)


def main():
    print("Loading gas_phase...")
    gas4d = read_gas(INPUT_MAT)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Building mask...")
    mask4d = create_mask(gas4d)
    mask_arr = sitk.GetArrayFromImage(mask4d)
    save_mat(SAVE_DIR, 'mask', mask_arr[0].astype(np.uint8))

    elx_dir = os.path.join(SAVE_DIR, '_elx_tmp')
    os.makedirs(elx_dir, exist_ok=True)

    print(f"Running groupwise PCAMetric2 registration (16 frames, iters={ITERATIONS})...")
    t0 = time.time()
    ef = sitk.ElastixImageFilter()
    ef.SetFixedImage(gas4d)
    ef.SetMovingImage(gas4d)
    ef.SetFixedMask(mask4d)
    ef.SetMovingMask(mask4d)
    ef.SetOutputDirectory(elx_dir)
    ef.SetParameterMap(make_param_map(CYCLIC, ITERATIONS))
    ef.SetLogToConsole(False)
    ef.Execute()
    tx = ef.GetTransformParameterMap()
    print(f"Registration done in {(time.time()-t0)/60:.1f} min.")

    tf = sitk.TransformixImageFilter()
    tf.SetMovingImage(gas4d)
    tf.SetTransformParameterMap(tx)
    tf.SetOutputDirectory(elx_dir)
    tf.SetLogToConsole(False)
    tf.Execute()
    reg_gas = tf.GetResultImage()

    tf_df = sitk.TransformixImageFilter()
    tf_df.SetMovingImage(sitk.Cast(gas4d, sitk.sitkFloat32))
    tf_df.SetTransformParameterMap(tx)
    tf_df.SetOutputDirectory(elx_dir)
    tf_df.SetLogToConsole(False)
    tf_df.ComputeDeformationFieldOn()
    tf_df.Execute()
    jac_arr = jac_from_deffield(tf_df.GetDeformationField())   # (T, Z, Y, X)

    shutil.rmtree(elx_dir, ignore_errors=True)

    print("Saving...")
    sitk.WriteImage(sitk.Cast(reg_gas, sitk.sitkFloat64), os.path.join(SAVE_DIR, 'gas.nii'))
    gas_out = sitk.GetArrayFromImage(reg_gas)   # (T, Z, Y, X)
    save_mat(SAVE_DIR, 'gas', gas_out.astype(np.float64))

    jac_cons = jac_arr[1:] / jac_arr[:-1]
    jac_EE   = jac_arr / jac_arr[0:1]
    save_mat(SAVE_DIR, 'jac_det', jac_arr.astype(np.float32))
    save_mat(SAVE_DIR, 'jac_cons', jac_cons.astype(np.float32))
    save_mat(SAVE_DIR, 'jac_from_EE', jac_EE.astype(np.float32))

    print(f"Done -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
