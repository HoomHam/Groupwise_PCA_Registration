import os
import time
import numpy as np
import SimpleITK as sitk
import scipy.io
import ants
import nibabel as nib
import bm3d
from scipy.ndimage import label
from skimage import exposure
from enum import Enum

save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2025-05-06_Phantom/T100/reg/5/'
input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2025-05-06_Phantom/T100/rec/'

endinhale = 6 
refno = 6
threshold = 0.04  # Cutoff value for mask creation
time_threshold = 5

# Find the input files
def find_file_with_prefix(directory, prefix):
    for filename in os.listdir(directory):  
        if filename.startswith(prefix) and filename.endswith('.mat'):
            return os.path.join(directory, filename)
    raise ValueError(f"No file starting with '{prefix}' found in {directory}")

class TransformationPosition(Enum):
    BEGINNING = 1
    END = 2

def read_files(directory):
    """
    Load 4D volumes from `directory`:
      - gas‐space MAT (“rspace”, key='image’)
      - CLAHE MAT (“cspace”, key='clahe’)
      - dissolved‐phase complex MAT (“dspace”, key='dspace’) -> splits to dp_rbc & dp_mem
    Returns: (image4d, dp_rbc4d, dp_mem4d, clahe4d)
    """
    def find_file_with_prefix(prefix):
        for fn in os.listdir(directory):
            if fn.startswith(prefix) and fn.endswith('.mat'):
                return os.path.join(directory, fn)
        raise FileNotFoundError(f"No .mat starting with '{prefix}' in {directory}")

    def process_mat_file(path, key):
        mat = scipy.io.loadmat(path)
        data = mat.get(key)
        if data is None:
            raise KeyError(f"Key '{key}' not found in {path}")
        vec = sitk.VectorOfImage()
        # assume data.shape = [X, Y, Z, T]
        for t in range(data.shape[3]):
            arr3d = np.transpose(data[:, :, :, t], (2, 1, 0))
            vec.push_back(sitk.GetImageFromArray(arr3d))
        return sitk.JoinSeries(vec)

    # gas‐space
    image4d = process_mat_file(find_file_with_prefix('rspace'), 'image')

    # CLAHE‐space
    clahe4d = process_mat_file(find_file_with_prefix('cspace'), 'clahe')

    # dissolved‐phase complex → real/imag splits
    dpath   = find_file_with_prefix('dspace')
    mat     = scipy.io.loadmat(dpath)
    cplx    = mat.get('dspace')
    if cplx is None:
        raise KeyError(f"Key 'dspace' not found in {dpath}")
    vr, vm = sitk.VectorOfImage(), sitk.VectorOfImage()
    for t in range(cplx.shape[3]):
        arr = np.transpose(cplx[:, :, :, t], (2, 1, 0))
        vr.push_back(sitk.GetImageFromArray(np.real(arr)))
        vm.push_back(sitk.GetImageFromArray(np.imag(arr)))
    dp_rbc4d = sitk.JoinSeries(vr); dp_rbc4d.CopyInformation(image4d)
    dp_mem4d = sitk.JoinSeries(vm); dp_mem4d.CopyInformation(image4d)

    return image4d, dp_rbc4d, dp_mem4d, clahe4d

# def read_files(directory):
#     def process_mat_file(file_path, key):
#         _, file_extension = os.path.splitext(file_path)
#         if file_extension.lower() == '.mat':
#             mat_content = scipy.io.loadmat(file_path)
#             data_4d = mat_content.get(key)

#             if data_4d is None:
#                 raise ValueError(f"Key '{key}' not found in MATLAB file")

#             vector_of_images = sitk.VectorOfImage()
#             for t in range(data_4d.shape[3]):  # Change here to loop over the last dimension
#                 data_3d = data_4d[:, :, :, t]  # Change here to extract the 3D image at each time point
#                 data_3d_transposed = np.transpose(data_3d, (2, 1, 0))  # Transpose to (z, y, x)
    
#                 sitk_image = sitk.GetImageFromArray(data_3d_transposed)
#                 vector_of_images.push_back(sitk_image)

#             return sitk.JoinSeries(vector_of_images)
#         else:
#             raise ValueError("Unsupported file format")

#     # Find files with specific prefixes
#     image_file = find_file_with_prefix(directory, 'rspace')
#     dspace_file = find_file_with_prefix(directory, 'dspace') 
#     clahe_file = find_file_with_prefix(directory, 'cspace')

#     # Process images
#     images = process_mat_file(image_file, 'image')
#     clahe_images = process_mat_file(clahe_file, 'clahe')

# # --- Process dissolved‐phase complex data (dspace) ---
#     mat_content = scipy.io.loadmat(dspace_file)
#     data_cplx   = mat_content.get('dspace')
#     if data_cplx is None:
#         raise ValueError("Key 'dspace' not found in MATLAB file")

#     vec_rbc = sitk.VectorOfImage()
#     vec_mem = sitk.VectorOfImage()
#     for t in range(data_cplx.shape[3]):
#         arr = data_cplx[:, :, :, t]
#         arr = np.transpose(arr, (2, 1, 0))
#         # split into real (RBC) & imag (Membrane)
#         r = np.real(arr)
#         m = np.imag(arr)
#         vec_rbc.push_back(sitk.GetImageFromArray(r))
#         vec_mem.push_back(sitk.GetImageFromArray(m))

#     dp_rbc = sitk.JoinSeries(vec_rbc)
#     dp_rbc.CopyInformation(images)
#     dp_mem = sitk.JoinSeries(vec_mem)
#     dp_mem.CopyInformation(images)

#     # --- end DP processing ---
#     return images, dp_rbc, dp_mem, clahe_images

# print image sizez for debugging
def print_image_sizes(image_list):
    for i, img in enumerate(image_list):
        size = img.GetSize()
        print(f"Image {i} size: {size}")

# breaking down the 4D ITK images to 3D+t 
def extract_image_3d(image):
    # Get the size of the 4D image
    size = image.GetSize()
    if len(size) != 4:
        raise ValueError("Input image must be 4D.")

    # Extract each 3D image
    image_size = [size[0], size[1], size[2], 0]
    num_images = size[3]
    extracted_images = []
    for i in range(num_images):
        index = [0, 0, 0, i]
        sub_image = sitk.Extract(image, image_size, index)
        extracted_images.append(sub_image)

    return extracted_images


# combining 3D+t images to generate ANTs images
def join_image3d_ants(image3D):
    vector_of_images = sitk.VectorOfImage()
    for img in image3D:
        if not isinstance(img, sitk.Image):
            raise TypeError(f"Expected sitk.Image, got {type(img)}")
        if img.GetDimension() != 3:
            raise ValueError("Expected 3D image")
        vector_of_images.push_back(img)
    return sitk.JoinSeries(vector_of_images)

# creare mask from image
# def create_mask_from_4d_image(image4d, threshold=0.05, time_threshold=5):
#     # Convert 4D SimpleITK image to numpy array
#     image_array = sitk.GetArrayFromImage(image4d)

#     # Apply initial threshold to create a 4D mask
#     mask_4d = np.copy(image_array)
#     mask_4d[mask_4d < threshold] = 0
#     mask_4d[mask_4d >= threshold] = 1

#     # Sum across time dimension and apply time threshold
#     summed_image = np.sum(mask_4d, axis=0)
#     maskf = np.zeros_like(summed_image)
#     maskf[summed_image >= time_threshold] = 1
#     maskf[summed_image < time_threshold] = 0

#     # Connected component analysis
#     labeled_array, num_features = label(maskf)
#     sizes = np.bincount(labeled_array.ravel())
#     max_label = sizes[1:].argmax() + 1  # +1 because we ignore the background label (0)

#     # Isolating the largest component
#     maskf[labeled_array != max_label] = 0

#     # Replicate the 2D mask across all time points
#     final_mask = np.repeat(maskf[np.newaxis, :, :], image_array.shape[0], axis=0)

#     # Initialize a vector to store each 3D mask slice
#     mask_vector = sitk.VectorOfImage()

#     # Iterate over each time point to create a 3D mask slice
#     for i in range(final_mask.shape[0]):  # Iterate over the first dimension (time)
#         mask_slice = final_mask[i, :, :, :]  # Extract the 3D mask for the current time point
#         mask_slice_sitk = sitk.GetImageFromArray(mask_slice.astype(np.uint8))
#         mask_slice_sitk.SetSpacing(image4d.GetSpacing()[0:3])  # Set spacing for 3D
#         mask_slice_sitk.SetOrigin(image4d.GetOrigin())
#         mask_vector.push_back(mask_slice_sitk)

#     # Combine the 3D slices into a 4D image
#     mask_4d_sitk = sitk.JoinSeries(mask_vector)
#     mask_4d_sitk.SetSpacing((image4d.GetSpacing()[-1],) + image4d.GetSpacing()[0:3])  # Set 4D spacing

#     return mask_4d_sitk

def create_mask_from_4d_image(image4d, threshold=0.05, time_threshold=5):
    """
    Build a binary lung mask by thresholding each timepoint, summing across time,
    keeping only the largest connected component (if any), and replicating it back in time.
    """
    print("Creating Masks...")
    arr = sitk.GetArrayFromImage(image4d)  # [t, z, y, x]
    # 1) per‐timepoint threshold
    bin4d = (arr >= threshold).astype(np.uint8)
    # 2) sum across time
    summed = np.sum(bin4d, axis=0)
    mask2d = (summed >= time_threshold).astype(np.uint8)
    
    # 3) connected‐components on mask2d
    labels, _ = label(mask2d)
    sizes = np.bincount(labels.ravel())
    # if there's at least one non‐background label
    if sizes.size > 1:
        # find the largest component (ignore background at index 0)
        largest = np.argmax(sizes[1:]) + 1
        mask2d[labels != largest] = 0
    else:
        # everything is zero anyway
        mask2d[:] = 0

    # 4) replicate back into time
    t_dim = arr.shape[0]
    final = np.repeat(mask2d[np.newaxis, ...], t_dim, axis=0)

    # 5) re‐assemble into a SimpleITK 4D image
    vec = sitk.VectorOfImage()
    for t in range(t_dim):
        mimg = sitk.GetImageFromArray(final[t].astype(np.uint8))
        mimg.SetSpacing(image4d.GetSpacing()[0:3])
        mimg.SetOrigin( image4d.GetOrigin() )
        vec.push_back(mimg)

    mask4d = sitk.JoinSeries(vec)
    # ensure the 4D spacing is (time, x, y, z)
    mask4d.SetSpacing((image4d.GetSpacing()[-1],) + image4d.GetSpacing()[0:3])
    return mask4d

def create_bm3d_from_4d_image(image4d, noise_var = 0.02):
    print("BM3D Denoising...")
    # Convert SimpleITK images to numpy arrays
    image_array = sitk.GetArrayFromImage(image4d)  # Shape: [t, z, y, x]

    # Initialize a list to store the enhanced 3D SimpleITK images
    enhanced_images = []

    # Iterate over each 3D volume in the 4th dimension (time)
    for t in range(image_array.shape[0]):
        # Initialize an array to store the enhanced 3D volume for the current time point
        enhanced_3d_array = np.zeros(image_array[t].shape)  # Shape: [z, y, x]

        # Iterate over each slice in the 3rd dimension (z-axis)
        for z in range(image_array.shape[1]):
            # Extract the 2D slice and corresponding mask slice
            slice_2d = image_array[t, z, :, :]

            # Apply CLAHE only to the masked regions
            # enhanced_slice = np.where(mask_slice, exposure.equalize_adapthist(slice_2d, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins), slice_2d)
            enhanced_slice = bm3d.bm3d(slice_2d, noise_var)

            # Store the enhanced slice back into the 3D volume
            enhanced_3d_array[z, :, :] = enhanced_slice

        # Convert the enhanced 3D numpy array back to a SimpleITK image for the current time point
        enhanced_3d_image = sitk.GetImageFromArray(enhanced_3d_array, isVector=False)
        enhanced_3d_image.CopyInformation(sitk.Extract(image4d, image4d.GetSize()[:3] + (0,), (0, 0, 0, t)))
        enhanced_images.append(enhanced_3d_image)

    # Combine the enhanced 3D SimpleITK images into a 4D image
    enhanced_image4d = sitk.JoinSeries(enhanced_images)
    enhanced_image4d.SetSpacing(image4d.GetSpacing())
    enhanced_image4d.SetOrigin(image4d.GetOrigin())
    enhanced_image4d.SetDirection(image4d.GetDirection())

    return enhanced_image4d

# Adding images to improve SNR for end-exhale images
def enhance_selected_bins(image4d, index_sets, target_weights):
    print("Enhance Images by Averaging...")
    if len(index_sets) != len(target_weights):
        raise ValueError("Length of index_sets and target_weights must be equal")

    # Convert 4D SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(image4d)

    # Initialize an array to store the enhanced images
    enhanced_image_array = np.copy(image_array)

    # Flatten the index sets into a single set for easy checking
    all_indices = set().union(*index_sets)

    # Iterate over each bin in the 4D image
    for bin_index in range(image_array.shape[0]):
        # Check if the current bin is in any of the index sets
        if bin_index in all_indices:
            # Find the set that contains the current bin
            for set_index, index_set in enumerate(index_sets):
                if bin_index in index_set:
                    target_weight = target_weights[set_index]
                    # Calculate the weighted sum for the current bin
                    weighted_sum = target_weight * image_array[bin_index]
                    for other_bin in index_set:
                        if other_bin != bin_index:
                            weighted_sum += image_array[other_bin]

                    # Calculate the total weight
                    total_weight = target_weight + len(index_set) - 1

                    # Store the enhanced image
                    enhanced_image_array[bin_index] = weighted_sum / total_weight
                    break
        # If the bin is not in any index set, leave it unchanged
        else:
            enhanced_image_array[bin_index] = image_array[bin_index]

    # Transpose the array to match SimpleITK's expected order
    transposed_array = np.transpose(enhanced_image_array, (1, 2, 3, 0))

    # Create a vector to store each 3D image slice
    vector_of_images = sitk.VectorOfImage()
    for i in range(transposed_array.shape[3]):
        # Extract the 3D image for the current time point
        image_3d = transposed_array[:, :, :, i]
        image_3d_sitk = sitk.GetImageFromArray(image_3d)
        image_3d_sitk.SetSpacing(image4d.GetSpacing()[0:3])  # Set spacing for 3D
        image_3d_sitk.SetOrigin(image4d.GetOrigin())
        vector_of_images.push_back(image_3d_sitk)

    # Combine the 3D slices into a 4D image
    enhanced_image4d = sitk.JoinSeries(vector_of_images)
    enhanced_image4d.SetSpacing((image4d.GetSpacing()[-1],) + image4d.GetSpacing()[0:3])  # Set 4D spacing

    return enhanced_image4d

# groupwise 4D (3D+t) registration w/ PCA2 [Huiziga]
def register_groupwise(
    image4d, registeror, dp_rbc4d, dp_mem4d, clahe4d, savedir,
    time_threshold, cyclic,
    debug=False,
    save_flags=(True, True, True)
):
    """
       
    Single groupwise‐PCA run on 4D image, producing
    image, mask, registeror, clahe, dp_rbc & dp_mem outputs.
    
    Args:
      image4d: gas-space 4D SITK image
      dp_rbc4d: RBC 4D SITK image
      dp_mem4d: MEM 4D SITK image
      savedir: output directory
      time_threshold: time threshold for mask
      cyclic: whether to use cyclic transform
      debug: if False, suppress all intermediate writes and prints
      save_flags: tuple of booleans (save_init_mask, save_warped, save_final_mask)
    Returns:
      reg_img, reg_mask, reg_rbc, reg_mem, elastix_filter
    """

    import os
    os.makedirs(savedir, exist_ok=True)
    save_init_mask, save_warped, save_final_mask = save_flags

    # 1) initial mask
    mask = create_mask_from_4d_image(image4d, threshold, time_threshold)
    if save_init_mask:
        sitk.WriteImage(mask, os.path.join(savedir, "ns_mask_init.nii"))
        if debug:
            print(f"[register_groupwise] wrote initial mask to {savedir}")

    sitk.WriteImage(mask, os.path.join(savedir, "masktest.nii"))


    # 2) Elastix setup
    elastixFilter = sitk.ElastixImageFilter()
    elastixFilter.SetFixedImage(registeror)
    elastixFilter.SetMovingImage(registeror)
    elastixFilter.SetFixedMask(mask)
    elastixFilter.SetMovingMask(mask)

    parameterMap = sitk.GetDefaultParameterMap('groupwise')
    parameterMap['UseCyclicTransform'] = ['true'] if cyclic else ['false']
    
        # Define Image Type
    parameterMap['FixedInternalImagePixelType'] = ['float']
    parameterMap['MovingInternalImagePixelType'] = ['float']
    parameterMap['FixedImageDimension'] = ['4']
    parameterMap['MovingImageDimension'] = ['4']
    parameterMap['UseDirectionCosines'] = ['true']

    # Components
    parameterMap['Registration'] = ['MultiResolutionRegistration']
    parameterMap['Interpolator'] = ['ReducedDimensionBSplineInterpolator']
    parameterMap['ResampleInterpolator'] = ['FinalReducedDimensionBSplineInterpolator']
    parameterMap['Resampler'] = ['DefaultResampler']
    parameterMap['BSplineInterpolationOrder'] = ['1']
    parameterMap['BSplineTransformSplineOrder'] = ['3']
    parameterMap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
    parameterMap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    parameterMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    parameterMap['HowToCombineTransforms'] = ['Compose']
    parameterMap['Transform'] = ['BSplineStackTransform']      
    
    # Groupwise Metric
    parameterMap['Metric'] = ['PCAMetric2']
    # parameterMap['Metric'] = ['SumOfPairwiseCorrelationCoefficientsMetric']

    parameterMap['SubtractMean'] = ['true']
    parameterMap['MovingImageDerivativeScales'] = ['1','1','1','0']
    parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']
    
    # Optimizer Setting
    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['AutomaticParameterEstimation'] = ['true']
    parameterMap['ASGDParameterEstimationMethod'] = ['DisplacementDistribution']
    # parameterMap['MaximumNumberOfIterations'] = ['10000', '20000', '30000', '40000']
    # parameterMap['MaximumNumberOfIterations'] = ['20000', '40000', '60000', '80000']
    # parameterMap['MaximumNumberOfIterations'] = ['10000']
    # parameterMap['MaximumNumberOfIterations'] = ['100']
    # parameterMap['MaximumNumberOfIterations'] = ['10000']
    parameterMap['MaximumNumberOfIterations'] = ['5']
    # parameterMap['MaximumNumberOfIterations'] = ['1000']
    
    # Pyramid Setting
    parameterMap['GridSpacingSchedule'] = ['4','3','2','1']
    parameterMap['ImagePyramidSchedule'] = ['8','8','8' ,'0','4','4','4','0','2','2','2','0','1','1','1','0']
    
    # Sampler Parameters
    parameterMap['NumberOfSpatialSamples'] = ['1024']
    # parameterMap['NumberOfSpatialSamples'] = ['10']
    parameterMap['NewSamplesEveryIteration'] = ['true']
    parameterMap['ImageSampler'] = ['RandomSparseMask']
    # parameterMap['ImageSampler'] = ['RandomCoordinate']
    parameterMap['CheckNumberOfSamples'] = ['true']

    # Mask Setting
    parameterMap['ErodeMask'] = ['false']
    parameterMap['ErodeFixedMask'] = ['false']
    parameterMap['ErodeMovingMask'] = ['false']

    # Output Setting
    parameterMap['DefaultPixelValue'] = ['0']
    parameterMap['WriteResultImage'] = ['true']
    parameterMap['ResultImagePixelType'] = ['float']

    elastixFilter.SetParameterMap(parameterMap)
    elastixFilter.Execute()
    tx_map = elastixFilter.GetTransformParameterMap()

    # helper warp
    def _warp(vol, name):
        out = None
        tx = sitk.TransformixImageFilter()
        tx.SetMovingImage(vol)
        tx.SetTransformParameterMap(tx_map)
        tx.Execute()
        out = tx.GetResultImage()
        if save_warped:
            sitk.WriteImage(out, os.path.join(savedir, f"ns_{name}_groupwise.nii"))
            if debug:
                print(f"[register_groupwise] wrote warped {name} to {savedir}")
        return out

    reg_img = _warp(image4d, "image")
    # reg_reg = _warp(registeror,"registeror")
    reg_rbc = _warp(dp_rbc4d, "rbc")
    reg_mem = _warp(dp_mem4d, "mem")
    reg_clahe = _warp(clahe4d, "clahe")

    # 3) final mask
    final_mask = create_mask_from_4d_image(reg_img, threshold, time_threshold)
    if save_final_mask:
        sitk.WriteImage(final_mask, os.path.join(savedir, "ns_mask_groupwise.nii"))
        if debug:
            print(f"[register_groupwise] wrote final mask to {savedir}")

    # return reg_img, final_mask, reg_reg, reg_rbc, reg_mem, reg_clahe, elastixFilter
    return reg_img, final_mask, reg_rbc, reg_mem, reg_clahe, elastixFilter



# def register_groupwise(image, registeror, dp_rbc, dp_mem, savedir, time_threshold, cyclic):

#     """
#     Single groupwise‐PCA run on 4D image, producing
#     image, mask, registeror, clahe, dp_rbc & dp_mem outputs.
#     """
#     # ensure the folder exists
#     os.makedirs(savedir, exist_ok=True)

#     # Creating Mask from Images
#     mask = create_mask_from_4d_image(image, threshold, time_threshold)
#     sitk.WriteImage(mask, os.path.join(savedir,"ns_mask_init.nii"))

#     elastixImageFilter=sitk.ElastixImageFilter()
#     elastixImageFilter.SetFixedImage(registeror)
#     elastixImageFilter.SetMovingImage(registeror)

#     elastixImageFilter.SetFixedMask(mask)
#     elastixImageFilter.SetMovingMask(mask)

#     # Define Groupwise Methodology
#     parameterMap = sitk.GetDefaultParameterMap('groupwise')

#     if cyclic:
#         parameterMap['UseCyclicTransform'] = ['true']
#     else:
#         parameterMap['UseCyclicTransform'] = ['false']

#     # Define Image Type
#     parameterMap['FixedInternalImagePixelType'] = ['float']
#     parameterMap['MovingInternalImagePixelType'] = ['float']
#     parameterMap['FixedImageDimension'] = ['4']
#     parameterMap['MovingImageDimension'] = ['4']
#     parameterMap['UseDirectionCosines'] = ['true']

#     # Components
#     parameterMap['Registration'] = ['MultiResolutionRegistration']
#     parameterMap['Interpolator'] = ['ReducedDimensionBSplineInterpolator']
#     parameterMap['ResampleInterpolator'] = ['FinalReducedDimensionBSplineInterpolator']
#     parameterMap['Resampler'] = ['DefaultResampler']
#     parameterMap['BSplineInterpolationOrder'] = ['1']
#     parameterMap['BSplineTransformSplineOrder'] = ['3']
#     parameterMap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
#     parameterMap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
#     parameterMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']
#     parameterMap['HowToCombineTransforms'] = ['Compose']
#     parameterMap['Transform'] = ['BSplineStackTransform']      
    
#     # Groupwise Metric
#     parameterMap['Metric'] = ['PCAMetric2']
#     # parameterMap['Metric'] = ['SumOfPairwiseCorrelationCoefficientsMetric']

#     parameterMap['SubtractMean'] = ['true']
#     parameterMap['MovingImageDerivativeScales'] = ['1','1','1','0']
#     parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']
    
#     # Optimizer Setting
#     parameterMap['NumberOfResolutions'] = ['4']
#     parameterMap['AutomaticParameterEstimation'] = ['true']
#     parameterMap['ASGDParameterEstimationMethod'] = ['DisplacementDistribution']
#     # parameterMap['MaximumNumberOfIterations'] = ['10000', '20000', '30000', '40000']
#     # parameterMap['MaximumNumberOfIterations'] = ['20000', '40000', '60000', '80000']
#     # parameterMap['MaximumNumberOfIterations'] = ['10000']
#     # parameterMap['MaximumNumberOfIterations'] = ['100']
#     # parameterMap['MaximumNumberOfIterations'] = ['10000']
#     parameterMap['MaximumNumberOfIterations'] = ['5000']
#     # parameterMap['MaximumNumberOfIterations'] = ['1000']
    
#     # Pyramid Setting
#     parameterMap['GridSpacingSchedule'] = ['4','3','2','1']
#     parameterMap['ImagePyramidSchedule'] = ['8','8','8' ,'0','4','4','4','0','2','2','2','0','1','1','1','0']
    
#     # Sampler Parameters
#     parameterMap['NumberOfSpatialSamples'] = ['1024']
#     # parameterMap['NumberOfSpatialSamples'] = ['10']
#     parameterMap['NewSamplesEveryIteration'] = ['true']
#     parameterMap['ImageSampler'] = ['RandomSparseMask']
#     # parameterMap['ImageSampler'] = ['RandomCoordinate']
#     parameterMap['CheckNumberOfSamples'] = ['true']

#     # Mask Setting
#     parameterMap['ErodeMask'] = ['false']
#     parameterMap['ErodeFixedMask'] = ['false']
#     parameterMap['ErodeMovingMask'] = ['false']

#     # Output Setting
#     parameterMap['DefaultPixelValue'] = ['0']
#     parameterMap['WriteResultImage'] = ['true']
#     parameterMap['ResultImagePixelType'] = ['float']

#     elastixImageFilter.SetParameterMap(parameterMap)
#     elastixImageFilter.Execute()

#     tx_map = elastixImageFilter.GetTransformParameterMap()

#     # Helper to run Transformix once
#     def _warp(vol, name):
#         tx = sitk.TransformixImageFilter()
#         tx.SetMovingImage(vol)
#         tx.SetTransformParameterMap(tx_map)
#         tx.Execute()
#         out = tx.GetResultImage()
#         sitk.WriteImage(out, os.path.join(savedir, f"ns_{name}_groupwise.nii"))
#         return out

#     # 3) Warp each stream
#     resultImage4d     = _warp(image,     "image")
#     resultRegisteror4d= _warp(registeror,"registeror")
#     resultDpRbc4d     = _warp(dp_rbc,    "dp_rbc")
#     resultDpMem4d     = _warp(dp_mem,    "dp_mem")

#     # 4) Recompute mask on the warped image
#     resultMask4d = create_mask_from_4d_image(resultImage4d, threshold, time_threshold)
#     sitk.WriteImage(resultMask4d, os.path.join(savedir, "ns_mask_groupwise.nii"))

#     # 5) Return all six outputs + the Elastix filter for chaining
#     return (
#         resultImage4d,
#         resultMask4d,
#         resultRegisteror4d,
#         resultDpRbc4d,
#         resultDpMem4d,
#         elastixImageFilter
#     )

def orchestrate_registration_workflow(
    image4d: sitk.Image,
    clahe4d: sitk.Image,
    dp_rbc4d: sitk.Image,
    dp_mem4d: sitk.Image,
    savedir: str,
    target_weights: list[int],
    index_sets:    list[set[int]],
    cyclic: bool,
    time_threshold: int,
    run_flags:    tuple[int,int,int,int] = (1,1,1,1),
    debug: bool = False
) -> dict:
    """
    Multi‑stage groupwise PCA registration pipeline.

    run_flags = (do_enhance, do_denoise, do_clahe, do_image)
    """
    do_enhance, do_denoise, do_clahe, do_image = run_flags

    def _maybe_write(img: sitk.Image, stage: str, fname: str):
        if debug:
            outdir = os.path.join(savedir, stage)
            os.makedirs(outdir, exist_ok=True)
            sitk.WriteImage(img, os.path.join(outdir, fname))

    # Top‑level save dir
    os.makedirs(savedir, exist_ok=True)

    #
    # Stage 0: SNR enhancement
    #
    if do_enhance:
        enh = enhance_selected_bins(image4d, index_sets, target_weights)
        _maybe_write(enh, "stage0", "00-enhanced.nii")
        regor0 = enh
    else:
        regor0 = image4d

    stage0 = os.path.join(savedir, "stage0")
    os.makedirs(stage0, exist_ok=True)
    img0, msk0, rbc0, mem0, clh0, T0 = register_groupwise(
        image4d, regor0, dp_rbc4d, dp_mem4d, clahe4d, 
        savedir, time_threshold = 1, cyclic = False
    )
    _maybe_write(msk0, "stage0", "00-mask.nii")

    #
    # Stage 1: BM3D denoise
    #
    if do_denoise:
        filt = create_bm3d_from_4d_image(img0)
        _maybe_write(filt, "stage1", "10-filtered.nii")
        regor1 = filt
    else:
        regor1 = img0

    stage1 = os.path.join(savedir, "stage1")
    os.makedirs(stage1, exist_ok=True)
    img1, msk1, rbc1, mem1, clh1, T1 = register_groupwise(
        img0, regor1, rbc0, mem0, clh0,
        stage1, time_threshold = 1, cyclic = False
    )
    _maybe_write(msk1, "stage1", "10-mask.nii")

    #
    # Stage 2: CLAHE
    #
    if do_clahe:
        _maybe_write(clahe4d, "stage2", "20-clahe-input.nii")
        sitk.WriteImage(clahe4d, os.path.join(savedir, "clahethatsucks.nii"))
        regor2 = clahe4d
    else:
        regor2 = img1

    stage2 = os.path.join(savedir, "stage2")
    os.makedirs(stage2, exist_ok=True)
    print('I Stuck Here!')
    img2, msk2, rbc2, mem2, clh2, T2 = register_groupwise(
        img1, regor2, rbc1, mem1, clh1,
        stage2, time_threshold = 1, cyclic = False
    )
    _maybe_write(msk2, "stage2", "20-mask.nii")

    #
    # Stage 3: image‑as‑registeror
    #
    regor3 = img2 if do_image else img2

    stage3 = os.path.join(savedir, "stage3")
    os.makedirs(stage3, exist_ok=True)
    img3, msk3, rbc3, mem3, clh3, T3 = register_groupwise(
        img2, regor3, rbc2, mem2, clh2,
        stage3, time_threshold = 1, cyclic = False
    )
    _maybe_write(msk3, "stage3", "30-mask.nii")

    # Final 64‑bit cast & write
    final_img = sitk.Cast(img3, sitk.sitkFloat64)
    final_rbc = sitk.Cast(rbc3, sitk.sitkFloat64)
    final_mem = sitk.Cast(mem3, sitk.sitkFloat64)
    final_clh = sitk.Cast(clh3, sitk.sitkFloat64)

    sitk.WriteImage(final_img, os.path.join(savedir, "final_image.nii"))
    sitk.WriteImage(final_rbc, os.path.join(savedir, "final_rbc.nii"))
    sitk.WriteImage(final_mem, os.path.join(savedir, "final_mem.nii"))
    sitk.WriteImage(final_clh, os.path.join(savedir, "final_clahe.nii"))
    sitk.WriteImage(msk3,      os.path.join(savedir, "final_mask.nii"))

    return {
        "resultImage4d":   final_img,
        "resultMask4d":    msk3,
        "resultDpRbc4d":   final_rbc,
        "resultDpMem4d":   final_mem,
        "resultClahe4d":   final_clh,
        "transformations": [T0, T1, T2, T3],
        "run_flags":       run_flags,
    }


# def orchestrate_registration_workflow(image, dp_rbc, dp_mem,
#                                       savedir, target_weights, index_sets, cyclic):
#     print('Image...')
#     # 4) Raw‐image run
#     img, msk, reg, dp_rbc, dp_mem, T = register_groupwise(
#         image, image, dp_rbc, dp_mem,
#         os.path.join(savedir, 'groupwise/3/'),
#         time_threshold=1,
#         cyclic=cyclic
#     )

#     # Convert to float64 if you like
#     img_64   = sitk.Cast(img,   sitk.sitkFloat64)
#     dp_rbc   = sitk.Cast(dp_rbc,   sitk.sitkFloat64)
#     dp_mem   = sitk.Cast(dp_mem,   sitk.sitkFloat64)

#     # Collect transforms
#     transformations = T

#     # Return all four 4D series
#     return {
#         "resultImage4d":   img_64,
#         "resultMask4d":    msk,
#         "resultRegisteror4d": reg,
#         "resultDpRbc4d":   dp_rbc,
#         "resultDpMem4d":   dp_mem,
#         "transformations": transformations
#     }

def average_4d_image_stack(image4d):
    """Calculate the average of a 4D SimpleITK image stack across the time dimension.

    Args:
        image4d (SimpleITK.Image): A 4D SimpleITK image.

    Returns:
        SimpleITK.Image: A 3D SimpleITK image representing the average.
    """
    # Convert the 4D image to a numpy array
    image_array = sitk.GetArrayFromImage(image4d)

    # Calculate the average across the time dimension
    average_array = np.mean(image_array, axis=0)

    # Convert the average array back to a SimpleITK image
    average_image = sitk.GetImageFromArray(average_array)
    average_image.CopyInformation(sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, 0)))

    return average_image


def apply_transformations_in_sequence(image4d, transformations, savedir, position):
    # Ensure transformations are provided in the order: Te, Tn, Tc, Ti
    # if len(transformations) != 4:
    # raise ValueError("Expected four transformation parameter maps: Te, Tn, Tc, Ti")

    num_images = image4d.GetSize()[-1]
    transformed_images = []

    if num_images == 4:
        # Logic for four images remains the same as your original implementation
        # for i in range(num_images):
        #     current_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0,0, 0, i))
        #     reordered_images = [current_image] + [sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j)) for j in range(num_images) if j != i]
        #     reordered_image_4d = sitk.JoinSeries(reordered_images)
            
        # Loop over each image in the 4D stack
        for i in range(num_images):
            # Extract the 3D image at position i
            current_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, i))

            # Initialize a list to store the reordered 3D images
            reordered_images = [current_image]

            # Loop over the remaining indices and extract the 3D images, excluding the current one
            for j in range(4):
                if j != i:
                    other_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j))
                    reordered_images.append(other_image)
            
            # Combine the reordered 3D images into a 4D image
            reordered_image_4d = sitk.JoinSeries(reordered_images)               

            for transformation in transformations:
                transformix = sitk.TransformixImageFilter()
                transformix.SetMovingImage(reordered_image_4d)
                transformix.SetTransformParameterMap(transformation.GetTransformParameterMap())
                transformix.Execute()
                reordered_image_4d = transformix.GetResultImage()

            transformed_image = sitk.Extract(reordered_image_4d, reordered_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, 0))
            transformed_images.append(transformed_image)

    elif num_images == 7:
        # Logic for seven images
        for i in range(num_images):
            current_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, i))
            if position == TransformationPosition.BEGINNING:
                # Place the current image at the beginning of the stack
                reordered_images = [current_image] + [sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j)) for j in range(1, 4)]  # Using first three arbitrary images
            elif position == TransformationPosition.END:
                # Place the current image at the end of the stack
                reordered_images = [sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j)) for j in range(num_images - 4, num_images - 1)] + [current_image]  # Using last three arbitrary images

            reordered_image_4d = sitk.JoinSeries(reordered_images)

            # for transformation in transformations:
            for transformation in transformations:
                transformix = sitk.TransformixImageFilter()
                transformix.SetMovingImage(reordered_image_4d)
                transformix.SetTransformParameterMap(transformation.GetTransformParameterMap())
                transformix.Execute()
                reordered_image_4d = transformix.GetResultImage()

            # Extract the transformed current image
            transformed_image = sitk.Extract(reordered_image_4d, reordered_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, 0 if position == TransformationPosition.BEGINNING else 3))
            transformed_images.append(transformed_image)

    # Combine the transformed 3D images back into a 4D image
    transformed_images_64 = [sitk.Cast(img, sitk.sitkFloat64) for img in transformed_images]
    resultImage4d = sitk.JoinSeries(transformed_images_64)

    # Save the final transformed 4D image
    sitk.WriteImage(resultImage4d, os.path.join(savedir, "ns_transformed_image.nii"))

    return resultImage4d

# def apply_transformations_in_sequence(image4d, transformations, savedir, position):
#     # Ensure transformations are provided in the order: Te, Tn, Tc, Ti
#     # if len(transformations) != 4:
#     # raise ValueError("Expected four transformation parameter maps: Te, Tn, Tc, Ti")

#     num_images = image4d.GetSize()[-1]
#     transformed_images = []

#     if num_images == 4:
#         # Logic for four images remains the same as your original implementation
#         # for i in range(num_images):
#         #     current_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0,0, 0, i))
#         #     reordered_images = [current_image] + [sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j)) for j in range(num_images) if j != i]
#         #     reordered_image_4d = sitk.JoinSeries(reordered_images)
            
#         # Loop over each image in the 4D stack
#         for i in range(num_images):
#             # Extract the 3D image at position i
#             current_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, i))

#             # Initialize a list to store the reordered 3D images
#             reordered_images = [current_image]

#             # Loop over the remaining indices and extract the 3D images, excluding the current one
#             for j in range(4):
#                 if j != i:
#                     other_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j))
#                     reordered_images.append(other_image)
            
#             # Combine the reordered 3D images into a 4D image
#             reordered_image_4d = sitk.JoinSeries(reordered_images)               

#             # for transformation in transformations:
#             transformix = sitk.TransformixImageFilter()
#             transformix.SetMovingImage(reordered_image_4d)
#             transformix.SetTransformParameterMap(transformations.GetTransformParameterMap())
#             transformix.Execute()
#             reordered_image_4d = transformix.GetResultImage()

#             transformed_image = sitk.Extract(reordered_image_4d, reordered_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, 0))
#             transformed_images.append(transformed_image)

#     elif num_images == 7:
#         # Logic for seven images
#         for i in range(num_images):
#             current_image = sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, i))
#             if position == TransformationPosition.BEGINNING:
#                 # Place the current image at the beginning of the stack
#                 reordered_images = [current_image] + [sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j)) for j in range(1, 4)]  # Using first three arbitrary images
#             elif position == TransformationPosition.END:
#                 # Place the current image at the end of the stack
#                 reordered_images = [sitk.Extract(image4d, image4d.GetSize()[0:3] + (0,), (0, 0, 0, j)) for j in range(num_images - 4, num_images - 1)] + [current_image]  # Using last three arbitrary images

#             reordered_image_4d = sitk.JoinSeries(reordered_images)

#             # for transformation in transformations:
#             transformix = sitk.TransformixImageFilter()
#             transformix.SetMovingImage(reordered_image_4d)
#             transformix.SetTransformParameterMap(transformations.GetTransformParameterMap())
#             transformix.Execute()
#             reordered_image_4d = transformix.GetResultImage()

#             # Extract the transformed current image
#             transformed_image = sitk.Extract(reordered_image_4d, reordered_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, 0 if position == TransformationPosition.BEGINNING else 3))
#             transformed_images.append(transformed_image)

#     # Combine the transformed 3D images back into a 4D image
#     transformed_images_64 = [sitk.Cast(img, sitk.sitkFloat64) for img in transformed_images]
#     resultImage4d = sitk.JoinSeries(transformed_images_64)

#     # Save the final transformed 4D image
#     sitk.WriteImage(resultImage4d, os.path.join(savedir, "ns_transformed_image.nii"))

#     return resultImage4d
 
def process_images(image, index_range):
    # Cutting the 4D stack from start_index to end_index
    start_index, end_index = index_range
    Image4d = sitk.JoinSeries([sitk.Extract(image, image.GetSize()[0:3] + (0,), (0, 0, 0, i)) for i in range(start_index, end_index)]) 

    return Image4d

def update_images(image, index_range, average_image):
    # Adding the average_image to the new images from index_range
    start_index, end_index = index_range
    
    original_size = list(image.GetSize())
    extract_size = original_size[:3] + [0]

    updatedImage4d = sitk.VectorOfImage()
    updatedImage4d.push_back(average_image)
    print(type(average_image))
    print(average_image.GetPixelIDTypeAsString())

    for i in range(start_index, end_index):
        updatedImage4d.push_back(sitk.Cast(sitk.Extract(image, extract_size, [0, 0, 0, i]), sitk.sitkFloat64))        
    
    finalUpdatedImage4d = sitk.JoinSeries(updatedImage4d)
    
    return finalUpdatedImage4d

def update_images_final(image, index_range, average_image_first, average_image_last):
    # Adding the average_image to the new images from index_range
    start_index, end_index = index_range
    
    original_size = list(image.GetSize())
    extract_size = original_size[:3] + [0]
    
    updatedImage4d = sitk.VectorOfImage()

    updatedImage4d.push_back(average_image_first)
    
    for i in range(start_index, end_index):
        # updatedImage4d.push_back(sitk.Extract(image, extract_size, [0, 0, 0, i])) 
        updatedImage4d.push_back(sitk.Cast(sitk.Extract(image, extract_size, [0, 0, 0, i]), sitk.sitkFloat64))               
    
    updatedImage4d.push_back(average_image_last)

    finalUpdatedImage4d = sitk.JoinSeries(updatedImage4d)
    
    return finalUpdatedImage4d    

def combine_transformed_and_registered(transfered_image, registered_image, index_transfer_range, index_register_range):

    original_size = list(transfered_image.GetSize())
    extract_size = original_size[:3] + [0]

    # Combine the transferred and registered images
    combined_image = sitk.VectorOfImage()

    start_transfer, end_transfer = index_transfer_range
    start_register, end_register = index_register_range

    for i in range(start_transfer, end_transfer):
        combined_image.push_back(sitk.Extract(transfered_image, extract_size, [0, 0, 0, i]))

    for i in range(start_register, end_register):
        combined_image.push_back(sitk.Extract(registered_image, extract_size, [0, 0, 0, i]))

    finalCombinedImage4d = sitk.JoinSeries(combined_image)

    return finalCombinedImage4d

def reverse_last_seven_images(image):
    original_size = list(image.GetSize())
    extract_size = original_size[:3] + [0]
    
    reversedImage4d = sitk.VectorOfImage()

    # Start from the last image and go backwards
    for i in range(original_size[3] - 1, original_size[3] - 8, -1):
        reversedImage4d.push_back(sitk.Extract(image, extract_size, [0, 0, 0, i]))

    reversedImage4d = sitk.JoinSeries(reversedImage4d)
    return reversedImage4d

def resort_to_original_order(reversed_processed_image):
    original_size = list(reversed_processed_image.GetSize())
    extract_size = original_size[:3] + [0]
    
    sortedImage4d = sitk.VectorOfImage()

    # Start from the last image of the processed stack and go backwards
    for i in range(original_size[3] - 1, original_size[3] - 8, -1):
        sortedImage4d.push_back(sitk.Extract(reversed_processed_image, extract_size, [0, 0, 0, i]))

    sortedImage4d = sitk.JoinSeries(sortedImage4d)
    return sortedImage4d

def combine_4d_images(list_of_first_images, list_of_last_images):
    combined_images = sitk.VectorOfImage()

    # Iterate over each 4D image in the first list
    for first_image_4d in list_of_first_images:
        for i in range(first_image_4d.GetSize()[-1]):
            img_3d = sitk.Extract(first_image_4d, first_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, i))
            combined_images.push_back(img_3d)

    # Iterate over each 4D image in the last list
    for last_image_4d in list_of_last_images:
        for i in range(last_image_4d.GetSize()[-1]):
            img_3d = sitk.Extract(last_image_4d, last_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, i))
            combined_images.push_back(img_3d)

    # Combine all 3D images into a new 4D image
    new_image_4d = sitk.JoinSeries(combined_images)

    return new_image_4d

def combine_4d_itk_images(image4d_1, image4d_2):
    # Initialize a list to hold the combined set of 3D images
    combined_images = []

    # Extract and append all 3D images from the first 4D image
    for i in range(image4d_1.GetSize()[-1]):  # Iterate over the time dimension
        img_3d = sitk.Extract(image4d_1, image4d_1.GetSize()[0:3] + (0,), (0, 0, 0, i))
        combined_images.append(img_3d)

    # Extract and append all 3D images from the second 4D image
    for i in range(image4d_2.GetSize()[-1]):  # Iterate over the time dimßension
        img_3d = sitk.Extract(image4d_2, image4d_2.GetSize()[0:3] + (0,), (0, 0, 0, i))
        combined_images.append(img_3d)

    # Use JoinSeries to combine all 3D images into a new 4D image
    combined_image_4d = sitk.JoinSeries(combined_images)

    return combined_image_4d

def extract_4d_subregion(image_4d, start_time, end_time):
    extracted_images = []
    for t in range(start_time, end_time):
        # Extract the 3D image at time point t
        img_3d = sitk.Extract(image_4d, image_4d.GetSize()[0:3] + (0,), (0, 0, 0, t))
        extracted_images.append(img_3d)

    # Combine the extracted 3D images into a new 4D image
    new_image_4d = sitk.JoinSeries(extracted_images)
    
    return new_image_4d

# def step_groupwise_registration(image, dp_rbc, dp_mem, savedir):
#     """
#     Perform the multi‐stage groupwise registration on 4D image, CLAHE, and dissolved‐phase channels.
#     """
#     # Define the first/last seven ranges
#     index_ranges = [(0, 7), (image.GetSize()[3] - 7, image.GetSize()[3])]

#     # Storage for processed 4D stacks
#     processed_images_first_seven = []
#     processed_images_last_seven  = []
#     processed_rbcs_first_seven   = []
#     processed_rbcs_last_seven    = []
#     processed_mems_first_seven   = []
#     processed_mems_last_seven    = []

#     for start_index, end_index in index_ranges:
#         # Make sure the block folder exists
#         block_dir = os.path.join(savedir, str(start_index))
#         os.makedirs(block_dir, exist_ok=True)

#         # Reverse for the last‐seven block
#         if start_index > 0:
#             image_to_work = reverse_last_seven_images(image)
#             rbc_to_work   = reverse_last_seven_images(dp_rbc)
#             mem_to_work   = reverse_last_seven_images(dp_mem)
#         else:
#             image_to_work = image
#             rbc_to_work   = dp_rbc
#             mem_to_work   = dp_mem

#         # Stage A: cut 4 bins
#         Image4d = process_images(image_to_work, index_range=(0, 4))
#         Rbc4d   = process_images(rbc_to_work,   index_range=(0, 4))
#         Mem4d   = process_images(mem_to_work,   index_range=(0, 4))

#         registration_four_results = orchestrate_registration_workflow(
#             Image4d, Rbc4d, Mem4d,
#             block_dir,
#             target_weights=[7],
#             index_sets=[{0,1,2,3}],
#             cyclic=False
#         )

#         # Compute averages
#         avg_img = average_4d_image_stack(registration_four_results["resultImage4d"])
#         avg_rbc = average_4d_image_stack(registration_four_results["resultDpRbc4d"])
#         avg_mem = average_4d_image_stack(registration_four_results["resultDpMem4d"])

#         # Stage B: update bins 4–7 with averages
#         UpdatedImage4d = update_images(image_to_work, index_range=(4, 7), average_image=avg_img)
#         UpdatedRbc4d   = update_images(rbc_to_work,   index_range=(4, 7), average_image=avg_rbc)
#         UpdatedMem4d   = update_images(mem_to_work,   index_range=(4, 7), average_image=avg_mem)

#         registration_four_three_results = orchestrate_registration_workflow(
#             UpdatedImage4d, UpdatedRbc4d, UpdatedMem4d,
#             block_dir,
#             target_weights=[7],
#             index_sets=[{0,1,2,3}],
#             cyclic=False
#         )

#         # Stage C: warp back into full 7‐bin series
#         Ts = registration_four_three_results["transformations"]

#         t_img = apply_transformations_in_sequence(
#             registration_four_results["resultImage4d"], Ts, block_dir, TransformationPosition.BEGINNING
#         )

#         combined_img = combine_transformed_and_registered(
#             t_img, registration_four_three_results["resultImage4d"],
#             index_transfer_range=(0,4), index_register_range=(1,4)
#         )

#         t_rbc = apply_transformations_in_sequence(
#             registration_four_results["resultDpRbc4d"], Ts, block_dir, TransformationPosition.BEGINNING
#         )

#         combined_rbc = combine_transformed_and_registered(
#             t_rbc, registration_four_three_results["resultDpRbc4d"],
#             index_transfer_range=(0,4), index_register_range=(1,4)
#         )

#         t_mem = apply_transformations_in_sequence(
#             registration_four_results["resultDpMem4d"], Ts, block_dir, TransformationPosition.BEGINNING
#         )

#         combined_mem = combine_transformed_and_registered(
#             t_mem, registration_four_three_results["resultDpMem4d"],
#             index_transfer_range=(0,4), index_register_range=(1,4)
#         )

#         FinalUpdatedImage4d = process_images(combined_img, index_range=(0,7))
#         FinalUpdatedRbc4d   = process_images(combined_rbc, index_range=(0,7))
#         FinalUpdatedMem4d   = process_images(combined_mem, index_range=(0,7))

#         # Stage D: 7‐bin PCA
#         registration_seven_results = orchestrate_registration_workflow(
#             FinalUpdatedImage4d, FinalUpdatedRbc4d, FinalUpdatedMem4d,
#             block_dir,
#             target_weights=[14],
#             index_sets=[set(range(7))],
#             cyclic=False
#         )

#         # Restore original order & collect
#         if start_index > 0:
#             registration_seven_results["resultImage4d"]   = resort_to_original_order(registration_seven_results["resultImage4d"])
#             registration_seven_results["resultDpRbc4d"]   = resort_to_original_order(registration_seven_results["resultDpRbc4d"])
#             registration_seven_results["resultDpMem4d"]   = resort_to_original_order(registration_seven_results["resultDpMem4d"])
#             processed_images_last_seven.append(registration_seven_results["resultImage4d"])
#             processed_rbcs_last_seven.append(registration_seven_results["resultDpRbc4d"])
#             processed_mems_last_seven.append(registration_seven_results["resultDpMem4d"])
#         else:
#             processed_images_first_seven.append(registration_seven_results["resultImage4d"])
#             processed_rbcs_first_seven.append(registration_seven_results["resultDpRbc4d"])
#             processed_mems_first_seven.append(registration_seven_results["resultDpMem4d"])

#     # Merge first & last seven into a 14‑bin set
#     combined_image_4d = combine_4d_images(processed_images_first_seven, processed_images_last_seven)
#     combined_rbc_4d   = combine_4d_images(processed_rbcs_first_seven, processed_rbcs_last_seven)
#     combined_mem_4d   = combine_4d_images(processed_mems_first_seven, processed_mems_last_seven)

#     # ---- NEW: build & save the merged 14-frame mask for inspection ----
#     combined_mask_4d = create_mask_from_4d_image(
#         combined_image_4d,
#         threshold=threshold,
#         time_threshold=time_threshold
#     )
#     sitk.WriteImage(
#         combined_mask_4d,
#         os.path.join(savedir, "ns_combined_mask_groupwise.nii")
#     )

#     sitk.WriteImage(combined_image_4d, os.path.join(savedir, "ns_combined_image_groupwise.nii"))
#     sitk.WriteImage(combined_rbc_4d, os.path.join(savedir, "ns_combined_rbc_groupwise.nii"))
#     sitk.WriteImage(combined_mem_4d, os.path.join(savedir, "ns_combined_mem_groupwise.nii"))

#     # 14‑bin cyclic PCA
#     registration_fourteen_results = orchestrate_registration_workflow(
#         combined_image_4d, combined_rbc_4d, combined_mem_4d,
#         savedir,
#         target_weights=[30],
#         index_sets=[set(range(14))],
#         cyclic=True
#     )

#     return registration_fourteen_results


def step_groupwise_registration(
    image4d: sitk.Image,
    dp_rbc4d: sitk.Image,
    dp_mem4d: sitk.Image,
    clahe4d:  sitk.Image,
    savedir:  str,
    run_flags: tuple[int,int,int,int] = (1,1,1,1),
    debug: bool = False
) -> dict:
    """
    1) Split into two 7‑frame blocks (reverse the 2nd)
    2) For each block: Stage A–D via orchestrate_registration_workflow
    3) Un‑reverse 2nd, collect registered gas/RBC/MEM
    4) Stitch into 14 frames
    5) [opt] insert 2 middle averages → 16 frames
    6) Final cyclic PCA over all frames
    """

    os.makedirs(savedir, exist_ok=True)
    N = image4d.GetSize()[3]
    blocks = [(0,7), (N-7, N)]

    first_imgs, first_rbcs, first_mems, first_clhs   = [], [], [], []
    last_imgs,  last_rbcs,  last_mems, last_clhs    = [], [], [], []

    for start, end in blocks:
        block_dir = os.path.join(savedir, str(start))
        os.makedirs(block_dir, exist_ok=True)

        # pick the 7‐frame substack (and reverse if it's the “last”)
        if start > 0:
            img_blk = reverse_last_seven_images(image4d)
            rbc_blk = reverse_last_seven_images(dp_rbc4d)
            mem_blk = reverse_last_seven_images(dp_mem4d)
            clh_blk  = reverse_last_seven_images(clahe4d)
        else:
            img_blk, rbc_blk, mem_blk, clh_blk = image4d, dp_rbc4d, dp_mem4d, clahe4d

        # ── Stage A: register **frames 0–3** ──
        img4 = process_images(img_blk, (0,4))
        rbc4 = process_images(rbc_blk, (0,4))
        mem4 = process_images(mem_blk, (0,4))
        clh4  = process_images(clh_blk,  (0,4))

        res4 = orchestrate_registration_workflow(
            image4d    = img4,
            clahe4d     = clh4,
            dp_rbc4d    = rbc4,
            dp_mem4d    = mem4,
            savedir     = os.path.join(block_dir, "stageA"),
            target_weights=[7],
            index_sets   =[ {0,1,2,3} ],
            cyclic      = False,
            time_threshold =  time_threshold
        )

        # ── Stage B: rebuild **frames 4–6** by inserting average(0–3) ──
        avg_img = average_4d_image_stack(res4["resultImage4d"])
        avg_rbc = average_4d_image_stack(res4["resultDpRbc4d"])
        avg_mem = average_4d_image_stack(res4["resultDpMem4d"])
        avg_clh  = average_4d_image_stack(res4["resultClahe4d"])

        up_img = update_images(img_blk, (4,7), avg_img)
        up_rbc = update_images(rbc_blk, (4,7), avg_rbc)
        up_mem = update_images(mem_blk, (4,7), avg_mem)
        up_clh  = update_images(clh_blk,  (4,7), avg_clh)

        res43 = orchestrate_registration_workflow(
            image4d    = up_img,
            clahe4d     = up_clh,
            dp_rbc4d    = up_rbc,
            dp_mem4d    = up_mem,
            savedir     = os.path.join(block_dir, "stageB"),
            target_weights=[7],
            index_sets   =[ {0,1,2,3} ],
            cyclic      = False,
            time_threshold =  time_threshold
        )

        # ── Stage C: warp back & **splice** into a 7‑frame block ──
        Ts = res43["transformations"]

        t_img = apply_transformations_in_sequence(
            res4["resultImage4d"], Ts, os.path.join(block_dir,"stageC"), TransformationPosition.BEGINNING
        )
        combined_img = combine_transformed_and_registered(
            t_img, res43["resultImage4d"],
            index_transfer_range=(0,4), index_register_range=(1,4)
        )

        t_rbc = apply_transformations_in_sequence(
            res4["resultDpRbc4d"], Ts, os.path.join(block_dir,"stageC"), TransformationPosition.BEGINNING
        )
        combined_rbc = combine_transformed_and_registered(
            t_rbc, res43["resultDpRbc4d"],
            index_transfer_range=(0,4), index_register_range=(1,4)
        )

        t_mem = apply_transformations_in_sequence(
            res4["resultDpMem4d"], Ts, os.path.join(block_dir,"stageC"), TransformationPosition.BEGINNING
        )
        combined_mem = combine_transformed_and_registered(
            t_mem, res43["resultDpMem4d"],
            index_transfer_range=(0,4), index_register_range=(1,4)
        )

        t_clh = apply_transformations_in_sequence(
            res4["resultClahe4d"], Ts, os.path.join(block_dir,"stageC"), TransformationPosition.BEGINNING
        )
        combined_clh = combine_transformed_and_registered(
            t_clh, res43["resultClahe4d"],
            index_transfer_range=(0,4), index_register_range=(1,4)
        )

        # now cut back to exactly 7 frames
        final7_img = process_images(combined_img, (0,7))
        final7_rbc = process_images(combined_rbc, (0,7))
        final7_mem = process_images(combined_mem, (0,7))
        final7_clh  = process_images(combined_clh,  (0,7))

        # ── Stage D: 7‑frame PCA ──
        res7 = orchestrate_registration_workflow(
            image4d    = final7_img,
            clahe4d     = final7_clh,
            dp_rbc4d    = final7_rbc,
            dp_mem4d    = final7_mem,
            savedir     = os.path.join(block_dir, "stageD"),
            target_weights=[14],
            index_sets   =[ set(range(7)) ],
            cyclic      = False,
            time_threshold =  time_threshold
        )

        # if it was the “last” block, undo the reverse order:
        out_img = res7["resultImage4d"]
        out_rbc = res7["resultDpRbc4d"]
        out_mem = res7["resultDpMem4d"]
        out_clh  = res7["resultClahe4d"]
        
        if start > 0:
            out_img = resort_to_original_order(out_img)
            out_rbc = resort_to_original_order(out_rbc)
            out_mem = resort_to_original_order(out_mem)
            out_clh  = resort_to_original_order(out_clh)
            last_imgs.append(out_img)
            last_rbcs.append(out_rbc)
            last_mems.append(out_mem)
            last_clhs.append(out_clh)
        else:
            first_imgs.append(out_img)
            first_rbcs.append(out_rbc)
            first_mems.append(out_mem)
            first_clhs.append(out_clh)

        # --- Merge into a 14‑frame series ---
    combined14_img   = combine_4d_images(first_imgs, last_imgs)                                        
    combined14_rbc   = combine_4d_images(first_rbcs, last_rbcs)                                        
    combined14_mem   = combine_4d_images(first_mems,last_mems)                                        
    combined14_clahe = combine_4d_images(first_clhs, last_clhs)

    if debug:
        sitk.WriteImage(combined14_img,   os.path.join(savedir,"14_combined_image.nii"))
        sitk.WriteImage(combined14_clahe, os.path.join(savedir,"14_combined_clahe.nii"))
        sitk.WriteImage(combined14_rbc,   os.path.join(savedir,"14_combined_rbc.nii"))
        sitk.WriteImage(combined14_mem,   os.path.join(savedir,"14_combined_mem.nii"))

    res14 = orchestrate_registration_workflow(
        image4d    = combined14_img,
        clahe4d     = combined14_clahe,
        dp_rbc4d    = combined14_rbc,
        dp_mem4d    = combined14_mem,
        savedir     = os.path.join(savedir, "stageE"),
        target_weights=[30],
        index_sets   =[ set(range(14)) ],
        cyclic      = True,
        time_threshold =  time_threshold
    )        

    first_seven_image_4d = extract_4d_subregion(res14["resultImage4d"], 0, 7)
    last_seven_image_4d = extract_4d_subregion(res14["resultImage4d"], combined14_img.GetSize()[3] - 7, combined14_img.GetSize()[3])
    average_image_first = average_4d_image_stack(first_seven_image_4d)
    average_image_last = average_4d_image_stack(last_seven_image_4d)

    first_seven_rbc_4d = extract_4d_subregion(res14["resultDpRbc4d"], 0, 7)
    last_seven_rbc_4d = extract_4d_subregion(res14["resultDpRbc4d"], combined14_img.GetSize()[3] - 7, combined14_img.GetSize()[3])
    average_rbc_first = average_4d_image_stack(first_seven_rbc_4d)
    average_rbc_last = average_4d_image_stack(last_seven_rbc_4d)

    first_seven_mem_4d = extract_4d_subregion(res14["resultDpMem4d"], 0, 7)
    last_seven_mem_4d = extract_4d_subregion(res14["resultDpMem4d"], combined14_img.GetSize()[3] - 7, combined14_img.GetSize()[3])
    average_mem_first = average_4d_image_stack(first_seven_mem_4d)
    average_mem_last = average_4d_image_stack(last_seven_mem_4d)

    first_seven_clahe_4d = extract_4d_subregion(res14["resultClahe4d"], 0, 7)
    last_seven_clahe_4d = extract_4d_subregion(res14["resultClahe4d"], combined14_img.GetSize()[3] - 7, combined14_img.GetSize()[3])
    average_clahe_first = average_4d_image_stack(first_seven_clahe_4d)
    average_clahe_last = average_4d_image_stack(last_seven_clahe_4d)

    last_up_img = update_images_final(image4d, index_range=(7, 9), average_image_first = average_image_first, average_image_last = average_image_last)
    last_up_rbc = update_images_final(dp_rbc4d, index_range=(7, 9), average_image_first = average_rbc_first, average_image_last = average_rbc_last)
    last_up_mem = update_images_final(dp_mem4d, index_range=(7, 9), average_image_first = average_mem_first, average_image_last = average_mem_last)
    last_up_clh = update_images_final(clahe4d, index_range=(7, 9), average_image_first = average_clahe_first, average_image_last = average_clahe_last)

    res142 = orchestrate_registration_workflow(
        image4d    = last_up_img,
        clahe4d     = last_up_clh,
        dp_rbc4d    = last_up_rbc,
        dp_mem4d    = last_up_mem,
        savedir     = os.path.join(block_dir, "stageL"),
        target_weights=[7],
        index_sets   =[ {0,1,2,3} ],
        cyclic      = True,
        time_threshold =  time_threshold        
    )

    Tf = res142["transformations"]


    transfered_first_seven_image_output = apply_transformations_in_sequence(first_seven_image_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.BEGINNING)
    transfered_last_seven_image_output = apply_transformations_in_sequence(last_seven_image_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.END)

    finalCombinedImage4d = combine_transformed_and_registered(
        transfered_first_seven_image_output, res142["resultImage4d"], 
        index_transfer_range=(0, 7), index_register_range=(1, 3)
    )


    transfered_first_seven_rbc_output = apply_transformations_in_sequence(first_seven_rbc_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.BEGINNING)
    transfered_last_seven_rbc_output = apply_transformations_in_sequence(last_seven_rbc_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.END)

    finalCombinedDpRbc4d = combine_transformed_and_registered(
        transfered_first_seven_rbc_output, res142["resultDpRbc4d"], 
        index_transfer_range=(0, 7), index_register_range=(1, 3)
    )


    transfered_first_seven_mem_output = apply_transformations_in_sequence(first_seven_mem_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.BEGINNING)
    transfered_last_seven_mem_output = apply_transformations_in_sequence(last_seven_mem_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.END)

    finalCombinedDpMem4d = combine_transformed_and_registered(
        transfered_first_seven_mem_output, res142["resultDpMem4d"], 
        index_transfer_range=(0, 7), index_register_range=(1, 3)
    )


    transfered_first_seven_clahe_output = apply_transformations_in_sequence(first_seven_clahe_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.BEGINNING)
    transfered_last_seven_clahe_output = apply_transformations_in_sequence(last_seven_clahe_4d, Tf, os.path.join(block_dir,"stageD"), TransformationPosition.END)

    finalCombinedClahe4d = combine_transformed_and_registered(
        transfered_first_seven_clahe_output, res142["resultClahe4d"],
        index_transfer_range=(0, 7), index_register_range=(1, 3)
    )


    combined16_img = combine_4d_itk_images(finalCombinedImage4d, transfered_last_seven_image_output)
    combined16_rbc = combine_4d_itk_images(finalCombinedDpRbc4d, transfered_last_seven_rbc_output)
    combined16_mem = combine_4d_itk_images(finalCombinedDpMem4d, transfered_last_seven_mem_output)
    combined16_clahe = combine_4d_itk_images(finalCombinedClahe4d, transfered_last_seven_clahe_output)

    # --- Final cyclic PCA over ALL frames (16) ---
    final_dir = os.path.join(savedir, "final_cyclic")
    os.makedirs(final_dir, exist_ok=True)
    final_res = orchestrate_registration_workflow(
        image4d    = combined16_img,
        clahe4d     = combined16_clahe,
        dp_rbc4d    = combined16_rbc,
        dp_mem4d    = combined16_mem,
        savedir     = final_dir,
        target_weights = [35],          
        index_sets      = [ set(range(16)) ],
        cyclic       = True,
        time_threshold =  time_threshold        
    )

        # --- Package and return everything ---
    return {
        "resultImage4d":   final_res["resultImage4d"],
        "resultMask4d":    final_res["resultMask4d"],
        "resultDpRbc4d":   final_res["resultDpRbc4d"],
        "resultDpMem4d":   final_res["resultDpMem4d"],
        "resultClahe4d":   final_res["resultClahe4d"],
        # all transforms in order: [T_block1, T_block2, T_cyclic14or16]
        "transformations": final_res["transformations"],
        "run_flags": run_flags
    }


# def step_groupwise_registration(image, dp_rbc, dp_mem, savedir):
#     """
#     Perform the two-block (7+7) groupwise registration on gas+DP,
#     merge into 14 frames, *then* insert two ‘middle’ frames to get 16,
#     and finally do a cyclic run on all 16.
#     """
#     # --- your existing 7+7 split & registration unchanged ---
#     index_ranges = [(0, 7), (image.GetSize()[3] - 7, image.GetSize()[3])]
#     processed_images_first_seven = []
#     processed_rbcs_first_seven   = []
#     processed_mems_first_seven   = []
#     processed_images_last_seven  = []
#     processed_rbcs_last_seven    = []
#     processed_mems_last_seven    = []

#     for start, end in index_ranges:
#         block_dir = os.path.join(savedir, str(start))
#         os.makedirs(block_dir, exist_ok=True)

#         if start > 0:
#             img_blk = reverse_last_seven_images(image)
#             rbc_blk = reverse_last_seven_images(dp_rbc)
#             mem_blk = reverse_last_seven_images(dp_mem)
#         else:
#             img_blk, rbc_blk, mem_blk = image, dp_rbc, dp_mem

#         # Stage A: cut 4 bins
#         img4 = process_images(img_blk, (0,4))
#         rbc4 = process_images(rbc_blk, (0,4))
#         mem4 = process_images(mem_blk, (0,4))

#         res4 = orchestrate_registration_workflow(
#             img4, rbc4, mem4,
#             block_dir,
#             target_weights=[7],
#             index_sets=[{0,1,2,3}],
#             cyclic=False
#         )

#         # Stage B: update 4–7 with average
#         avg_img = average_4d_image_stack(res4["resultImage4d"])
#         avg_rbc = average_4d_image_stack(res4["resultDpRbc4d"])
#         avg_mem = average_4d_image_stack(res4["resultDpMem4d"])

#         up_img = update_images(img_blk, (4,7), avg_img)
#         up_rbc = update_images(rbc_blk, (4,7), avg_rbc)
#         up_mem = update_images(mem_blk, (4,7), avg_mem)

#         res43 = orchestrate_registration_workflow(
#             up_img, up_rbc, up_mem,
#             block_dir,
#             target_weights=[7],
#             index_sets=[{0,1,2,3}],
#             cyclic=False
#         )

#         # Stage C: warp back & combine into 7
#         Ts = res43["transformations"]

#         t_img = apply_transformations_in_sequence(res4["resultImage4d"], Ts, block_dir, TransformationPosition.BEGINNING)
#         combined_img = combine_transformed_and_registered(
#             t_img, res43["resultImage4d"],
#             index_transfer_range=(0,4), index_register_range=(1,4)
#         )

#         t_rbc = apply_transformations_in_sequence(res4["resultDpRbc4d"], Ts, block_dir, TransformationPosition.BEGINNING)
#         combined_rbc = combine_transformed_and_registered(
#             t_rbc, res43["resultDpRbc4d"],
#             index_transfer_range=(0,4), index_register_range=(1,4)
#         )

#         t_mem = apply_transformations_in_sequence(res4["resultDpMem4d"], Ts, block_dir, TransformationPosition.BEGINNING)
#         combined_mem = combine_transformed_and_registered(
#             t_mem, res43["resultDpMem4d"],
#             index_transfer_range=(0,4), index_register_range=(1,4)
#         )

#         final7_img = process_images(combined_img, (0,7))
#         final7_rbc = process_images(combined_rbc, (0,7))
#         final7_mem = process_images(combined_mem, (0,7))

#         # Stage D: 7-bin PCA
#         res7 = orchestrate_registration_workflow(
#             final7_img, final7_rbc, final7_mem,
#             block_dir,
#             target_weights=[14],
#             index_sets=[set(range(7))],
#             cyclic=False
#         )

#         # un‐reverse last block
#         if start > 0:
#             res7["resultImage4d"] = resort_to_original_order(res7["resultImage4d"])
#             res7["resultDpRbc4d"] = resort_to_original_order(res7["resultDpRbc4d"])
#             res7["resultDpMem4d"] = resort_to_original_order(res7["resultDpMem4d"])
#             processed_images_last_seven.append(res7["resultImage4d"])
#             processed_rbcs_last_seven.append(res7["resultDpRbc4d"])
#             processed_mems_last_seven.append(res7["resultDpMem4d"])
#         else:
#             processed_images_first_seven.append(res7["resultImage4d"])
#             processed_rbcs_first_seven.append(res7["resultDpRbc4d"])
#             processed_mems_first_seven.append(res7["resultDpMem4d"])

#     # --- now merge your two 7-frame blocks into a 14-frame series ---
#     combined_image_4d = combine_4d_images(processed_images_first_seven, processed_images_last_seven)
#     combined_rbc_4d = combine_4d_images(processed_rbcs_first_seven,   processed_rbcs_last_seven)
#     combined_mem_4d = combine_4d_images(processed_mems_first_seven,   processed_mems_last_seven)

#     # (optional) inspect 14-frame mask
#     combined_mask_4d = create_mask_from_4d_image(combined_image_4d, threshold, time_threshold)
#     sitk.WriteImage(combined_mask_4d, os.path.join(savedir, "ns_combined14_mask.nii"))

#     sitk.WriteImage(combined_image_4d, os.path.join(savedir, "ns_combined14_image.nii"))
#     sitk.WriteImage(combined_rbc_4d, os.path.join(savedir, "ns_combined14_rbc.nii"))
#     sitk.WriteImage(combined_mem_4d, os.path.join(savedir, "ns_combined14_mem.nii"))

#     registration_fourteen_results = orchestrate_registration_workflow(
#         combined_image_4d, combined_rbc_4d, combined_mem_4d,
#         block_dir,
#         target_weights=[30],
#         index_sets=[set(range(14))],
#         cyclic=True
#     )

#     first_seven_image_4d = extract_4d_subregion(registration_fourteen_results["resultImage4d"], 0, 7)
#     last_seven_image_4d = extract_4d_subregion(registration_fourteen_results["resultImage4d"], combined_image_4d.GetSize()[3] - 7, combined_image_4d.GetSize()[3])

#     first_seven_rbc_4d = extract_4d_subregion(registration_fourteen_results["resultDpRbc4d"], 0, 7)
#     last_seven_rbc_4d = extract_4d_subregion(registration_fourteen_results["resultDpRbc4d"], combined_rbc_4d.GetSize()[3] - 7, combined_rbc_4d.GetSize()[3])

#     first_seven_mem_4d = extract_4d_subregion(registration_fourteen_results["resultDpMem4d"], 0, 7)
#     last_seven_mem_4d = extract_4d_subregion(registration_fourteen_results["resultDpMem4d"], combined_rbc_4d.GetSize()[3] - 7, combined_rbc_4d.GetSize()[3])

#         # Optionally, save the separated first and last seven images
#     sitk.WriteImage(first_seven_image_4d, os.path.join(savedir, "ns_first_seven_image_groupwise.nii"))
#     sitk.WriteImage(last_seven_image_4d, os.path.join(savedir, "ns_last_seven_image_groupwise.nii"))
    
#     sitk.WriteImage(first_seven_rbc_4d, os.path.join(savedir, "ns_first_seven_rbc_groupwise.nii"))
#     sitk.WriteImage(last_seven_rbc_4d, os.path.join(savedir, "ns_last_seven_rbc_groupwise.nii"))
    
#     sitk.WriteImage(first_seven_mem_4d, os.path.join(savedir, "ns_first_seven_mem_groupwise.nii"))
#     sitk.WriteImage(last_seven_mem_4d, os.path.join(savedir, "ns_last_seven_mem_groupwise.nii"))    


#     average_image_first = average_4d_image_stack(first_seven_image_4d)
#     average_rbc_first = average_4d_image_stack(first_seven_rbc_4d)
#     average_mem_first = average_4d_image_stack(first_seven_mem_4d)
    
#     average_image_last = average_4d_image_stack(last_seven_image_4d)
#     average_rbc_last = average_4d_image_stack(last_seven_rbc_4d)
#     average_mem_last = average_4d_image_stack(last_seven_mem_4d)

#     sitk.WriteImage(average_image_first, os.path.join(savedir, "ns_first_seven_image_average_groupwise.nii"))
#     sitk.WriteImage(average_image_last, os.path.join(savedir, "ns_last_seven_image_average_groupwise.nii"))

#     UpdatedLastImage4d = update_images_final(image, index_range=(7, 9), average_image_first = average_image_first, average_image_last = average_image_last)
#     UpdatedLastRbc4d = update_images_final(dp_rbc, index_range=(7, 9), average_image_first = average_rbc_first, average_image_last = average_rbc_last)
#     UpdatedLastMem4d = update_images_final(dp_mem, index_range=(7, 9), average_image_first = average_mem_first, average_image_last = average_mem_last)

#     registration_fourteen_two_results = orchestrate_registration_workflow(
#         UpdatedLastImage4d, UpdatedLastRbc4d, UpdatedLastMem4d,
#         block_dir,
#         target_weights=[7],
#         index_sets=[set(range(4))],
#         cyclic=True
#     )

#     transfered_first_seven_image_output = apply_transformations_in_sequence(first_seven_image_4d, registration_fourteen_two_results["transformations"], savedir, TransformationPosition.BEGINNING)
#     transfered_first_seven_rbc_output = apply_transformations_in_sequence(first_seven_rbc_4d, registration_fourteen_two_results["transformations"], savedir, TransformationPosition.BEGINNING)
#     transfered_first_seven_mem_output = apply_transformations_in_sequence(first_seven_mem_4d, registration_fourteen_two_results["transformations"], savedir, TransformationPosition.BEGINNING)
    
#     transfered_last_seven_image_output = apply_transformations_in_sequence(last_seven_image_4d, registration_fourteen_two_results["transformations"], savedir, TransformationPosition.END)
#     transfered_last_seven_rbc_output = apply_transformations_in_sequence(last_seven_rbc_4d, registration_fourteen_two_results["transformations"], savedir, TransformationPosition.END)
#     transfered_last_seven_mem_output = apply_transformations_in_sequence(last_seven_mem_4d, registration_fourteen_two_results["transformations"], savedir, TransformationPosition.END)

#     sitk.WriteImage(transfered_first_seven_image_output, os.path.join(savedir, "ns_first_seven_image_transformed_groupwise.nii"))
#     sitk.WriteImage(transfered_first_seven_rbc_output, os.path.join(savedir, "ns_first_seven_rbc_transformed_groupwise.nii"))
#     sitk.WriteImage(transfered_first_seven_mem_output, os.path.join(savedir, "ns_first_seven_mem_transformed_groupwise.nii"))

#     sitk.WriteImage(transfered_last_seven_image_output, os.path.join(savedir, "ns_last_seven_image_transformed_groupwise.nii"))
#     sitk.WriteImage(transfered_last_seven_rbc_output, os.path.join(savedir, "ns_last_seven_rbc_transformed_groupwise.nii"))
#     sitk.WriteImage(transfered_last_seven_mem_output, os.path.join(savedir, "ns_last_seven_mem_transformed_groupwise.nii"))

#     finalCombinedImage4d = combine_transformed_and_registered(transfered_first_seven_image_output, registration_fourteen_two_results["resultImage4d"], index_transfer_range=(0, 7), index_register_range=(1, 3))
#     finalCombinedRbc4d = combine_transformed_and_registered(transfered_first_seven_rbc_output, registration_fourteen_two_results["resultDpRbc4d"], index_transfer_range=(0, 7), index_register_range=(1, 3))
#     finalCombinedMem4d = combine_transformed_and_registered(transfered_first_seven_mem_output, registration_fourteen_two_results["resultDpMem4d"], index_transfer_range=(0, 7), index_register_range=(1, 3))

#     final_combined_image_4d = combine_4d_itk_images(finalCombinedImage4d, transfered_last_seven_image_output)
#     final_combined_rbc_4d = combine_4d_itk_images(finalCombinedRbc4d, transfered_last_seven_rbc_output)
#     final_combined_mem_4d = combine_4d_itk_images(finalCombinedMem4d, transfered_last_seven_mem_output)

#     # --- final cyclic PCA on all 16 frames ---
#     final_registration_results = orchestrate_registration_workflow(
#         final_combined_image_4d, final_combined_rbc_4d, final_combined_mem_4d,
#         savedir,
#         target_weights=[35],
#         index_sets=[set(range(16))],
#         cyclic=True
#     )

#     sitk.WriteImage(final_registration_results["resultImage4d"], os.path.join(savedir, "final_image_groupwise.nii"))
#     sitk.WriteImage(final_registration_results["resultDpRbc4d"], os.path.join(savedir, "final_rbc_groupwise.nii"))
#     sitk.WriteImage(final_registration_results["resultDpMem4d"], os.path.join(savedir, "final_mem_groupwise.nii"))

#     # Return the separated first and last seven images as needed
#     return final_registration_results
    

    # --- HERE: build the *16-frame* block by inserting two averages at positions 7 & 8 ---
    # extract first 7 & last 7
    # first7 = extract_4d_subregion(finalCombinedImage4d, 0, 7)
    # last7  = extract_4d_subregion(finalCombinedImage4d, 7, 14)

    # # average each
    # avg_first_img = average_4d_image_stack(first7)
    # avg_last_img  = average_4d_image_stack(last7)
    # avg_first_rbc = average_4d_image_stack(extract_4d_subregion(combined14_rbc, 0, 7))
    # avg_last_rbc  = average_4d_image_stack(extract_4d_subregion(combined14_rbc, 7,14))
    # avg_first_mem = average_4d_image_stack(extract_4d_subregion(combined14_mem, 0, 7))
    # avg_last_mem  = average_4d_image_stack(extract_4d_subregion(combined14_mem, 7,14))

    # # insert them:
    # combined16_img = update_images_final(combined14_img, (7,9), avg_first_img, avg_last_img)
    # combined16_rbc = update_images_final(combined14_rbc, (7,9), avg_first_rbc, avg_last_rbc)
    # combined16_mem = update_images_final(combined14_mem, (7,9), avg_first_mem, avg_last_mem)

    # # save your full 16-frame series & mask
    # sitk.WriteImage(combined16_img, os.path.join(savedir, "ns_combined16_image.nii"))
    # sitk.WriteImage(combined16_rbc, os.path.join(savedir, "ns_combined16_rbc.nii"))
    # sitk.WriteImage(combined16_mem, os.path.join(savedir, "ns_combined16_mem.nii"))

    # combined16_mask = create_mask_from_4d_image(combined16_img, threshold, time_threshold)
    # sitk.WriteImage(combined16_mask, os.path.join(savedir, "ns_combined16_mask.nii"))

    # # --- final cyclic PCA on all 16 frames ---
    # res16 = orchestrate_registration_workflow(
    #     combined16_img, combined16_rbc, combined16_mem,
    #     savedir,
    #     target_weights=[35],
    #     index_sets=[set(range(16))],
    #     cyclic=True
    # )

    # return res16

def register_EI_ants_hoom(image3D, clahe3D, dp_rbc3D, dp_mem3D, refno, savedir):
    """
    Pairwise SyN‐CC ANTs on each bin, applying the resulting transforms
    to the original image, the CLAHE image, and both DP channels.
    Returns four parallel lists of SimpleITK 3D images.
    """
    # 1) Convert all four streams to ANTs images
    ants_imgs   = [ants.from_numpy(sitk.GetArrayFromImage(i))    for i in image3D]
    ants_clh    = [ants.from_numpy(sitk.GetArrayFromImage(c))    for c in clahe3D]
    ants_rbcs   = [ants.from_numpy(sitk.GetArrayFromImage(r))    for r in dp_rbc3D]
    ants_mems   = [ants.from_numpy(sitk.GetArrayFromImage(m))    for m in dp_mem3D]

    # Prepare output containers
    resultImg3d   = []
    resultClh3d   = []
    resultRbc3d   = []
    resultMem3d   = []

    # 2) Loop over each bin
    for idx in range(len(ants_imgs)):
        fixed  = ants_clh[refno]
        moving = ants_clh[idx]

        # SyN with CC on CLAHE
        reg = ants.registration(fixed=fixed,
                                 moving=moving,
                                 type_of_transform='SyNCC')

        # apply same transforms to all four streams
        fwd = reg['fwdtransforms']
        out_img = ants.apply_transforms(fixed=fixed, moving=ants_imgs[idx],  transformlist=fwd)
        out_clh = ants.apply_transforms(fixed=fixed, moving=ants_clh[idx],  transformlist=fwd)
        out_rbc = ants.apply_transforms(fixed=fixed, moving=ants_rbcs[idx], transformlist=fwd)
        out_mem = ants.apply_transforms(fixed=fixed, moving=ants_mems[idx], transformlist=fwd)

        # convert back to SITK and copy geometry
        def to_sitk(a, src):
            im = sitk.GetImageFromArray(a.numpy())
            im.SetSpacing(src.GetSpacing())
            im.SetOrigin( src.GetOrigin())
            im.SetDirection(src.GetDirection())
            return im

        resultImg3d.append(to_sitk(out_img, image3D[idx]))
        resultClh3d.append(to_sitk(out_clh, clahe3D[idx]))
        resultRbc3d.append(to_sitk(out_rbc, dp_rbc3D[idx]))
        resultMem3d.append(to_sitk(out_mem, dp_mem3D[idx]))

    return resultImg3d, resultClh3d, resultRbc3d, resultMem3d    

def register_EI_ants(image3D, dp_rbc3D, dp_mem3D, refno, savedir):
    """
    Pairwise SyN-CC ANTs on each time-bin, applying the resulting transforms
    to the original image, the CLAHE image, and both DP channels.
    Returns four parallel lists of SimpleITK 3D images.
    """
    # 1) Convert all four SimpleITK→ANTs
    ants_imgs  = [ants.from_numpy(sitk.GetArrayFromImage(im))  for im in image3D]
    ants_rbcs  = [ants.from_numpy(sitk.GetArrayFromImage(r))   for r  in dp_rbc3D]
    ants_mems  = [ants.from_numpy(sitk.GetArrayFromImage(m))   for m  in dp_mem3D]

    # 2) Resample fixed to “big_size” grid (for faster SyN)
    big_size = 3  # tune as you like
    new_vox  = [1.0/big_size]*3
    original_spacing = ants_imgs[0].spacing

    fixed_img_rs = ants.resample_image(
        ants_imgs[refno], resample_params=new_vox, use_voxels=False, interp_type=0
    )

    out_imgs, out_rbcs, out_mems = [], [], []

    for i in range(len(ants_imgs)):
        # 3) resample moving CLAHE & image

        mov_img_rs = ants.resample_image(
            ants_imgs[i], resample_params=new_vox, use_voxels=False, interp_type=0
        )

        # 4) SyNCC on CLAHE
        reg = ants.registration(
            fixed=fixed_img_rs,
            moving=mov_img_rs,
            # type_of_transform='SyNCC'
            type_of_transform='SyN'
        )
        fwd = reg['fwdtransforms']

        # 5) apply transforms to all four streams
        ant_out_img = ants.apply_transforms(
            fixed=fixed_img_rs, moving=mov_img_rs, transformlist=fwd
        )
        ant_out_rbc = ants.apply_transforms(
            fixed=fixed_img_rs, moving=ants_rbcs[i], transformlist=fwd
        )
        ant_out_mem = ants.apply_transforms(
            fixed=fixed_img_rs, moving=ants_mems[i], transformlist=fwd
        )

        # 6) resample everything back to original spacing
        img_back = ants.resample_image(
            ant_out_img, resample_params=original_spacing, use_voxels=False, interp_type=0
        )
        rbc_back = ants.resample_image(
            ant_out_rbc, resample_params=original_spacing, use_voxels=False, interp_type=0
        )
        mem_back = ants.resample_image(
            ant_out_mem, resample_params=original_spacing, use_voxels=False, interp_type=0
        )

        # helper to convert back to SITK and copy geometry
        def to_sitk(a, src):
            im = sitk.GetImageFromArray(a.numpy())
            im.SetSpacing(src.GetSpacing())
            im.SetOrigin( src.GetOrigin())
            im.SetDirection(src.GetDirection())
            return im

        out_imgs.append( to_sitk(img_back, image3D[i]) )
        out_rbcs.append( to_sitk(rbc_back, dp_rbc3D[i]) )
        out_mems.append( to_sitk(mem_back, dp_mem3D[i]) )

    return out_imgs, out_rbcs, out_mems


def main():
    # Read the initial 
    image, dp_rbc, dp_mem, clahe = read_files(input_files)
   
    s0  = time.time()
    ################## CURRRENT REGISTRATION #########################################
    final_registration_results = step_groupwise_registration(
        image,
        dp_rbc,
        dp_mem,
        clahe,
        save_dir
    )
    #################################################################################
    e0 = time.time()

    ################## CURRRENT REGISTRATION #############
    # unpack all four 4D results
    image4D   = final_registration_results["resultImage4d"]
    dp_rbc4D  = final_registration_results["resultDpRbc4d"]
    dp_mem4D  = final_registration_results["resultDpMem4d"]

    # image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2024-11-17_005DS/reg/5000/ns_combined_image_groupwise.nii')
    # clahe4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2024-11-17_005DS/reg/5000/ns_combined_clahe_groupwise.nii')
    # dp_rbc4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2024-11-17_005DS/reg/5000/ns_combined_rbc_groupwise.nii')
    # dp_mem4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2024-11-17_005DS/reg/5000/ns_combined_mem_groupwise.nii')

    print(f"Groupwise took {(e0 - s0)/60:.2f} minutes")

        # First ANTs pass: “original” → registered_orig.nii
    s6 = time.time()
    image3Dn  = extract_image_3d(image4D)
    dp_rbc3Dn = extract_image_3d(dp_rbc4D)
    dp_mem3Dn = extract_image_3d(dp_mem4D)

    print('ANTs in progress…')
    image3Dn, dp_rbc3Dn, dp_mem3Dn = register_EI_ants(
        image3Dn, dp_rbc3Dn, dp_mem3Dn, refno, save_dir
    )
    image4Dn = join_image3d_ants(image3Dn)
    rbc4Dn = join_image3d_ants(dp_rbc3Dn)
    mem4Dn = join_image3d_ants(dp_mem3Dn)

    sitk.WriteImage(image4Dn, os.path.join(save_dir, "gas_registered_orig.nii"))
    sitk.WriteImage(rbc4Dn, os.path.join(save_dir, "rbc_registered_orig.nii"))
    sitk.WriteImage(mem4Dn, os.path.join(save_dir, "mem_registered_orig.nii"))

    e6 = time.time()
    print('Last EI Registration took', round((e6 - s6)/60, 2), 'minutes.')

if __name__ == "__main__":
    main()  