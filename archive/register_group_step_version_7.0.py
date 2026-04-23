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

# Global configuration
# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS/'

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/rec/'
# endinhale = 8  
# cutoff = 0.04  # Cutoff value for mask creation

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-01-31_000LL/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-01-31_000LL/rec/'
# endinhale = 7
# threshold = 0.05  # Cutoff value for mask creation
# time_threshold = 5

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/rec/'
# endinhale = 6
# refno = 6
# threshold = 0.05  # Cutoff value for mask creation
# time_threshold = 5 # Cutoff value for mask sum creation

# target_weights = [7, 4]
# index_sets = [{0, 1, 2, 14, 15}, {3, 12, 13}]  # Sets of bins to enhance together

# index_sets = None
# target_weights = None

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-10-10_002BB/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-10-10_002BB/rec/'
# endinhale = 7
# refno = 7
# threshold = 0.06  # Cutoff value for mask creation
# time_threshold = 5 # Cutoff value for mask sum creation
# index_sets = None
# target_weights = None

save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/sevencmbestc/'
input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/rec/sevenor/'
endinhale = 6
refno = 6
threshold = 0.03  # Cutoff value for mask creation
time_threshold = 1 # Cutoff value for mask sum creation

# For the sets of 4 (A: 1/2/3/4,  B: A/5/6/7, C: 13/14/15/16, D: 10/11/12/B,  B/8/9/D) 
target_weights = [7]
index_sets = [{0, 1, 2, 3}]  # Sets of bins to enhance together

# For the sets of 7 (1/2/3/4/5/6/7 & 10/11/12/13/14/15/16/17)
target_weights = [14]
index_sets = [{0, 1, 2, 3, 4, 5, 6}]  # Sets of bins to enhance together

# For the set of 16
# target_weights = [30]
# index_sets = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}]  # Sets of bins to enhance together

# target_weights = [16]
# index_sets = [{0, 1, 2, 3, 4, 5a, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}]  # Sets of bins to enhance together
# target_weights = [5, 4, 3, 3, 3]
# index_sets = [{0, 1, 2, 15}, {3, 13, 14}, {4, 12} , {5, 10}, {6, 8}]  # Sets of bins to enhance together

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-04-11_000KR/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-04-11_000KR/rec/'
# endinhale = 7
# refno = 7
# threshold = 0.05  # Cutoff value for mask creation
# time_threshold = 5 # Cutoff value for mask sum creation
# index_sets = None
# target_weights = None

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-04-14_000KR/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-04-14_000KR/rec/'
# endinhale = 7
# refno = 7
# threshold = 0.05  # Cutoff value for mask creation
# time_threshold = 5 # Cutoff value for mask sum creation
# index_sets = None
# target_weights = None

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-10-10_002BB/reg'

# Find the input files
def find_file_with_prefix(directory, prefix):
    for filename in os.listdir(directory):  
        if filename.startswith(prefix) and filename.endswith('.mat'):
            return os.path.join(directory, filename)
    raise ValueError(f"No file starting with '{prefix}' found in {directory}")

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
#     clahe_file = find_file_with_prefix(directory, 'cspace')
#     pimage_file = find_file_with_prefix(directory, 'pspace')

#     # Process images
#     images = process_mat_file(image_file, 'image')

#     # Process CLAHE images
#     clahe_images = process_mat_file(clahe_file, 'clahe')

#     # Process CLAHE images
#     pimages = process_mat_file(pimage_file, 'pimage')

#     # sitk.WriteImage(images, os.path.join("/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/reg/","image_init.nii"))

#     return images, clahe_images, pimages

def read_files(directory):
    def process_mat_file(file_path, key):
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.mat':
            mat_content = scipy.io.loadmat(file_path)
            data_4d = mat_content.get(key)

            if data_4d is None:
                raise ValueError(f"Key '{key}' not found in MATLAB file")

            vector_of_images = sitk.VectorOfImage()
            for t in range(data_4d.shape[3]):  # Change here to loop over the last dimension
                data_3d = data_4d[:, :, :, t]  # Change here to extract the 3D image at each time point
                data_3d_transposed = np.transpose(data_3d, (2, 1, 0))  # Transpose to (z, y, x)
    
                sitk_image = sitk.GetImageFromArray(data_3d_transposed)
                vector_of_images.push_back(sitk_image)

            return sitk.JoinSeries(vector_of_images)
        else:
            raise ValueError("Unsupported file format")

    # Find files with specific prefixes
    image_file = find_file_with_prefix(directory, 'rspace')
    clahe_file = find_file_with_prefix(directory, 'cspace')

    # Process images
    images = process_mat_file(image_file, 'image')

    # Process CLAHE images
    clahe_images = process_mat_file(clahe_file, 'clahe')
    # sitk.WriteImage(images, os.path.join("/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/reg/","image_init.nii"))

    return images, clahe_images


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

# combining 3D+t images to generate 4D ITK images
def join_image3d(image):
    vector_of_images = sitk.VectorOfImage()
    for x in range(16):
        vector_of_images.push_back(image[x])
    return sitk.JoinSeries(vector_of_images)

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
def create_mask_from_4d_image(image4d, threshold=0.05, time_threshold=5):
    # Convert 4D SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(image4d)

    # Apply initial threshold to create a 4D mask
    mask_4d = np.copy(image_array)
    mask_4d[mask_4d < threshold] = 0
    mask_4d[mask_4d >= threshold] = 1

    # Sum across time dimension and apply time threshold
    summed_image = np.sum(mask_4d, axis=0)
    maskf = np.zeros_like(summed_image)
    maskf[summed_image >= time_threshold] = 1
    maskf[summed_image < time_threshold] = 0

    # Connected component analysis
    labeled_array, num_features = label(maskf)
    sizes = np.bincount(labeled_array.ravel())
    max_label = sizes[1:].argmax() + 1  # +1 because we ignore the background label (0)

    # Isolating the largest component
    maskf[labeled_array != max_label] = 0

    # Replicate the 2D mask across all time points
    final_mask = np.repeat(maskf[np.newaxis, :, :], image_array.shape[0], axis=0)

    # Initialize a vector to store each 3D mask slice
    mask_vector = sitk.VectorOfImage()

    # Iterate over each time point to create a 3D mask slice
    for i in range(final_mask.shape[0]):  # Iterate over the first dimension (time)
        mask_slice = final_mask[i, :, :, :]  # Extract the 3D mask for the current time point
        mask_slice_sitk = sitk.GetImageFromArray(mask_slice.astype(np.uint8))
        mask_slice_sitk.SetSpacing(image4d.GetSpacing()[0:3])  # Set spacing for 3D
        mask_slice_sitk.SetOrigin(image4d.GetOrigin())
        mask_vector.push_back(mask_slice_sitk)

    # Combine the 3D slices into a 4D image
    mask_4d_sitk = sitk.JoinSeries(mask_vector)
    mask_4d_sitk.SetSpacing((image4d.GetSpacing()[-1],) + image4d.GetSpacing()[0:3])  # Set 4D spacing

    return mask_4d_sitk


def create_bm3d_from_4d_image(image4d, noise_var = 0.02):
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

def create_clahe_from_4d_image(image4d, mask4d, kernel_size=(16, 16), clip_limit=0.45, nbins=512):
    # Ensure the input image and mask have the same size
    if image4d.GetSize() != mask4d.GetSize():
        raise ValueError("Image and mask must have the same size.")

    # Convert SimpleITK images to numpy arrays
    image_array = sitk.GetArrayFromImage(image4d)  # Shape: [t, z, y, x]
    mask_array = sitk.GetArrayFromImage(mask4d)    # Shape: [t, z, y, x]

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
            mask_slice = mask_array[t, z, :, :]

            # Apply CLAHE only to the masked regions
            enhanced_slice = np.where(mask_slice, exposure.equalize_adapthist(slice_2d, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins), slice_2d)

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
def register_stepgroupwise(image, clahe, pimage, savedir, enhanced, target_weights, claheinternal, noisereduction):
    sitk.WriteImage(image, os.path.join(savedir,"image_init.nii"))
    
    # Creating Mask from Images
    mask = create_mask_from_4d_image(image, threshold, time_threshold)
    sitk.WriteImage(mask, os.path.join(savedir,"mask_init.nii"))

    # Check if index_sets is provided and not empty
    if noisereduction:
        print('Noise Reduction')   
        nreduced_image = create_bm3d_from_4d_image(image)
        print('Done!')   
        if enhanced:
            print('Enhanced Image')
            # Enhance selected bins and create CLAHE from the enhanced image
            enhanced_image4d = enhance_selected_bins(nreduced_image, index_sets, target_weights)
            sitk.WriteImage(enhanced_image4d, os.path.join(savedir, "enhanced_init.nii"))
            # Do or no do CLAHE 
            if claheinternal:
                print('Internal CLAHE')
                # clahe = create_bm3d_from_4d_image(enhanced_image4d, mask)
                clahe = create_clahe_from_4d_image(enhanced_image4d, mask)
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = enhanced_image4d            
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))
        else:
            print('bypassed Enhanced; use no Enhanced')
            if claheinternal:
                print('Internal CLAHE')                
                clahe = create_clahe_from_4d_image(nreduced_image, mask)
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = nreduced_image          
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))                
    else:    
        if enhanced:
            print('Enhanced Image')
            # Enhance selected bins and create CLAHE from the enhanced image
            enhanced_image4d = enhance_selected_bins(image, index_sets, target_weights)
            sitk.WriteImage(enhanced_image4d, os.path.join(savedir, "enhanced_init.nii"))
            # Do or no do CLAHE 
            if claheinternal:
                print('Internal CLAHE')
                # clahe = create_bm3d_from_4d_image(enhanced_image4d, mask)
                clahe = create_clahe_from_4d_image(enhanced_image4d, mask)
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = enhanced_image4d            
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))
        else:
            print('bypassed Enhanced; use no Enhanced')
            if claheinternal:
                print('Internal CLAHE')
                # clahe = create_bm3d_from_4d_image(image)
                # clahe = create_clahe_from_4d_image(image, mask)
                clahe = clahe
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = image          
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))    

    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(clahe)
    elastixImageFilter.SetMovingImage(clahe)

    elastixImageFilter.SetFixedMask(mask)
    elastixImageFilter.SetMovingMask(mask)

    # Define Groupwise Methodology
    parameterMap = sitk.GetDefaultParameterMap('groupwise')
    # parameterMap['UseCyclicTransform'] = ['true']
    parameterMap['UseCyclicTransform'] = ['flase']

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
    parameterMap['ASGDParameterEstimationMethod'] = ['AdaptiveStocDisplacementDistributionhasticGradientDescent']
    # parameterMap['MaximumNumberOfIterations'] = ['10000', '20000', '30000', '40000']
    # parameterMap['MaximumNumberOfIterations'] = ['20000', '40000', '60000', '80000']
    parameterMap['MaximumNumberOfIterations'] = ['10000']
    # parameterMap['MaximumNumberOfIterations'] = ['1']

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

    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(image)
    transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformix.Execute()
    # resultImage4d = elastixImageFilter.GetResultImage()
    resultImage4d = transformix.GetResultImage()

    vectorOfMasks = sitk.VectorOfImage()
    
    resultMask4d = create_mask_from_4d_image(resultImage4d, threshold, time_threshold)
    #sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_groupwise.nii"))

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(clahe)
    transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformix.Execute()
    resultClahe4d = transformix.GetResultImage()
    
    # Initialize a list to hold the transformed images
    transformed_images = []

    # Initialize a list to store the reordered 4D images for each iteration
    reordered_4d_images = []

    # Loop over each index in the 4D stack
    for i in range(4):
        # Extract the 3D image at position i
        current_image = sitk.Extract(pimage, pimage.GetSize()[0:3] + (0,), (0, 0, 0, i))
        
        # Initialize a list to store the reordered 3D images
        reordered_images = [current_image]
        
        # Loop over the remaining indices and extract the 3D images, excluding the current one
        for j in range(4):
            if j != i:
                other_image = sitk.Extract(pimage, pimage.GetSize()[0:3] + (0,), (0, 0, 0, j))
                reordered_images.append(other_image)
        
        # Combine the reordered 3D images into a 4D image
        reordered_image_4d = sitk.JoinSeries(reordered_images)          

        # Apply the transformation to the reordered 4D image
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(reordered_image_4d)
        transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformix.Execute()

        # Extract the transformed image and add it to the list
        # transformed_image = transformix.GetResultImage()
        transformed_image = sitk.Extract(transformix.GetResultImage(), reordered_image_4d.GetSize()[0:3] + (0,), (0, 0, 0, 0))
        # sitk.WriteImage(transformed_image, os.path.join(savedir, str(i), "pimage_init.nii"))   
        transformed_images.append(transformed_image)

    # Combine the transformed images back into a single 4D image
    resultPimage4d = sitk.JoinSeries(transformed_images)

    # saving the temple file obtained from the population
    sitk.WriteImage(resultImage4d, os.path.join(savedir, "image_groupwise.nii"))
    sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_groupwise.nii"))
    sitk.WriteImage(resultClahe4d, os.path.join(savedir, "clahe_groupwise.nii"))
    sitk.WriteImage(resultPimage4d, os.path.join(savedir, "pimage_groupwise.nii"))
    
    return resultImage4d, resultMask4d, resultClahe4d, resultPimage4d


# groupwise 4D (3D+t) registration w/ PCA2 [Huiziga]
def register_groupwise(image, clahe, savedir, enhanced, target_weights, claheinternal, noisereduction):
    sitk.WriteImage(image, os.path.join(savedir,"image_init.nii"))
    
    # Creating Mask from Images
    mask = create_mask_from_4d_image(image, threshold, time_threshold)
    sitk.WriteImage(mask, os.path.join(savedir,"mask_init.nii"))

    # Check if index_sets is provided and not empty
    if noisereduction:
        print('Noise Reduction')   
        nreduced_image = create_bm3d_from_4d_image(image)
        print('Done!')   
        if enhanced:
            print('Enhanced Image')
            # Enhance selected bins and create CLAHE from the enhanced image
            enhanced_image4d = enhance_selected_bins(nreduced_image, index_sets, target_weights)
            sitk.WriteImage(enhanced_image4d, os.path.join(savedir, "enhanced_init.nii"))
            # Do or no do CLAHE 
            if claheinternal:
                print('Internal CLAHE')
                # clahe = create_bm3d_from_4d_image(enhanced_image4d, mask)
                clahe = create_clahe_from_4d_image(enhanced_image4d, mask)
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = enhanced_image4d            
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))
        else:
            print('bypassed Enhanced; use no Enhanced')
            if claheinternal:
                print('Internal CLAHE')                
                clahe = create_clahe_from_4d_image(nreduced_image, mask)
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = nreduced_image          
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))                
    else:    
        if enhanced:
            print('Enhanced Image')
            # Enhance selected bins and create CLAHE from the enhanced image
            enhanced_image4d = enhance_selected_bins(image, index_sets, target_weights)
            sitk.WriteImage(enhanced_image4d, os.path.join(savedir, "enhanced_init.nii"))
            # Do or no do CLAHE 
            if claheinternal:
                print('Internal CLAHE')
                # clahe = create_bm3d_from_4d_image(enhanced_image4d, mask)
                clahe = create_clahe_from_4d_image(enhanced_image4d, mask)
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = enhanced_image4d            
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))
        else:
            print('bypassed Enhanced; use no Enhanced')
            if claheinternal:
                print('Internal CLAHE')
                # clahe = create_bm3d_from_4d_image(image)
                # clahe = create_clahe_from_4d_image(image, mask)
                clahe = clahe
            else:            
                print('bypassed Internal CLAHE; use no CLAHE')
                clahe  = image          
            sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))     

    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(clahe)
    elastixImageFilter.SetMovingImage(clahe)

    elastixImageFilter.SetFixedMask(mask)
    elastixImageFilter.SetMovingMask(mask)

    # Define Groupwise Methodology
    parameterMap = sitk.GetDefaultParameterMap('groupwise')
    parameterMap['UseCyclicTransform'] = ['true']

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
    parameterMap['ASGDParameterEstimationMethod'] = ['AdaptiveStocDisplacementDistributionhasticGradientDescent']
    parameterMap['MaximumNumberOfIterations'] = ['10000', '20000', '30000', '40000']
    # parameterMap['MaximumNumberOfIterations'] = ['20000', '40000', '60000', '80000']
    # parameterMap['MaximumNumberOfIterations'] = ['10000']
    # parameterMap['MaximumNumberOfIterations'] = ['10']
    
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

    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(image)
    transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformix.Execute()
    # resultImage4d = elastixImageFilter.GetResultImage()
    resultImage4d = transformix.GetResultImage()

    vectorOfMasks = sitk.VectorOfImage()
    
    resultMask4d = create_mask_from_4d_image(resultImage4d, threshold, time_threshold)
    #sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_groupwise.nii"))

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(clahe)
    transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformix.Execute()
    resultClahe4d = transformix.GetResultImage()

    # saving the temple file obtained from the population
    sitk.WriteImage(resultImage4d, os.path.join(savedir, "image_groupwise.nii"))
    sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_groupwise.nii"))
    sitk.WriteImage(resultClahe4d, os.path.join(savedir, "clahe_groupwise.nii"))
    
    return resultImage4d, resultMask4d, resultClahe4d


# step-by-step 3-16 bins groupwise 4D (3D+t) registration w/ PCA2 [Hooman]
def register_stepwise(image, clahe, refno, savedir):
    # Create a mask from the 4D image
    mask = create_mask_from_4d_image(image, threshold, time_threshold)
    sitk.WriteImage(mask, os.path.join(save_dir,"mask_init.nii"))

    elastixImageFilter=sitk.ElastixImageFilter()

    # Define Groupwise Methodology
    parameterMap = sitk.GetDefaultParameterMap('groupwise')
    parameterMap['UseCyclicTransform'] = ['false']

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
    # parameterMap['Metric'] = ['SumOfPairwiseCorrelationCoefficientsMetric']
    parameterMap['Metric'] = ['PCAMetric2']
    # parameterMap['Metric'] = ['LinearGroupwiseMI']

    parameterMap['SubtractMean'] = ['true']
    parameterMap['MovingImageDerivativeScales'] = ['1','1','1','0']
    parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']
    
    # Optimizer Setting
    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['AutomaticParameterEstimation'] = ['true']
    parameterMap['ASGDParameterEstimationMethod'] = ['AdaptiveStocDisplacementDistributionhasticGradientDescent']
    parameterMap['MaximumNumberOfIterations'] = ['1000']
    # parameterMap['MaximumNumberOfIterations'] = ['1']
    
    # Pyramid Setting
    parameterMap['GridSpacingSchedule'] = ['4','3','2','1']
    parameterMap['ImagePyramidSchedule'] = ['8','8','8' ,'0','4','4','4','0','2','2','2','0','1','1','1','0']
    
    # Sampler Parameters
    parameterMap['NumberOfSpatialSamples'] = ['1024']
    parameterMap['NewSamplesEveryIteration'] = ['true']
    # parameterMap['ImageSampler'] = ['RandomCoordinate']
    parameterMap['ImageSampler'] = ['RandomSparseMask']
    parameterMap['CheckNumberOfSamples'] = ['true']

    # Mask Setting
    parameterMap['ErodeMask'] = ['false']
    parameterMap['ErodeFixedMask'] = ['false']
    parameterMap['ErodeMovingMask'] = ['false']

    # Output Setting
    parameterMap['DefaultPixelValue'] = ['0']
    parameterMap['WriteResultImage'] = ['true']
    parameterMap['ResultImagePixelType'] = ['float']

    elastixImageFilter.AddParameterMap(parameterMap)

    # Register each image to the next one
    n = 16  # The size of the circular buffer
    
    # Check if index_sets is provided and not empty
    if index_sets is not None and target_weights is not None:
        # Enhance selected bins and create CLAHE from the enhanced image
        enhanced_image4d = enhance_selected_bins(image, index_sets, target_weights)
        sitk.WriteImage(enhanced_image4d, os.path.join(savedir, "enhanced_init.nii"))
        clahe = create_clahe_from_4d_image(enhanced_image4d)
        sitk.WriteImage(clahe, os.path.join(save_dir, "clahe_init.nii"))
    else:
        # Use the input CLAHE image directly
        sitk.WriteImage(clahe, os.path.join(savedir, "clahe_init.nii"))

    for y in range(2,n):        
        x_range = n if y < n - 1 else n + 1

        for x in range(x_range):
            if y == n - 1 and x == n:
                parameterMap['UseCyclicTransform'] = ['true']

            m = y + 1

            indices_to_stack = [(x + i) % n for i in range(m)]
            print(indices_to_stack)

            # Extract and stack the 3D images for the current indices
            Image4d = sitk.JoinSeries([sitk.Extract(image, image.GetSize()[0:3] + (0,), (0, 0, 0, i)) for i in indices_to_stack])
            Clahe4d = sitk.JoinSeries([sitk.Extract(clahe, clahe.GetSize()[0:3] + (0,), (0, 0, 0, i)) for i in indices_to_stack])

            # Create a mask for the current 4D image
            updatedMask4d = sitk.VectorOfImage()
            for i in range(m):
                Mask3d = sitk.Extract(mask, mask.GetSize()[0:3] + (0,), (0, 0, 0, i))
                updatedMask4d.push_back(Mask3d)
            Mask4d = sitk.JoinSeries(updatedMask4d)                
            sitk.WriteImage(Mask4d, os.path.join(savedir, str(m), "mask_stepwise.nii"))

            elastixImageFilter.SetFixedImage(Clahe4d)
            elastixImageFilter.SetMovingImage(Clahe4d)

            elastixImageFilter.SetFixedMask(Mask4d)
            elastixImageFilter.SetMovingMask(Mask4d)

            elastixImageFilter.SetParameterMap(parameterMap)
            elastixImageFilter.Execute()

            transformix = sitk.TransformixImageFilter()
            transformix.SetMovingImage(Image4d)
            transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformix.Execute()
            # resultImage4d = elastixImageFilter.GetResultImage()
            resultImage4d = transformix.GetResultImage()
            
            # Extract the 3D images from the result
            resultImage3d = [sitk.Extract(resultImage4d, resultImage4d.GetSize()[0:3] + (0,), (0, 0, 0, i)) for i in range(m)]
    
            transformix = sitk.TransformixImageFilter()
            transformix.SetMovingImage(Clahe4d)
            transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformix.Execute()
            resultClahe4d = transformix.GetResultImage()    

            # Extract the 3D images from the result
            resultClahe3d = [sitk.Extract(resultClahe4d, resultClahe4d.GetSize()[0:3] + (0,), (0, 0, 0, i)) for i in range(m)]

            # Update the original 4D image stacks
            updatedImage4d = sitk.VectorOfImage()
            updatedClahe4d = sitk.VectorOfImage()

            # Store the size and type of the first image to ensure consistency
            firstImageSize = None
            firstImagePixelType = None

            for i in range(image.GetSize()[-1]):
                if i in indices_to_stack:
                    idx = indices_to_stack.index(i)
                    currentImage = resultImage3d[idx]
                    currentClahe = resultClahe3d[idx]
                else:
                    currentImage = sitk.Extract(image, image.GetSize()[0:3] + (0,), (0, 0, 0, i))
                    currentClahe = sitk.Extract(clahe, clahe.GetSize()[0:3] + (0,), (0, 0, 0, i))

                # Convert the pixel type to 32-bit float if necessary
                if currentImage.GetPixelID() != sitk.sitkFloat32:
                    currentImage = sitk.Cast(currentImage, sitk.sitkFloat32)
                if currentClahe.GetPixelID() != sitk.sitkFloat32:
                    currentClahe = sitk.Cast(currentClahe, sitk.sitkFloat32)

                updatedImage4d.push_back(currentImage)
                updatedClahe4d.push_back(currentClahe)

            # Join the updated images into a single 4D image
            finalUpdatedImage4d = sitk.JoinSeries(updatedImage4d)
            finalUpdatedClahe4d = sitk.JoinSeries(updatedClahe4d)    

            image = finalUpdatedImage4d
            mask = create_mask_from_4d_image(image, threshold, time_threshold)

            # Check if index_sets is provided and not empty
            if index_sets is not None and target_weights is not None:
                # Enhance selected bins and create CLAHE from the enhanced image
                enhanced_image4d = enhance_selected_bins(image, index_sets, target_weights)
                sitk.WriteImage(enhanced_image4d, os.path.join(savedir, "enhanced_init.nii"))
                clahe = create_clahe_from_4d_image(enhanced_image4d)
            else:
                # Use the input CLAHE image directly
                clahe = finalUpdatedClahe4d
                            
        sitk.WriteImage(resultImage4d, os.path.join(savedir, str(m), "image_stepwise.nii"))
        sitk.WriteImage(resultClahe4d, os.path.join(savedir, str(m), "clahe_stepwise.nii"))      
        
    # saving the file obtained from the population
    sitk.WriteImage(image, os.path.join(savedir, "image_stepwise.nii"))
    sitk.WriteImage(mask, os.path.join(savedir, "mask_stepwise.nii"))
    sitk.WriteImage(clahe, os.path.join(savedir, "clahe_stepwise.nii"))

    return image, mask, clahe

# pairwise neighbor to neighbor ANTs registration CC
def register_pairwise_ants(image3D, clahe3D, refno):
    
    ants_images = [ants.from_numpy(sitk.GetArrayFromImage(img)) for img in image3D]
    ants_clahe = [ants.from_numpy(sitk.GetArrayFromImage(clahe)) for clahe in clahe3D]

    resultImage3d = []
    resultClahe3d = []

    for x in range(16):
        if x != refno-1:
            fixed_image = ants_clahe[x+1] if x < refno else ants_clahe[x-1]
            moving_image = ants_clahe[x]
        
            # Perform registration using CLAHE images
            registered = ants.registration(fixed = fixed_image, moving = moving_image, type_of_transform = 'SyNCC', flow_sigma = 0 , aff_iterations = (5, 5, 5, 5), aff_smoothing_sigmas = (0, 0, 0, 0)) 

            # Apply the transformation to the original image, mask, and CLAHE image
            transformed_img = ants.apply_transforms(fixed=fixed_image, moving=ants_images[x], transformlist=registered['fwdtransforms'])
            transformed_clahe = ants.apply_transforms(fixed=fixed_image, moving=ants_clahe[x], transformlist=registered['fwdtransforms'])

            # Convert transformed images back to SimpleITK format
            numpy_image = transformed_img.numpy()
            numpy_clahe = transformed_clahe.numpy()

            resultImage = sitk.GetImageFromArray(numpy_image)
            resultClahe = sitk.GetImageFromArray(numpy_clahe)

            # Set the spacing, origin, and direction for the SimpleITK images
            resultImage.SetSpacing(image3D[x].GetSpacing())
            resultImage.SetOrigin(image3D[x].GetOrigin())
            resultImage.SetDirection(image3D[x].GetDirection())

            resultClahe.SetSpacing(clahe3D[x].GetSpacing())
            resultClahe.SetOrigin(clahe3D[x].GetOrigin())
            resultClahe.SetDirection(clahe3D[x].GetDirection())

            resultImage3d.append(resultImage)
            resultClahe3d.append(resultClahe)

        else:
            # Directly use the reference image, mask, and CLAHE image
            resultImage3d.append(image3D[x])
            resultClahe3d.append(clahe3D[x])

    return resultImage3d, resultClahe3d

# pairwise end-inhlae ANTs registration CC
def register_EI_ants(image3D, clahe3D, refno, savedir):
    # Convert images, masks, and CLAHE images to ANTs format
    ants_images = [ants.from_numpy(sitk.GetArrayFromImage(img)) for img in image3D]
    ants_clahe = [ants.from_numpy(sitk.GetArrayFromImage(clahe)) for clahe in clahe3D]

    resultImage3d = []
    resultClahe3d = []

    for x in range(7):
        fixed_image = ants_clahe[refno]
        moving_image = ants_clahe[x]

        # Perform registration using CLAHE images
        registered = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyNCC')

        # Apply the transformation to the original image, mask, and CLAHE image
        transformed_img = ants.apply_transforms(fixed=fixed_image, moving=ants_images[x], transformlist=registered['fwdtransforms'])
        transformed_clahe = ants.apply_transforms(fixed=fixed_image, moving=ants_clahe[x], transformlist=registered['fwdtransforms'])

        # Convert transformed images back to SimpleITK format
        numpy_image = transformed_img.numpy()
        numpy_clahe = transformed_clahe.numpy()

        resultImage = sitk.GetImageFromArray(numpy_image)
        resultClahe = sitk.GetImageFromArray(numpy_clahe)

        # Apply binary threshold to the mask
        # resultMask = sitk.BinaryThreshold(resultMask, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)

        # Set the spacing, origin, and direction for the SimpleITK images
        resultImage.SetSpacing(image3D[x].GetSpacing())
        resultImage.SetOrigin(image3D[x].GetOrigin())
        resultImage.SetDirection(image3D[x].GetDirection())

        resultClahe.SetSpacing(clahe3D[x].GetSpacing())
        resultClahe.SetOrigin(clahe3D[x].GetOrigin())
        resultClahe.SetDirection(clahe3D[x].GetDirection())

        resultImage3d.append(resultImage)
        # resultMask3d.append(resultMask)
        resultClahe3d.append(resultClahe)

    return resultImage3d, resultClahe3d

def main():
    # Read the initial files
    image, clahe = read_files(input_files)    

    # image = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/fourorcccm/groupwise/3/image_groupwise.nii')
    # clahe = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/fourorcccm/groupwise/3/image_groupwise.nii') 

    # sitk.WriteImage(image, os.path.join(save_dir, "image_init.nii"))
    # sitk.WriteImage(clahe, os.path.join(save_dir, "clahe_init.nii"))
    
    # s0  = time.time()
    # # Perform initial groupwise registration (False: USE INTERNAL CLAHE IF ENHANCED)    
    # image4D, mask4D, clahe4D = register_groupwise(image, clahe, os.path.join(save_dir, 'groupwise/0/'), enhanced=True, target_weights = [14], claheinternal=False, noisereduction=False)
    # # image4D, mask4D, clahe4D = register_groupwise(image, clahe, os.path.join(save_dir, 'groupwise/3/'), enhanced=False, target_weights = [7], claheinternal=True, noisereduction=False)
    # # image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/groupwise/0N/image_groupwise.nii')
    # # mask4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/groupwise/0N/mask_groupwise.nii')
    # # clahe4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/groupwise/0N/clahe_groupwise.nii') 
    # e0 = time.time()


    # s1  = time.time()
    # # Perform initial groupwise registration (False: USE INTERNAL CLAHE IF ENHANCED)
    # image4D, mask4D, clahe4D = register_groupwise(image4D, clahe4D, os.path.join(save_dir, 'groupwise/1/'), enhanced=False, target_weights = [14], claheinternal=False, noisereduction=True)
    # # image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-10-10_002BB/reg/registered.nii')
    # # mask4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-10-10_002BB/reg/groupwise/0/image_groupwise.nii')
    # # clahe4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-10-10_002BB/reg/groupwise/0/image_groupwise.nii') 
    # e1 = time.time()

    # s2  = time.time()
    # # # # # Perform step registration
    # # # # image4D, mask4D, clahe4D = register_stepwise(image4D, clahe4D, endinhale, os.path.join(save_dir, 'stepwise/'))
    # e2 = time.time()


    # s3  = time.time()
    # # # Repeat the registration process
    # # # Perform initial groupwise registration (True: USE IMAGE ITSELF, ENHANCED or NONENHAMCED)
    # image4D, mask4D, clahe4D = register_groupwise(image4D, clahe4D, os.path.join(save_dir, 'groupwise/2/'), enhanced=False, target_weights = [14], claheinternal=True, noisereduction=False)
    # # # image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/groupwise/2/image_groupwise.nii')
    # # # mask4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/groupwise/2/mask_groupwise.nii')
    # # # clahe4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/groupwise/2/clahe_groupwise.nii') 
    # e3 = time.time()

    # # # # # # Extract 3D images from the 4D results
    # # # # # image3D = extract_image_3d(image4D)
    # # # # # clahe3D = extract_image_3d(clahe4D)

    # s4  = time.time()
    # # # # # # Perform registration with ANTs
    # # # # # refno = 7
    # # # # # image3D, clahe3D = register_pairwise_ants(image3D, clahe3D, endinhale)
    # e4 = time.time()

    # # # # Join 3D images into a 4D image
    # # # image4D = join_image3d(image3D)
    # # # clahe4D = join_image3d(clahe3D)

    # # # # # Write the final images
    # # # # sitk.WriteImage(image4D, os.path.join(os.path.join(save_dir, 'pairwise/'), "aresult_pairwise.nii"))
    # # # # sitk.WriteImage(clahe4D, os.path.join(os.path.join(save_dir, 'pairwise/'), "aclahe_pairwise.nii"))

    # # # # index_sets = None
    # # # # target_weights = None

    # s5  = time.time()
    # # # # Final groupwise registration
    # image4D, mask4D, clahe4D = register_groupwise(image4D, clahe4D, os.path.join(save_dir, 'groupwise/3/'), enhanced=False, target_weights = [14], claheinternal=False, noisereduction=False)
    image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/sevencmbestc/groupwise/3/image_groupwise.nii')
    mask4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/sevencmbestc/groupwise/3/mask_groupwise.nii')
    clahe4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-11-23_039CP/reg/sevencmbestc/groupwise/3/clahe_groupwise.nii') 
    # e5 = time.time()
    

    # # Extract and register images for the final time
    image3D = extract_image_3d(image4D) 
    clahe3D = extract_image_3d(clahe4D)

    s6  = time.time()
    # # Final registration    
    image3D, clahe3D = register_EI_ants(image3D, image3D, refno, save_dir)
    e6 = time.time()

    # Join the final 3D images into a 4D image and save
    image4D = join_image3d_ants(image3D)
    sitk.WriteImage(image4D, os.path.join(save_dir, "registered.nii"))

    # Print the total time taken for registration
    # print('1st Groupwise Registration took', round((e0 - s0) / 60, 2), 'minutes.')
    # print('1st Groupwise Registration took', round((e1 - s1) / 60, 2), 'minutes.')
    # print('Stepwise Registration took', round((e2 - s2) / 60, 2), 'minutes.')
    # print('2nd Groupwise Registration took', round((e3 - s3) / 60, 2), 'minutes.')
    # print('ANTs Pairwise Registration took', round((e4 - s4) / 60, 2), 'minutes.')
    # print('3rd Groupwise Registration took', round((e5 - s5) / 60, 2), 'minutes.')
    print('Last EI Registration took', round((e6 - s6) / 60, 2), 'minutes.')

if __name__ == "__main__":
    main()