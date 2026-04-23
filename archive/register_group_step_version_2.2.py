import os
import time
import numpy as np
import SimpleITK as sitk
import scipy.io
import ants
from scipy.ndimage import label
import nibabel as nib

# Global configuration
# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS/'

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/reg/'

# endinhale = 8
# cutoff = 0.04  # Cutoff value for mask creation

# save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-01-31_000LL/reg/'
# input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-01-31_000LL/reg/'

# endinhale = 7
# cutoff = 0.015  # Cutoff value for mask creation

save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/'
input_files = '/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/rec/'

endinhale = 6
threshold = 0.05  # Cutoff value for mask creation
time_threshold = 5

# Find the input files
def find_file_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.mat'):
            return os.path.join(directory, filename)
    raise ValueError(f"No file starting with '{prefix}' found in {directory}")

def read_files(directory):
    def process_mat_file(file_path, key):
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.mat':
            mat_content = scipy.io.loadmat(file_path)
            data_4d = mat_content.get(key)

            if data_4d is None:
                raise ValueError(f"Key '{key}' not found in MATLAB file")

            vector_of_images = sitk.VectorOfImage()
            for t in range(data_4d.shape[0]):
                data_3d = data_4d[t, :, :, :]
                data_3d_transposed = np.transpose(data_3d, (2, 1, 0))
                sitk_image = sitk.GetImageFromArray(data_3d_transposed)
                vector_of_images.push_back(sitk_image)

            return sitk.JoinSeries(vector_of_images)
        else:
            raise ValueError("Unsupported file format")

    # Find files with specific prefixes
    image_file = find_file_with_prefix(directory, 'rspace')
    clahe_file = find_file_with_prefix(directory, 'cspace')

    # Process images
    images = process_mat_file(image_file, 'rspace')

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

# Function to create mask from image using cutoff
# def create_mask_from_image(image):
#     mask_array = sitk.GetArrayFromImage(image)
#     mask_array[mask_array < cutoff] = 0
#     mask_array[mask_array >= cutoff] = 1
#     mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
#     mask_sitk.CopyInformation(image)
#     return mask_sitk

# def create_mask_from_4d_image(image4d):
#     # Get the size of the 4D image
#     size = image4d.GetSize()
#     if len(size) != 4:
#         raise ValueError("Input image must be 4D.")

#     # Process each 3D image to create a mask
#     image_size = [size[0], size[1], size[2], 0]
#     num_images = size[3]
#     mask_vector = sitk.VectorOfImage()
#     for i in range(num_images):
#         index = [0, 0, 0, i]
#         sub_image = sitk.Extract(image4d, image_size, index)
#         mask_3d = create_mask_from_image(sub_image)
#         mask_vector.push_back(mask_3d)

#     return sitk.JoinSeries(mask_vector)

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
    for i in range(final_mask.shape[0]):
        mask_slice = final_mask[i, :, :, :]
        mask_slice_sitk = sitk.GetImageFromArray(mask_slice.astype(np.uint8))
        mask_slice_sitk.SetSpacing(image4d.GetSpacing()[0:3])  # Set spacing for 3D
        mask_slice_sitk.SetOrigin(image4d.GetOrigin())
        mask_vector.push_back(mask_slice_sitk)

    # Combine the 3D slices into a 4D image
    mask_4d_sitk = sitk.JoinSeries(mask_vector)
    mask_4d_sitk.SetSpacing((image4d.GetSpacing()[-1],) + image4d.GetSpacing()[0:3])  # Set 4D spacing

    return mask_4d_sitk


    # groupwise 4D (3D+t) registration w/ PCA2 [Huiziga]
def register_groupwise(image, clahe, savedir):
   
    # Creating Mask from Images
    mask = create_mask_from_4d_image(image)
    sitk.WriteImage(mask, os.path.join("/Volumes/Macintosh HD 2/Work/Analysis/2023-11-03_000LL/reg/","mask_init.nii"))

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

    parameterMap['SubtractMean'] = ['true']
    parameterMap['MovingImageDerivativeScales'] = ['1','1','1','0']
    parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']
    
    # Optimizer Setting
    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['AutomaticParameterEstimation'] = ['true']
    parameterMap['ASGDParameterEstimationMethod'] = ['AdaptiveStocDisplacementDistributionhasticGradientDescent']
    parameterMap['MaximumNumberOfIterations'] = ['10000', '20000', '30000', '40000']
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

    # transformix = sitk.TransformixImageFilter()
    # transformix.SetMovingImage(mask)
    # transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    # transformix.Execute()
    # resultMask4d = transformix.GetResultImage()

    # Get the size of resultMask4d
    # size = resultMask4d.GetSize()
    # _, _, _, timePoints = size

    vectorOfMasks = sitk.VectorOfImage()
    # for t in range(timePoints):
    #     # Extract the 3D volume for each time point
    #     mask3d = resultMask4d[:, :, :, t]
        
    #     # Adjust the threshold values here
    #     lowerThreshold = 0.001  # Example: lowered from 0.5 to 0.2
    #     upperThreshold = 1.01  # Usually remains 1.0
    #     thresholdedMask3d = sitk.BinaryThreshold(mask3d, lowerThreshold, upperThreshold, insideValue=1, outsideValue=0)

    #     # Append the thresholded 3D volume to the vector
    #     vectorOfMasks.push_back(thresholdedMask3d) 

    # Combine the thresholded 3D volumes back into a 4D image
    # resultMask4d = sitk.JoinSeries(vectorOfMasks)
    resultMask4d = create_mask_from_4d_image(resultImage4d)

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
def register_stepwise(image, mask, clahe, refno, savedir):
    
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

    parameterMap['SubtractMean'] = ['true']
    parameterMap['MovingImageDerivativeScales'] = ['1','1','1','0']
    parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']
    
    # Optimizer Setting
    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['AutomaticParameterEstimation'] = ['true']
    parameterMap['ASGDParameterEstimationMethod'] = ['AdaptiveStocDisplacementDistributionhasticGradientDescent']
    parameterMap['MaximumNumberOfIterations'] = ['1000']
    
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
    resultImage3d = []
    
    for y in range(2,n):
        
        x_range = n if y < n - 1 else n + 1

        for x in range(x_range):
            
            if y == n - 1 and x == n:
                parameterMap['UseCyclicTransform'] = ['true']

            m = y + 1
            vectorOfImages = sitk.VectorOfImage()
            vectorOfMasks = sitk.VectorOfImage()
            vectorOfClahe = sitk.VectorOfImage()
        
            indices_to_stack = [(x + i) % n for i in range(m)]

            print(indices_to_stack)
            
            for i in indices_to_stack:
                vectorOfImages.push_back(image[i])
            Image4d = sitk.JoinSeries(vectorOfImages)

            print(int(m * .3) + (m % .3 > 0))    
            # sitk.WriteImage(Image4d, os.path.join(savedir, "image_test.nii"))        
            Mask4d = create_mask_from_4d_image(Image4d, 0.05, int(m * .3) + (m % .3 > 0))
            
            # for i in indices_to_stack:
            #     mask_sitk = mask[i]
            #     mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
            #     vectorOfMasks.push_back(mask_sitk)
            # Mask4d = sitk.JoinSeries(vectorOfMasks)

            for i in indices_to_stack:
                vectorOfClahe.push_back(clahe[i])
            Clahe4d = sitk.JoinSeries(vectorOfClahe)
            
            size = Mask4d.GetSize()
            _, _, _, timePoints = size

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

            # transformix = sitk.TransformixImageFilter()
            # transformix.SetMovingImage(Mask4d)
            # transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            # transformix.Execute()
            # resultMask4d = transformix.GetResultImage()

            # vectorOfMasks = sitk.VectorOfImage()
            # for t in range(timePoints):
            #     # Extract the 3D volume for each time point
            #     mask3d = resultMask4d[:, :, :, t]
                
            #     # Adjust the threshold values here
            #     lowerThreshold = 0.01  # Example: lowered from 0.5 to 0.2
            #     upperThreshold = 1.015  # Usually remains 1.0
            #     thresholdedMask3d = sitk.BinaryThreshold(mask3d, lowerThreshold, upperThreshold, insideValue=1, outsideValue=0)
                
            #     # Append the thresholded 3D volume to the vector
            #     vectorOfMasks.push_back(thresholdedMask3d)

            # # Combine the thresholded 3D volumes back into a 4D image
            # resultMask4d = sitk.JoinSeries(vectorOfMasks)
            
            resultMask4d = create_mask_from_4d_image(resultImage4d, 0.05, int(m * .3) + (m % .3 > 0))

            transformix = sitk.TransformixImageFilter()
            transformix.SetMovingImage(Clahe4d)
            transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformix.Execute()
            resultClahe4d = transformix.GetResultImage()    

            Image3d = extract_image_3d(resultImage4d)
            Mask3d = extract_image_3d(resultMask4d)
            Clahe3d = extract_image_3d(resultClahe4d)

            for i in range(m):
                image[indices_to_stack[i]] = Image3d[i]
                mask[indices_to_stack[i]] = Mask3d[i]
                clahe[indices_to_stack[i]] = Clahe3d[i]

            resultImage3d = image  
            resultMask3d = mask 
            resultClahe3d = clahe    

        sitk.WriteImage(resultImage4d, os.path.join(savedir, str(m), "image_stepwise.nii"))
        sitk.WriteImage(resultMask4d, os.path.join(savedir, str(m), "mask_stepwise.nii"))
        sitk.WriteImage(resultClahe4d, os.path.join(savedir, str(m), "clahe_stepwise.nii"))      

    # saving the file obtained from the population
    sitk.WriteImage(resultImage4d, os.path.join(savedir, "image_stepwise.nii"))
    sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_stepwise.nii"))
    sitk.WriteImage(resultClahe4d, os.path.join(savedir, "clahe_stepwise.nii"))

    return resultImage4d, resultMask4d, resultClahe4d

# pairwise neighbor to neighbor ANTs registration CC
def register_pairwise_ants(image3D, clahe3D, refno):
    
    ants_images = [ants.from_numpy(sitk.GetArrayFromImage(img)) for img in image3D]
    # ants_masks = [ants.from_numpy(sitk.GetArrayFromImage(mask)) for mask in mask3D]
    ants_clahe = [ants.from_numpy(sitk.GetArrayFromImage(clahe)) for clahe in clahe3D]

    resultImage3d = []
    # resultMask3d = []
    resultClahe3d = []

    for x in range(16):
        if x != refno-1:
            fixed_image = ants_clahe[x+1] if x < refno else ants_clahe[x-1]
            moving_image = ants_clahe[x]
        
            # Perform registration using CLAHE images
            registered = ants.registration(fixed = fixed_image, moving = moving_image, type_of_transform = 'SyNCC', flow_sigma = 0 , aff_iterations = (5, 5, 5, 5), aff_smoothing_sigmas = (0, 0, 0, 0)) 

            # Apply the transformation to the original image, mask, and CLAHE image
            transformed_img = ants.apply_transforms(fixed=fixed_image, moving=ants_images[x], transformlist=registered['fwdtransforms'])
            # transformed_mask = ants.apply_transforms(fixed=fixed_image, moving=ants_masks[x], transformlist=registered['fwdtransforms'], interpolator='nearestNeighbor')
            #transformed_mask = ants_images[x]

            transformed_clahe = ants.apply_transforms(fixed=fixed_image, moving=ants_clahe[x], transformlist=registered['fwdtransforms'])

            # Convert transformed images back to SimpleITK format
            numpy_image = transformed_img.numpy()
            # numpy_mask = transformed_mask.numpy()
            numpy_clahe = transformed_clahe.numpy()

            resultImage = sitk.GetImageFromArray(numpy_image)
            # resultMask = sitk.GetImageFromArray(numpy_mask)
            resultClahe = sitk.GetImageFromArray(numpy_clahe)

            # Apply binary threshold to the mask
            # resultMask = sitk.BinaryThreshold(resultMask, lowerThreshold=0.5, upperThreshold=1.00, insideValue=1, outsideValue=0)

            # Set the spacing, origin, and direction for the SimpleITK images
            resultImage.SetSpacing(image3D[x].GetSpacing())
            resultImage.SetOrigin(image3D[x].GetOrigin())
            resultImage.SetDirection(image3D[x].GetDirection())

            # resultMask.SetSpacing(mask3D[x].GetSpacing())
            # resultMask.SetOrigin(mask3D[x].GetOrigin())
            # resultMask.SetDirection(mask3D[x].GetDirection())

            resultClahe.SetSpacing(clahe3D[x].GetSpacing())
            resultClahe.SetOrigin(clahe3D[x].GetOrigin())
            resultClahe.SetDirection(clahe3D[x].GetDirection())

            resultImage3d.append(resultImage)
            # resultMask3d.append(resultMask)
            resultClahe3d.append(resultClahe)

        else:
            # Directly use the reference image, mask, and CLAHE image
            resultImage3d.append(image3D[x])
            # resultMask3d.append(mask3D[x])
            resultClahe3d.append(clahe3D[x])

    return resultImage3d, resultClahe3d

# pairwise end-inhlae ANTs registration CC
def register_EI_ants(image3D, clahe3D, refno, savedir):
    # Convert images, masks, and CLAHE images to ANTs format
    ants_images = [ants.from_numpy(sitk.GetArrayFromImage(img)) for img in image3D]
    # ants_masks = [ants.from_numpy(sitk.GetArrayFromImage(mask)) for mask in mask3D]
    ants_clahe = [ants.from_numpy(sitk.GetArrayFromImage(clahe)) for clahe in clahe3D]

    resultImage3d = []
    # resultMask3d = []
    resultClahe3d = []

    for x in range(16):
        fixed_image = ants_clahe[refno]
        moving_image = ants_clahe[x]

        # Perform registration using CLAHE images
        registered = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyNCC')

        # Apply the transformation to the original image, mask, and CLAHE image
        transformed_img = ants.apply_transforms(fixed=fixed_image, moving=ants_images[x], transformlist=registered['fwdtransforms'])
        # transformed_mask = ants.apply_transforms(fixed=fixed_image, moving=ants_masks[x], transformlist=registered['fwdtransforms'], interpolator='nearestNeighbor')
        transformed_clahe = ants.apply_transforms(fixed=fixed_image, moving=ants_clahe[x], transformlist=registered['fwdtransforms'])

        # Convert transformed images back to SimpleITK format
        numpy_image = transformed_img.numpy()
        # numpy_mask = transformed_mask.numpy()
        numpy_clahe = transformed_clahe.numpy()

        resultImage = sitk.GetImageFromArray(numpy_image)
        # resultMask = sitk.GetImageFromArray(numpy_mask)
        resultClahe = sitk.GetImageFromArray(numpy_clahe)

        # Apply binary threshold to the mask
        # resultMask = sitk.BinaryThreshold(resultMask, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)

        # Set the spacing, origin, and direction for the SimpleITK images
        resultImage.SetSpacing(image3D[x].GetSpacing())
        resultImage.SetOrigin(image3D[x].GetOrigin())
        resultImage.SetDirection(image3D[x].GetDirection())

        # resultMask.SetSpacing(mask3D[x].GetSpacing())
        # resultMask.SetOrigin(mask3D[x].GetOrigin())
        # resultMask.SetDirection(mask3D[x].GetDirection())

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

    s1  = time.time()
    # Perform initial groupwise registration
    image4D, mask4D, clahe4D = register_groupwise(image, clahe, os.path.join(save_dir, 'groupwise/1/'))
    # image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/groupwise/1/image_groupwise.nii')
    # mask4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/groupwise/1/mask_groupwise.nii')
    # clahe4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-05-15_035EK/reg/groupwise/1/clahe_groupwise.nii')
    
    e1 = time.time()

    # Extract 3D images from the 4D results
    image3D = extract_image_3d(image4D)
    mask3D = extract_image_3d(mask4D)
    clahe3D = extract_image_3d(clahe4D)

    s2  = time.time()
    # Perform step registration
    image4D, mask4D, clahe4D = register_stepwise(image3D, mask3D, clahe3D, endinhale, os.path.join(save_dir, 'stepwise/'))
    e2 = time.time()

    s3  = time.time()
    # Repeat the registration process
    image4D, mask4D, clahe4D = register_groupwise(image4D, clahe4D, os.path.join(save_dir, 'groupwise/2/'))
    e3 = time.time()

    # Extract 3D images from the 4D results
    image3D = extract_image_3d(image4D)
    # mask3D = extract_image_3d(mask4D)
    clahe3D = extract_image_3d(clahe4D)

    s4  = time.time()
    # Perform registration with ANTs
    refno = 7
    image3D, clahe3D = register_pairwise_ants(image3D, clahe3D, endinhale)
    e4 = time.time()

    # Join 3D images into a 4D image
    image4D = join_image3d(image3D)
    # mask4D = join_image3d(mask3D)
    clahe4D = join_image3d(clahe3D)

    # Write the final images
    sitk.WriteImage(image4D, os.path.join(os.path.join(save_dir, 'pairwise/'), "aresult_pairwise.nii"))
    # sitk.WriteImage(mask4D, os.path.join(os.path.join(save_dir, 'pairwise/'), "amask_pairwise.nii"))
    sitk.WriteImage(clahe4D, os.path.join(os.path.join(save_dir, 'pairwise/'), "aclahe_pairwise.nii"))

    s5  = time.time()
    # Final groupwise registration
    image4D, mask4D, clahe4D = register_groupwise(image4D, clahe4D, os.path.join(save_dir, 'groupwise/3/'))
    e5 = time.time()

    # Extract and register images for the final time
    image3D = extract_image_3d(image4D)
    # mask3D = extract_image_3d(mask4D)
    clahe3D = extract_image_3d(clahe4D)

    s6  = time.time()
    # Final registration    
    image3D, clahe3D = register_EI_ants(image3D, clahe3D, refno, save_dir)
    e6 = time.time()

    # Join the final 3D images into a 4D image and save
    image4D = join_image3d_ants(image3D)
    sitk.WriteImage(image4D, os.path.join(save_dir, "registered.nii"))

    # Print the total time taken for registration
    print('1st Groupwise Registration took', round((e1 - s1) / 60, 2), 'minutes.')
    print('Stepwise Registration took', round((e2 - s2) / 60, 2), 'minutes.')
    print('2nd Groupwise Registration took', round((e3 - s3) / 60, 2), 'minutes.')
    print('ANTs Pairwise Registration took', round((e4 - s4) / 60, 2), 'minutes.')
    print('3rd Groupwise Registration took', round((e5 - s5) / 60, 2), 'minutes.')
    print('Last EI Registration took', round((e6 - s6) / 60, 2), 'minutes.')

if __name__ == "__main__":
    main()