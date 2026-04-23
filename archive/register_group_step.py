import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time 
import scipy.io

import ants
from operator import truediv

# reading matlab 4D image and puting 3D images together
# def read_mat_file(file_path):
#     # Load the .mat file
#     mat_content = scipy.io.loadmat(file_path)
#     data_4d = mat_content['rspace']

#     vector_of_images = sitk.VectorOfImage()

#     for t in range(data_4d.shape[0]):
#     # Convert numpy array to SimpleITK image
#         data_3d = data_4d[t,:,:,:]
#         data_3d_transpose = np.transpose(data_3d, (2,1,0))
#         sitk_image = sitk.GetImageFromArray(data_3d_transpose)
#         print("the sitk mat shape is:", sitk_image.GetSize())
#         vector_of_images.push_back(sitk_image)

#     return sitk.JoinSeries(vector_of_images)

# def read_mat_masks(mask_path):
#     # Load the .mat file
#     mat_content = scipy.io.loadmat(mask_path)
#     data_4d = mat_content['mask']

#     vector_of_images = sitk.VectorOfImage()

#     for t in range(data_4d.shape[0]):
#     # Convert numpy array to SimpleITK image
#         data_3d = data_4d[t,:,:,:]
#         data_3d_transpose = np.transpose(data_3d, (2,1,0))
#         data_3d_unit8 = np.uint8(data_3d)
#         sitk_image = sitk.GetImageFromArray(data_3d_unit8)
#         print("the sitk mat shape is:", sitk_image.GetSize())
#         vector_of_images.push_back(sitk_image)

#     return sitk.JoinSeries(vector_of_images)

# # reading from drive and puting 3D images together
# def read_images(image_dir):
#     population = sorted(os.listdir(image_dir))
#     print("The population is in form:", len(population))
#     vector_of_images = sitk.VectorOfImage()
#     for filename in population:
#         mag = sitk.ReadImage(os.path.join(image_dir, filename))
#         print("the Mag shape is:", mag.GetSize())
#         vector_of_images.push_back(sitk.ReadImage(os.path.join(image_dir, filename)))
#     return sitk.JoinSeries(vector_of_images)

# # reading from drive and puting 3D masks together
# def read_masks(mask_dir):
#     population = sorted(os.listdir(mask_dir))
#     vector_of_masks = sitk.VectorOfImage()
#     for filename in population:
#         vector_of_masks.push_back(sitk.ReadImage(os.path.join(mask_dir, filename), sitk.sitkUInt8))
#     return sitk.JoinSeries(vector_of_masks)

# combining 3D+t images to generate 4D ITK images
# def make_image_4d(image):
#     for i in range(16): 
#         vectorOfImages.push_back(sitk.ReadImage(image[i]))    
#     return sitk.JoinSeries(vector_of_masks)

# def read_files(image_path_or_dir, mask_path_or_dir=None):
#     def process_nifti_directory(directory, is_mask):
#         vector_of_images = sitk.VectorOfImage()
#         for filename in sorted(os.listdir(directory)):
#             if filename.endswith(('.nii', '.nii.gz')):
#                 image_path = os.path.join(directory, filename)
#                 if is_mask:
#                     vector_of_images.push_back(sitk.ReadImage(image_path, sitk.sitkUInt8))
#                 else:
#                     vector_of_images.push_back(sitk.ReadImage(image_path))
#         return sitk.JoinSeries(vector_of_images)

#     def process_mat_file(file_path, is_mask):
#         _, file_extension = os.path.splitext(file_path)
#         if file_extension.lower() == '.mat':
#             mat_content = scipy.io.loadmat(file_path)
#             key = 'mask' if is_mask else 'rspace'
#             data_4d = mat_content[key]

#             vector_of_images = sitk.VectorOfImage()
#             for t in range(data_4d.shape[0]):
#                 data_3d = data_4d[t, :, :, :]
#                 if is_mask:
#                     data_3d = np.uint8(data_3d)
#                 data_3d_transposed = np.transpose(data_3d, (2, 1, 0))
#                 sitk_image = sitk.GetImageFromArray(data_3d_transposed)
#                 vector_of_images.push_back(sitk_image)

#             return sitk.JoinSeries(vector_of_images)
#         else:
#             raise ValueError("Unsupported file format")

#     # Process images
#     if os.path.isdir(image_path_or_dir):
#         images = process_nifti_directory(image_path_or_dir, False)
#     elif os.path.isfile(image_path_or_dir):
#         images = process_mat_file(image_path_or_dir, False)
#     else:
#         raise ValueError("Invalid image path or directory")

#     # Process masks if provided
#     masks = None
#     if mask_path_or_dir:
#         if os.path.isdir(mask_path_or_dir):
#             masks = process_nifti_directory(mask_path_or_dir, True)
#         elif os.path.isfile(mask_path_or_dir):
#             masks = process_mat_file(mask_path_or_dir, True)
#         else:
#             raise ValueError("Invalid mask path or directory")

#     return images, masks

# import os
# import numpy as np
# import SimpleITK as sitk
# import scipy.io

def read_files(image_path_or_dir, mask_path_or_dir=None, clahe_path_or_dir=None):
    def process_nifti_directory(directory, is_mask):
        vector_of_images = sitk.VectorOfImage()
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(('.nii', '.nii.gz')):
                image_path = os.path.join(directory, filename)
                if is_mask:
                    vector_of_images.push_back(sitk.ReadImage(image_path, sitk.sitkUInt8))
                else:
                    vector_of_images.push_back(sitk.ReadImage(image_path))
        return sitk.JoinSeries(vector_of_images)

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

    # Process images
    images = None
    if os.path.isdir(image_path_or_dir):
        images = process_nifti_directory(image_path_or_dir, False)
    elif os.path.isfile(image_path_or_dir):
        images = process_mat_file(image_path_or_dir, 'rspace')

    # Process masks if provided
    masks = None
    if mask_path_or_dir and os.path.isfile(mask_path_or_dir):
        masks = process_mat_file(mask_path_or_dir, 'mask')

    # Process CLAHE images if provided
    clahe_images = None
    if clahe_path_or_dir and os.path.isfile(clahe_path_or_dir):
        clahe_images = process_mat_file(clahe_path_or_dir, 'clahe')

    if mask_path_or_dir is None and clahe_path_or_dir is None:
        return images
    elif clahe_path_or_dir is None:
        return images, masks
    else:
        return images, masks, clahe_images

def print_image_sizes(image_list):
    """
    Print the size of each image in a list of SimpleITK images.

    :param image_list: List of SimpleITK images.
    """
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

# combining 3D+t masks to generate 4D ITK images
def join_mask3d():
    fixed_mask = sitk.ReadImage(mask_dir + 'mask' + '0' + str(8) + '.nii', sitk.sitkUInt8)
    vector_of_masks = sitk.VectorOfImage()
    for x in range(16):
        vector_of_masks.push_back(fixed_mask)
    return sitk.JoinSeries(vector_of_masks)

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

# combining 3D+t masks to generate ANTs images
def join_mask3d_ant(image3D):
    vector_of_masks = sitk.VectorOfImage()
    for img in image3D:
        if not isinstance(img, sitk.Image):
            raise TypeError(f"Expected sitk.Image, got {type(img)}")
        if img.GetDimension() != 3:
            raise ValueError("Expected 3D image")
        
        # Read the mask and adjust its properties to match the image
        # mask = sitk.ReadImage(mask_dir + 'mask' + '0' + str(8) + '.nii', sitk.sitkUInt8)
        mask.SetOrigin(img.GetOrigin())
        mask.SetSpacing(img.GetSpacing())
        mask.SetDirection(img.GetDirection())

        vector_of_masks.push_back(mask)
    return sitk.JoinSeries(vector_of_masks)

# groupwise 4D (3D+t) registration w/ PCA2 [Huiziga]
def register_groupwise(image, mask, clahe, savedir):
   
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
    # parameterMap['MaximumNumberOfIterations'] = ['10000', '20000', '30000', '40000']
    parameterMap['MaximumNumberOfIterations'] = ['1']
    # Pyramid Setting
    parameterMap['GridSpacingSchedule'] = ['4','3','2','1']
    parameterMap['ImagePyramidSchedule'] = ['8','8','8' ,'0','4','4','4','0','2','2','2','0','1','1','1','0']
    
    # Sampler Parameters
    parameterMap['NumberOfSpatialSamples'] = ['1024']
    parameterMap['NewSamplesEveryIteration'] = ['true']
    parameterMap['ImageSampler'] = ['RandomCoordinate']
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

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(mask)
    transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformix.Execute()
    # resultImage4d = elastixImageFilter.GetResultImage()
    resultMask4d = transformix.GetResultImage()
    # print("Type of resultMask4d:", type(resultMask4d))

    # Get the size of resultMask4d
    size = resultMask4d.GetSize()
    _, _, _, timePoints = size

    vectorOfMasks = sitk.VectorOfImage()
    for t in range(timePoints):
        # Extract the 3D volume for each time point
        mask3d = resultMask4d[:, :, :, t]
        
        # Adjust the threshold values here
        lowerThreshold = 0.001  # Example: lowered from 0.5 to 0.2
        upperThreshold = 1.1  # Usually remains 1.0
        thresholdedMask3d = sitk.BinaryThreshold(mask3d, lowerThreshold, upperThreshold, insideValue=1, outsideValue=0)

        # Append the thresholded 3D volume to the vector
        vectorOfMasks.push_back(thresholdedMask3d)

    # Combine the thresholded 3D volumes back into a 4D image
    resultMask4d = sitk.JoinSeries(vectorOfMasks)

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(clahe)
    transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformix.Execute()
    # resultImage4d = elastixImageFilter.GetResultImage()
    resultClahe4d = transformix.GetResultImage()

    # saving the temple file obtained from the population
    sitk.WriteImage(resultImage4d, os.path.join(savedir, "result_groupwise.nii"))
    sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_groupwise.nii"))
    sitk.WriteImage(resultClahe4d, os.path.join(savedir, "clahe_groupwise.nii"))
    
    return resultImage4d, resultMask4d, resultClahe4d

# step-by-step 3-16 bins groupwise 4D (3D+t) registration w/ PCA2 [Hooman]
def register_pairwise(image, mask, clahe, refno, savedir):
    
    # psave_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS/regnii/pairwise/'

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
    parameterMap['Metric'] = ['SumOfPairwiseCorrelationCoefficientsMetric']
    # parameterMap['Metric'] = ['PCAMetric2']

    parameterMap['SubtractMean'] = ['true']
    parameterMap['MovingImageDerivativeScales'] = ['1','1','1','0']
    parameterMap['(FinalGridSpacingInPhysicalUnits'] = ['6']
    
    # Optimizer Setting
    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['AutomaticParameterEstimation'] = ['true']
    parameterMap['ASGDParameterEstimationMethod'] = ['AdaptiveStocDisplacementDistributionhasticGradientDescent']
    # parameterMap['MaximumNumberOfIterations'] = ['5000', '10000', '20000', '30000']
    # parameterMap['MaximumNumberOfIterations'] = ['1000']
    parameterMap['MaximumNumberOfIterations'] = ['1']

    # Pyramid Setting
    parameterMap['GridSpacingSchedule'] = ['4','3','2','1']
    parameterMap['ImagePyramidSchedule'] = ['8','8','8' ,'0','4','4','4','0','2','2','2','0','1','1','1','0']
    
    # Sampler Parameters
    parameterMap['NumberOfSpatialSamples'] = ['1024']
    parameterMap['NewSamplesEveryIteration'] = ['true']
    parameterMap['ImageSampler'] = ['RandomCoordinate']
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
            
            # print(len(image))
            print(indices_to_stack)
            
            for i in indices_to_stack:
                vectorOfImages.push_back(image[i])
            Image4d = sitk.JoinSeries(vectorOfImages)

            for i in indices_to_stack:
                mask_sitk = mask[i]
                mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
                vectorOfMasks.push_back(mask_sitk)
            Mask4d = sitk.JoinSeries(vectorOfMasks)

            for i in indices_to_stack:
                vectorOfClahe.push_back(clahe[i])
            Clahe4d = sitk.JoinSeries(vectorOfClahe)
            
            sitk.WriteImage(Image4d, os.path.join(savedir, "hresult_stepwise.nii"))
            sitk.WriteImage(Mask4d, os.path.join(savedir, "hmask_stepwise.nii"))
            sitk.WriteImage(Clahe4d, os.path.join(savedir, "hclahe_stepwise.nii"))      

            # print("Type of resultMask4d:", type(resultMask4d))

            # Get the size of resultMask4d
            # size = Clahe4d.GetSize()
            # _, _, _, timePoints = size
            # print("Clahe size:", size)

            size = Mask4d.GetSize()
            _, _, _, timePoints = size
            print("Mask size:", size)

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

            transformix = sitk.TransformixImageFilter()
            transformix.SetMovingImage(Mask4d)
            transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformix.Execute()
            # resultImage4d = elastixImageFilter.GetResultImage()
            resultMask4d = transformix.GetResultImage()

            vectorOfMasks = sitk.VectorOfImage()
            for t in range(timePoints):
                # Extract the 3D volume for each time point
                mask3d = resultMask4d[:, :, :, t]
                
                # Adjust the threshold values here
                lowerThreshold = 0.01  # Example: lowered from 0.5 to 0.2
                upperThreshold = 1.01  # Usually remains 1.0
                thresholdedMask3d = sitk.BinaryThreshold(mask3d, lowerThreshold, upperThreshold, insideValue=1, outsideValue=0)
                
                # Append the thresholded 3D volume to the vector
                vectorOfMasks.push_back(thresholdedMask3d)

            # Combine the thresholded 3D volumes back into a 4D image
            resultMask4d = sitk.JoinSeries(vectorOfMasks)

            transformix = sitk.TransformixImageFilter()
            transformix.SetMovingImage(Clahe4d)
            transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformix.Execute()
            # resultImage4d = elastixImageFilter.GetResultImage()
            resultClahe4d = transformix.GetResultImage()    
            # resultImage4d = elastixImageFilter.GetResultImage()

            Image3d = extract_image_3d(resultImage4d)
            Mask3d = extract_image_3d(resultMask4d)
            Clahe3d = extract_image_3d(resultClahe4d)

            ## print('hellohellohellohellohellohellohello')
            ## image[x:x+3] = Image3d[x:x+3]
            # size = Image3d.GetSize()
            # # _, _, _, timePoints = size
            # print("Clahe size:", size)
            
            # print_image_sizes(Mask3d)

            # def print_image_sizes(image_list):
            # """
            # Print the size of each image in a list of SimpleITK images.

            # :param image_list: List of SimpleITK images.
            # """
            # for i, img in enumerate(image_list):
            #     size = img.GetSize()
            #     print(f"Image {i} size: {size}")

            for i in range(m):
                image[indices_to_stack[i]] = Image3d[i]
                mask[indices_to_stack[i]] = Mask3d[i]
                clahe[indices_to_stack[i]] = Clahe3d[i]
            
            # print(len(Image3d))
            # print(len(image))

            resultImage3d = image  
            resultMask3d = mask 
            resultClahe3d = clahe       

        sitk.WriteImage(resultImage4d, os.path.join(savedir, "result_stepwise.nii"))
        sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_stepwise.nii"))
        sitk.WriteImage(resultClahe4d, os.path.join(savedir, "clahe_stepwise.nii"))      


    # saving the temple file obtained from the population
    sitk.WriteImage(resultImage4d, os.path.join(savedir, "result_groupwise.nii"))
    sitk.WriteImage(resultMask4d, os.path.join(savedir, "mask_groupwise.nii"))
    sitk.WriteImage(resultClahe4d, os.path.join(savedir, "clahe_groupwise.nii"))

    return resultImage4d, resultMask4d, resultClahe4d

# pairwise neighbor to neighbor ANTs registration CC
def register_pairwise_ants(image3D, mask3D, clahe3D, refno, savedir):
    
    # Assuming image3D is a list of SimpleITK Image objects
    # numpy_images = [sitk.GetArrayFromImage(image) for image in image3D]
    # ants_images = [ants.from_numpy(image) for image in numpy_images]

    # numpy_masks = [sitk.GetArrayFromImage(mask) for mask in mask3D]
    # ants_masks = [ants.from_numpy(mask) for mask in numpy_masks]

    # numpy_clahe = [sitk.GetArrayFromImage(clahe) for clahe in clahe3D]
    # ants_clahe = [ants.from_numpy(clahe) for image in numpy_clahe]
    
    ants_images = [ants.from_numpy(sitk.GetArrayFromImage(img)) for img in image3D]
    ants_masks = [ants.from_numpy(sitk.GetArrayFromImage(mask)) for mask in mask3D]
    ants_clahe = [ants.from_numpy(sitk.GetArrayFromImage(clahe)) for clahe in clahe3D]

    resultImage3d = []
    resultMask3d = []
    resultClahe3d = []

    for x in range(16):
        if x != refno-1:
            fixed_image = ants_clahe[x+1] if x < refno else ants_clahe[x-1]
            moving_image = ants_clahe[x]
        
            # Perform registration using CLAHE images
            registered = ants.registration(fixed = fixed_image, moving = moving_image, type_of_transform = 'SyN', flow_sigma = 0 , aff_iterations = (5, 5, 5, 5), aff_smoothing_sigmas = (0, 0, 0, 0)) 

            # Apply the transformation to the original image, mask, and CLAHE image
            transformed_img = ants.apply_transforms(fixed=fixed_image, moving=ants_images[x], transformlist=registered['fwdtransforms'])
            transformed_mask = ants.apply_transforms(fixed=fixed_image, moving=ants_masks[x], transformlist=registered['fwdtransforms'], interpolator='nearestNeighbor')
            #transformed_mask = ants_images[x]
            transformed_clahe = ants.apply_transforms(fixed=fixed_image, moving=ants_clahe[x], transformlist=registered['fwdtransforms'])

            # Convert transformed images back to SimpleITK format
            numpy_image = transformed_img.numpy()
            numpy_mask = transformed_mask.numpy()
            numpy_clahe = transformed_clahe.numpy()

            resultImage = sitk.GetImageFromArray(numpy_image)
            resultMask = sitk.GetImageFromArray(numpy_mask)
            resultClahe = sitk.GetImageFromArray(numpy_clahe)

            # Apply binary threshold to the mask
            resultMask = sitk.BinaryThreshold(resultMask, lowerThreshold=0.5, upperThreshold=1.00, insideValue=1, outsideValue=0)

            # Set the spacing, origin, and direction for the SimpleITK images
            resultImage.SetSpacing(image3D[x].GetSpacing())
            resultImage.SetOrigin(image3D[x].GetOrigin())
            resultImage.SetDirection(image3D[x].GetDirection())

            resultMask.SetSpacing(mask3D[x].GetSpacing())
            resultMask.SetOrigin(mask3D[x].GetOrigin())
            resultMask.SetDirection(mask3D[x].GetDirection())

            resultClahe.SetSpacing(clahe3D[x].GetSpacing())
            resultClahe.SetOrigin(clahe3D[x].GetOrigin())
            resultClahe.SetDirection(clahe3D[x].GetDirection())

            resultImage3d.append(resultImage)
            resultMask3d.append(resultMask)
            resultClahe3d.append(resultClahe)

        else:
            # Directly use the reference image, mask, and CLAHE image
            resultImage3d.append(image3D[x])
            resultMask3d.append(mask3D[x])
            resultClahe3d.append(clahe3D[x])

    return resultImage3d, resultMask3d, resultClahe3d

# Example usage

        
    #     if x < refno:
    #         fixed_image = ants_images[x+1]
    #         moving_image = ants_images[x]

    #     if x == refno-1:
    #         resultImageANTs = ants_images[x]
    #         numpy_image = resultImageANTs.numpy()  # Convert ANTs image to NumPy array
    #         resultImage = sitk.GetImageFromArray(numpy_image)  
    #         resultImage3d.append(resultImage) 
            
    #     if x > refno-1:    
    #         fixed_image = ants_images[x-1]
    #         moving_image = ants_images[x]
 
    #     if x != refno-1:
    #         registered = ants.registration(fixed = fixed_image, moving = moving_image, type_of_transform = 'SyNCC', flow_sigma = 0 , aff_iterations = (5, 5, 5, 5), aff_smoothing_sigmas = (0, 0, 0, 0))    
    #         resultImageANTs = ants.apply_transforms(fixed = fixed_image , moving = moving_image, transformlist = registered['fwdtransforms'])
      
    #         # Convert ANTs image to NumPy array
    #         numpy_image = resultImageANTs.numpy()

    #         # Convert NumPy array to SimpleITK image
    #         resultImage = sitk.GetImageFromArray(numpy_image)

    #         # Set the spacing, origin, and direction for the SimpleITK image
    #         resultImage.SetSpacing(fixed_image.spacing)
    #         resultImage.SetOrigin(fixed_image.origin)
    #         flattened_direction = [float(val) for sublist in fixed_image.direction for val in sublist]
    #         resultImage.SetDirection(flattened_direction)

    #         # Append the SimpleITK image to the list
    #         resultImage3d.append(resultImage)    
    #         print(os.path.join(savedir, f'result_pairwise_{x+1}.nii'))

    # return resultImage3d

# pairwise end-inhlae ANTs registration CC
import os
import ants
import SimpleITK as sitk

def register_EI_ants(image3D, mask3D, clahe3D, refno, savedir):
    # Convert images, masks, and CLAHE images to ANTs format
    ants_images = [ants.from_numpy(sitk.GetArrayFromImage(img)) for img in image3D]
    ants_masks = [ants.from_numpy(sitk.GetArrayFromImage(mask)) for mask in mask3D]
    ants_clahe = [ants.from_numpy(sitk.GetArrayFromImage(clahe)) for clahe in clahe3D]

    resultImage3d = []
    resultMask3d = []
    resultClahe3d = []

    for x in range(16):
        fixed_image = ants_clahe[refno]
        moving_image = ants_clahe[x]

        # Perform registration using CLAHE images
        registered = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')

        # Apply the transformation to the original image, mask, and CLAHE image
        transformed_img = ants.apply_transforms(fixed=fixed_image, moving=ants_images[x], transformlist=registered['fwdtransforms'])
        transformed_mask = ants.apply_transforms(fixed=fixed_image, moving=ants_masks[x], transformlist=registered['fwdtransforms'], interpolator='nearestNeighbor')
        transformed_clahe = ants.apply_transforms(fixed=fixed_image, moving=ants_clahe[x], transformlist=registered['fwdtransforms'])

        # Convert transformed images back to SimpleITK format
        numpy_image = transformed_img.numpy()
        numpy_mask = transformed_mask.numpy()
        numpy_clahe = transformed_clahe.numpy()

        resultImage = sitk.GetImageFromArray(numpy_image)
        resultMask = sitk.GetImageFromArray(numpy_mask)
        resultClahe = sitk.GetImageFromArray(numpy_clahe)

        # Apply binary threshold to the mask
        resultMask = sitk.BinaryThreshold(resultMask, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)

        # Set the spacing, origin, and direction for the SimpleITK images
        resultImage.SetSpacing(image3D[x].GetSpacing())
        resultImage.SetOrigin(image3D[x].GetOrigin())
        resultImage.SetDirection(image3D[x].GetDirection())

        resultMask.SetSpacing(mask3D[x].GetSpacing())
        resultMask.SetOrigin(mask3D[x].GetOrigin())
        resultMask.SetDirection(mask3D[x].GetDirection())

        resultClahe.SetSpacing(clahe3D[x].GetSpacing())
        resultClahe.SetOrigin(clahe3D[x].GetOrigin())
        resultClahe.SetDirection(clahe3D[x].GetDirection())

        resultImage3d.append(resultImage)
        resultMask3d.append(resultMask)
        resultClahe3d.append(resultClahe)

    return resultImage3d, resultMask3d, resultClahe3d



# def register_EI_ants(image3D, refno, savedir):
#     # Assuming image3D is a list of SimpleITK Image objects
#     numpy_images = [sitk.GetArrayFromImage(image) for image in image3D]
#     ants_images = [ants.from_numpy(image) for image in numpy_images]

#     resultImage3d = []
#     for x in range(16):
    
#         fixed_image = ants_images[refno]
#         moving_image = ants_images[x]

#         registered = ants.registration(fixed = fixed_image, moving = moving_image, type_of_transform = 'SyNCC', flow_sigma = 0 , aff_iterations = (5, 5, 5, 5), aff_smoothing_sigmas = (0, 0, 0, 0))    
#         resultImageANTs = ants.apply_transforms(fixed = fixed_image , moving = moving_image, transformlist = registered['fwdtransforms'])
    
#         # Convert ANTs image to NumPy array
#         numpy_image = resultImageANTs.numpy()

#         # Convert NumPy array to SimpleITK image
#         resultImage = sitk.GetImageFromArray(numpy_image)

#         # Set the spacing, origin, and direction for the SimpleITK image
#         resultImage.SetSpacing(fixed_image.spacing)
#         resultImage.SetOrigin(fixed_image.origin)
#         flattened_direction = [float(val) for sublist in fixed_image.direction for val in sublist]
#         resultImage.SetDirection(flattened_direction)

#         # Append the SimpleITK image to the list
#         resultImage3d.append(resultImage)    
#         print(os.path.join(savedir, f'result_pairwise_{x+1}.nii'))

#     # return resultImage3d
#     return resultImage4d, resultMask4d, resultClahe4d

s  = time.time()

# file_path = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_mat/rspace_gas.mat'
# mask_path = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_mat/mspace_gas.mat'
# image_dir = '/Users/hooman/Upenn/Elastix/Analysis/2023_01-15-23_4GroupW/image3DO/'
mask_dir = '/Users/hooman/Upenn/Elastix/Analysis/2023_01-15-23_4GroupW/mask3DO/'

gsave_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_step/regnii/groupwise/'
# psave_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_step/regnii/pairwise/'
save_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_step1000/'

# image = read_mat_file(file_path)
# print("MAT file image size:", image.GetSize())

# image_nifti = read_images(image_dir)
# print("NIfTI images size:", image_nifti.GetSize())

# # mask = read_masks(mask_dir)
# mask = read_mat_masks(mask_path)


# Example usage
nifti_dir = '/Users/hooman/Upenn/Elastix/Analysis/2023_01-15-23_4GroupW/image3DO/'
nifti_mask_dir = '/Users/hooman/Upenn/Elastix/Analysis/2023_01-15-23_4GroupW/mask3DO/'
mat_file = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_mat/rspace_gas.mat'
mat_mask_file = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_mat/mspace_gas.mat'
clahe_path_or_dir = '/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_mat/cspace_gas.mat'  # Optional

image, mask = read_files(mat_file, mat_mask_file)
image, mask, clahe = read_files(mat_file, mat_mask_file, clahe_path_or_dir)

# replace with automatic version
endinhale = 8
#### fixed_mask = sitk.ReadImage(mask_dir + 'mask' + '0' + str(endinhale) + '.nii', sitk.sitkUInt8)

# image4D, mask4D, clahe4D = register_groupwise(image, mask, clahe, os.path.join(gsave_dir,'0/'))

# image3D = extract_image_3d(image4D)
# mask3D = extract_image_3d(mask4D)
# clahe3D = extract_image_3d(clahe4D)

# image4D, mask4D, clahe4D = register_pairwise(image3D, mask3D, clahe3D, endinhale, os.path.join(gsave_dir,'0/'))

image4D = sitk.ReadImage(os.path.join(os.path.join(gsave_dir,'0/'), "result_groupwise.nii"))
mask4D = sitk.ReadImage(os.path.join(os.path.join(gsave_dir,'0/'), "mask_groupwise.nii"))
clahe4D = sitk.ReadImage(os.path.join(os.path.join(gsave_dir,'0/'), "clahe_groupwise.nii"))


image4D, mask4D, clahe4D = register_groupwise(image4D, mask4D, clahe4D,  save_dir )
image3D = extract_image_3d(image4D)
mask3D = extract_image_3d(mask4D)
clahe3D = extract_image_3d(clahe4D)


refno = 7
### image4D = sitk.ReadImage('/Volumes/Macintosh HD 2/Work/Analysis/2023-03-03_032WS_step1000/result_groupwise.nii')

### image3D = extract_image_3d(image4D)
image3D, mask3D, clahe3D = register_pairwise_ants(image3D, mask3D, clahe3D, refno, save_dir)

### print_image_sizes(image3D)
### print_image_sizes(mask3D)
### print_image_sizes(clahe3D)

image4D = join_image3d(image3D)
mask4D = join_image3d(mask3D)
clahe4D = join_image3d(clahe3D)

sitk.WriteImage(image4D, os.path.join(save_dir, "aresult_pairpwise.nii"))
sitk.WriteImage(mask4D, os.path.join(save_dir, "amask_pairwise.nii"))
sitk.WriteImage(clahe4D, os.path.join(save_dir, "aclahe_pairwise.nii"))

image4D, mask4D, clahe4D = register_groupwise(image4D, mask4D, clahe4D,  save_dir )

# sitk.WriteImage(image4D, os.path.join(save_dir, "gresult_pairpwise.nii"))
# sitk.WriteImage(mask4D, os.path.join(save_dir, "gmask_pairwise.nii"))
# sitk.WriteImage(clahe4D, os.path.join(save_dir, "gclahe_pairwise.nii"))

mage3D = extract_image_3d(image4D)
mask3D = extract_image_3d(mask4D)
clahe3D = extract_image_3d(clahe4D)

image3D, mask3D, clahe3D = register_EI_ants(image3D, mask3D, clahe3D, refno, save_dir)

image4D = join_image3d_ants(image3D)

sitk.WriteImage(image4D, os.path.join(save_dir, "result_pairgroupEI.nii"))

### image3D = extract_image_3d(image4D)
### image3D = register_EI_ants(image3D, refno, save_dir)

### image4D = join_image3d_ants(image3D)

### sitk.WriteImage(image4D, os.path.join(save_dir, "result_pairgroupEI.nii"))

print('Registration took', round((time.time() -s )/60,2), 'minutes.')
# Example usage
