import os
import scipy.io
import ants
import time
import numpy as np
from operator import truediv

def register(matfname, index_of_phase=0, EI=truediv, EE=False, big_size=4, npyfile='', register_phase=-1):
    def mat_to_numpy(matfile):
        matfile = scipy.io.loadmat(matfile)
        return matfile[list(matfile.keys())[-1]]

    new_voxel_size = 1 / big_size
    if not (EI or EE) and register_phase == -1:
        print('You must select either EE or EI.')
        return 0
    if os.path.exists(matfname.split('.')[0] + '_registered_small_toEI.mat'):
        print('Registration Found! Skipping this file.')
        return 0
    print('Starting Registration for', matfname.split('.')[0].split('/')[-1])
    if type(npyfile) == str:
        npyfile = scipy.io.loadmat(matfname)
        npyfile = npyfile[list(npyfile.keys())[-1]]

    max_phase, min_phase, max_sum, min_sum = 0, 0, 0, 9999999999999
    all_images = []
    if npyfile.dtype == np.complex64: 
        npyfile = np.abs(npyfile)

    npyfile = npyfile / npyfile.mean()
    # Get the index of phase
    index_of_phase = np.argmin(npyfile.shape)

    for i in range(npyfile.shape[index_of_phase]):
        if index_of_phase == 3:
            all_images.append(npyfile[:, :, :, i])
        if index_of_phase == 0:
            all_images.append(npyfile[i])
        if index_of_phase == 1:
            all_images.append(npyfile[:, i, :, :])
        if index_of_phase == 2:
            all_images.append(npyfile[:, :, i, :])

        if all_images[i].sum() > max_sum: 
            max_sum, max_phase = all_images[i].sum(), i
        if all_images[i].sum() < min_sum: 
            min_sum, min_phase = all_images[i].sum(), i

    s = time.time()
    warped_images_all_big, warped_images_all_small = [], []

    if EI:
        f = ants.resample_image(ants.from_numpy(all_images[max_phase]),
                                [new_voxel_size, new_voxel_size, new_voxel_size], interp_type=0)
    if EE:
        f = ants.resample_image(ants.from_numpy(all_images[min_phase]),
                                [new_voxel_size, new_voxel_size, new_voxel_size], interp_type=0)
    if register_phase != -1:
        print('\tRegistering to phase', register_phase)
        f = ants.resample_image(ants.from_numpy(all_images[register_phase]),
                                [new_voxel_size, new_voxel_size, new_voxel_size], interp_type=0)

    for i in range(len(all_images)):
        m = ants.resample_image(ants.from_numpy(all_images[i]),
                                [new_voxel_size, new_voxel_size, new_voxel_size], interp_type=0)
        mytx = ants.registration(fixed=f, moving=m, type_of_transform='SyN')
        mywarpedimage = ants.apply_transforms(fixed=f, moving=m, transformlist=mytx['fwdtransforms'])
        warped_images_all_big.append(mywarpedimage.numpy())
        mywarpedimage = ants.resample_image(mywarpedimage, [1, 1, 1], interp_type=0)
        warped_images_all_small.append(mywarpedimage.numpy())
    warped_images_all_big = np.array(warped_images_all_big)
    warped_images_all_small = np.array(warped_images_all_small)

    if EE:
        scipy.io.savemat(matfname.split('.mat')[0] + '_registered_small_toEE.mat', {'registered': warped_images_all_small})
    if EI:
        scipy.io.savemat(matfname.split('.mat')[0] + '_registered_small_toEI.mat', {'registered': warped_images_all_small})
    if register_phase != -1:
        scipy.io.savemat(matfname.split('.mat')[0] + '_registered_small_to_R' + str(register_phase) + '.mat',
                         {'registered': warped_images_all_small})
    print('Registration took', round((time.time() - s) / 60, 2), 'minutes.')
    return 0

# Update this path to the location of your .mat file
matfname = r"/Volumes/Macintosh HD 2/Work/Analysis/2024-01-18_001JM/rec/rspace_gas.mat"

# Check if the file exists before running registration
if os.path.exists(matfname):
    register(matfname)
else:
    print("File not found. Please check the path and try again.")
