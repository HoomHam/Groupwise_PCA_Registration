"""
Reference montage: mid-slice along each of the 3 spatial axes of gas.mat,
frame 0, to identify which axis is sagittal/coronal/axial by eye.
"""
import os
import numpy as np
import scipy.io
import imageio.v3 as iio

REG_DIR = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/outputs/registered/23_oneshot'
OUT_PNG = os.path.join(REG_DIR, 'orientation_montage.png')
SCALE = 3

def norm_u8(img):
    lo, hi = img.min(), img.max()
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)

def upscale(img, s=SCALE):
    return np.kron(img, np.ones((s, s), dtype=np.uint8))

def main():
    d = scipy.io.loadmat(os.path.join(REG_DIR, 'gas.mat'))
    gas = d['gas'][:, :, :, 0]   # (X, Y, Z) frame 0

    nx, ny, nz = gas.shape
    slice_x = upscale(norm_u8(gas[nx // 2, :, :]))   # fix X -> (Y,Z) plane
    slice_y = upscale(norm_u8(gas[:, ny // 2, :]))   # fix Y -> (X,Z) plane
    slice_z = upscale(norm_u8(gas[:, :, nz // 2]))   # fix Z -> (X,Y) plane

    h = max(slice_x.shape[0], slice_y.shape[0], slice_z.shape[0])
    def pad(img):
        out = np.zeros((h, img.shape[1]), dtype=np.uint8)
        out[:img.shape[0], :] = img
        return out

    gap = np.full((h, 10), 128, dtype=np.uint8)
    montage = np.concatenate([pad(slice_x), gap, pad(slice_y), gap, pad(slice_z)], axis=1)
    iio.imwrite(OUT_PNG, montage)
    print(f"Done -> {OUT_PNG}")
    print("Left panel: fix X (dim0) -> Y-Z plane")
    print("Mid panel:  fix Y (dim1) -> X-Z plane")
    print("Right panel: fix Z (dim2) -> X-Y plane")

if __name__ == "__main__":
    main()
