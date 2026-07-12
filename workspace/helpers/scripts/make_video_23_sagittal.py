"""
Two-panel sagittal video (Y-Z plane) from registered gas.mat, one panel per
lung (fix X), across all 16 timepoints. X indices picked from a per-X
high-signal voxel-count profile: two lobes at X~30 and X~64, valley ~X50.
"""
import os
import shutil
import subprocess
import numpy as np
import scipy.io
import imageio.v3 as iio

REG_DIR   = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/outputs/registered/23_oneshot'
FRAME_DIR = os.path.join(REG_DIR, '_frames_sag_tmp')
OUT_MP4   = os.path.join(REG_DIR, 'gas_sagittal_LR.mp4')
FPS       = 6
SCALE     = 4
X_LEFT    = 30
X_RIGHT   = 64
GAP_PX    = 10


def main():
    d = scipy.io.loadmat(os.path.join(REG_DIR, 'gas.mat'))
    gas = d['gas']   # (X, Y, Z, T)
    T = gas.shape[3]

    left_stack  = gas[X_LEFT, :, :, :]    # (Y, Z, T)
    right_stack = gas[X_RIGHT, :, :, :]   # (Y, Z, T)

    lo = min(left_stack.min(), right_stack.min())
    hi = max(left_stack.max(), right_stack.max())

    def norm_u8(img):
        return ((img - lo) / (hi - lo) * 255).astype(np.uint8)

    def upscale(img, s=SCALE):
        return np.kron(img, np.ones((s, s), dtype=np.uint8))

    os.makedirs(FRAME_DIR, exist_ok=True)
    h = upscale(norm_u8(left_stack[:, :, 0])).shape[0]
    gap = np.full((h, GAP_PX), 128, dtype=np.uint8)

    for t in range(T):
        lf = upscale(norm_u8(left_stack[:, :, t]))
        rf = upscale(norm_u8(right_stack[:, :, t]))
        frame = np.concatenate([lf, gap, rf], axis=1)
        iio.imwrite(os.path.join(FRAME_DIR, f"frame_{t:03d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-r", str(FPS),
        "-i", os.path.join(FRAME_DIR, "frame_%03d.png"),
        "-vf", "format=yuv420p",
        OUT_MP4,
    ], check=True)

    shutil.rmtree(FRAME_DIR)
    print(f"Done -> {OUT_MP4}  (left panel X={X_LEFT}, right panel X={X_RIGHT})")


if __name__ == "__main__":
    main()
