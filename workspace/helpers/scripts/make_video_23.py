"""
Render the registered gas-phase channel (central axial slice, 16 timepoints)
from 23_oneshot as an mp4. PNG frames -> ffmpeg (no imageio-ffmpeg dep needed).
"""
import os
import shutil
import subprocess
import numpy as np
import scipy.io
import imageio.v3 as iio

REG_DIR   = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/outputs/registered/23_oneshot'
FRAME_DIR = os.path.join(REG_DIR, '_frames_tmp')
OUT_MP4   = os.path.join(REG_DIR, 'gas_axial.mp4')
FPS       = 6
SCALE     = 4   # upscale 100x100 -> 400x400 for visibility


def main():
    d = scipy.io.loadmat(os.path.join(REG_DIR, 'gas.mat'))
    gas = d['gas']   # (X, Y, Z, T)
    z_mid = gas.shape[2] // 2
    stack = gas[:, :, z_mid, :]   # (X, Y, T)

    lo, hi = stack.min(), stack.max()
    norm = ((stack - lo) / (hi - lo) * 255).astype(np.uint8)

    os.makedirs(FRAME_DIR, exist_ok=True)
    T = norm.shape[2]
    for t in range(T):
        frame = np.kron(norm[:, :, t], np.ones((SCALE, SCALE), dtype=np.uint8))
        iio.imwrite(os.path.join(FRAME_DIR, f"frame_{t:03d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-r", str(FPS),
        "-i", os.path.join(FRAME_DIR, "frame_%03d.png"),
        "-vf", "format=yuv420p",
        OUT_MP4,
    ], check=True)

    shutil.rmtree(FRAME_DIR)
    print(f"Done -> {OUT_MP4}")


if __name__ == "__main__":
    main()
