"""
2-row grid video from registered gas.mat:
  row 1: 6 sagittal slices (fix X, Y-Z plane)
  row 2: 6 coronal slices  (fix Y, X-Z plane)
across all 16 timepoints. X/Y indices evenly spaced across each axis's
lung-containing signal range (from a per-slice high-signal voxel-count
profile computed on frame 0).
"""
import os
import shutil
import subprocess
import numpy as np
import scipy.io
import imageio.v3 as iio

REG_DIR   = '/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/workspace/outputs/registered/23_oneshot'
FRAME_DIR = os.path.join(REG_DIR, '_frames_grid_tmp')
OUT_MP4   = os.path.join(REG_DIR, 'gas_grid_sag_cor.mp4')
FPS       = 6
SCALE     = 2
GAP_PX    = 6

X_SLICES = [14, 28, 43, 57, 72, 86]   # sagittal: fix X, plane (Y,Z)
Y_SLICES = [40, 49, 58, 68, 77, 86]   # coronal:  fix Y, plane (X,Z)


def norm_u8(img, lo, hi):
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def upscale(img, s=SCALE):
    return np.kron(img, np.ones((s, s), dtype=np.uint8))


def make_row(slices_2d, gap):
    tiles = [upscale(t) for t in slices_2d]
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        out = np.zeros((h, w), dtype=np.uint8)
        out[:t.shape[0], :t.shape[1]] = t
        padded.append(out)
    row = padded[0]
    for t in padded[1:]:
        row = np.concatenate([row, gap, t], axis=1)
    return row


def main():
    d = scipy.io.loadmat(os.path.join(REG_DIR, 'gas.mat'))
    gas = d['gas']   # (X, Y, Z, T)
    T = gas.shape[3]
    lo, hi = gas.min(), gas.max()

    os.makedirs(FRAME_DIR, exist_ok=True)

    for t in range(T):
        sag = [norm_u8(gas[x, :, :, t], lo, hi) for x in X_SLICES]     # (Y,Z)
        cor = [norm_u8(gas[:, y, :, t], lo, hi) for y in Y_SLICES]     # (X,Z)

        tile_w = upscale(sag[0]).shape[1]
        gap_col = np.full((upscale(sag[0]).shape[0], GAP_PX), 128, dtype=np.uint8)

        row_sag = make_row(sag, gap_col)
        row_cor = make_row(cor, gap_col)

        w = max(row_sag.shape[1], row_cor.shape[1])
        def pad_row(r):
            out = np.zeros((r.shape[0], w), dtype=np.uint8)
            out[:, :r.shape[1]] = r
            return out
        row_sag, row_cor = pad_row(row_sag), pad_row(row_cor)

        gap_row = np.full((GAP_PX, w), 128, dtype=np.uint8)
        frame = np.concatenate([row_sag, gap_row, row_cor], axis=0)
        iio.imwrite(os.path.join(FRAME_DIR, f"frame_{t:03d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-r", str(FPS),
        "-i", os.path.join(FRAME_DIR, "frame_%03d.png"),
        "-vf", "format=yuv420p",
        OUT_MP4,
    ], check=True)

    shutil.rmtree(FRAME_DIR)
    print(f"Done -> {OUT_MP4}")
    print(f"Row1 sagittal X={X_SLICES}")
    print(f"Row2 coronal  Y={Y_SLICES}")


if __name__ == "__main__":
    main()
