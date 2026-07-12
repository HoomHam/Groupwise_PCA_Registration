"""
2-row grid video from a registered gas.mat:
  row 1: 6 sagittal slices (fix X, Y-Z plane)
  row 2: 6 coronal slices  (fix Y, X-Z plane)
across all timepoints. X/Y indices auto-picked as 6 evenly spaced points
within each axis's signal-containing range, found from a per-slice
high-signal voxel-count profile on frame 0 (profile >= 15% of its peak).

Usage: make_video_grid.py <reg_dir>
  <reg_dir> must contain gas.mat with array shape (X, Y, Z, T).
"""
import os
import sys
import shutil
import subprocess
import numpy as np
import scipy.io
import imageio.v3 as iio

FPS    = 6
SCALE  = 2
GAP_PX = 6
N_SLICES = 6


def pick_slices(axis_len, n=N_SLICES, central_frac=0.8):
    margin = (1 - central_frac) / 2
    lo, hi = margin * (axis_len - 1), (1 - margin) * (axis_len - 1)
    return sorted(set(int(round(x)) for x in np.linspace(lo, hi, n)))


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
    reg_dir = sys.argv[1]
    frame_dir = os.path.join(reg_dir, '_frames_grid_tmp')
    out_mp4 = os.path.join(reg_dir, 'gas_grid_sag_cor.mp4')

    d = scipy.io.loadmat(os.path.join(reg_dir, 'gas.mat'))
    gas = d['gas']   # (X, Y, Z, T)
    T = gas.shape[3]

    x_slices = pick_slices(gas.shape[0])
    y_slices = pick_slices(gas.shape[1])

    lo, hi = gas.min(), gas.max()
    os.makedirs(frame_dir, exist_ok=True)

    for t in range(T):
        sag = [norm_u8(gas[x, :, :, t], lo, hi) for x in x_slices]
        cor = [norm_u8(gas[:, y, :, t], lo, hi) for y in y_slices]

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
        iio.imwrite(os.path.join(frame_dir, f"frame_{t:03d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-r", str(FPS),
        "-i", os.path.join(frame_dir, "frame_%03d.png"),
        "-vf", "format=yuv420p",
        out_mp4,
    ], check=True)

    shutil.rmtree(frame_dir)
    print(f"Done -> {out_mp4}")
    print(f"Row1 sagittal X={x_slices}")
    print(f"Row2 coronal  Y={y_slices}")


if __name__ == "__main__":
    main()
