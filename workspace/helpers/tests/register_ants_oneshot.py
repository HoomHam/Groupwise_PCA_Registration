import os
import time
import numpy as np
import SimpleITK as sitk
import scipy.io
import ants
from scipy.ndimage import label

# ── Config ───────────────────────────────────────────────────────────────────
input_files = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/'
save_base   = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/'

# Single-stage: raw gas image only (no CLAHE, no enhance, no denoise)
run_flags = (0, 0, 0)
pad       = False

# Metric options
METRIC_OPTIONS = {
    '1': ('SyNCC', 'CC',  'Local normalized cross-correlation — sharper, assumes local linearity'),
    '2': ('SyN',   'MI',  'Mattes mutual information — no linearity assumption, better at defect boundaries'),
}

THRESHOLD      = 0.04
TIME_THRESHOLD = 5
# ─────────────────────────────────────────────────────────────────────────────


# ── I/O helpers ──────────────────────────────────────────────────────────────

def find_mat(directory, prefix):
    for fn in os.listdir(directory):
        if fn.startswith(prefix) and fn.endswith('.mat'):
            return os.path.join(directory, fn)
    raise FileNotFoundError(f"No .mat starting with '{prefix}' in {directory}")


def read_files(directory):
    def load_mat(path, key):
        data = scipy.io.loadmat(path)[key]   # (X, Y, Z, T)
        vec = sitk.VectorOfImage()
        for t in range(data.shape[3]):
            arr = np.transpose(data[:, :, :, t], (2, 1, 0)).astype(np.float32)
            vec.push_back(sitk.GetImageFromArray(arr))
        return sitk.JoinSeries(vec)

    image4d = load_mat(find_mat(directory, 'rspace'), 'image')

    cplx = scipy.io.loadmat(find_mat(directory, 'dspace'))['dspace']
    vr, vm = sitk.VectorOfImage(), sitk.VectorOfImage()
    for t in range(cplx.shape[3]):
        arr = np.transpose(cplx[:, :, :, t], (2, 1, 0))
        vr.push_back(sitk.GetImageFromArray(np.real(arr).astype(np.float32)))
        vm.push_back(sitk.GetImageFromArray(np.imag(arr).astype(np.float32)))
    dp_rbc4d = sitk.JoinSeries(vr)
    dp_mem4d = sitk.JoinSeries(vm)

    return image4d, dp_rbc4d, dp_mem4d


def extract_3d_frames(image4d):
    n = image4d.GetSize()[3]
    size3 = list(image4d.GetSize()[:3]) + [0]
    return [sitk.Extract(image4d, size3, [0, 0, 0, t]) for t in range(n)]


def join_3d_to_4d(frames):
    vec = sitk.VectorOfImage()
    for f in frames:
        vec.push_back(f)
    return sitk.JoinSeries(vec)


def save_channels(out_dir, **channels):
    os.makedirs(out_dir, exist_ok=True)
    for name, img in channels.items():
        if img is None:
            continue
        sitk.WriteImage(img, os.path.join(out_dir, f"{name}.nii"))
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 4:
            arr = np.transpose(arr, (3, 2, 1, 0))
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        scipy.io.savemat(
            os.path.join(out_dir, f"{name}.mat"),
            {name: arr},
            do_compression=True,
        )


def _save_mat(out_dir, name, arr):
    if arr.ndim == 4:
        mat_arr = np.transpose(arr, (3, 2, 1, 0))
    elif arr.ndim == 3:
        mat_arr = np.transpose(arr, (2, 1, 0))
    else:
        mat_arr = arr
    scipy.io.savemat(
        os.path.join(out_dir, f"{name}.mat"),
        {name: mat_arr},
        do_compression=True,
    )


# ── Mask ─────────────────────────────────────────────────────────────────────

def compute_mask(image3D, threshold=THRESHOLD, time_threshold=TIME_THRESHOLD):
    """Binary lung mask from 3D frame list: threshold + largest connected component."""
    arr = np.stack([sitk.GetArrayFromImage(f) for f in image3D], axis=0)  # (T, Z, Y, X)
    bin4d = (arr >= threshold).astype(np.uint8)
    summed = bin4d.sum(axis=0)
    mask3d = (summed >= time_threshold).astype(np.uint8)
    labels, _ = label(mask3d)
    sizes = np.bincount(labels.ravel())
    if sizes.size > 1:
        largest = np.argmax(sizes[1:]) + 1
        mask3d[labels != largest] = 0
    return mask3d.astype(np.uint8)


# ── Conversion helpers ───────────────────────────────────────────────────────

def sitk_to_ants(img):
    return ants.from_numpy(sitk.GetArrayFromImage(img).astype(np.float32))


def ants_to_sitk(img_ants, ref_sitk):
    out = sitk.GetImageFromArray(img_ants.numpy())
    out.SetSpacing(ref_sitk.GetSpacing())
    out.SetOrigin(ref_sitk.GetOrigin())
    out.SetDirection(ref_sitk.GetDirection())
    return out


def cleanup_ants_files(path_list):
    for path in (path_list or []):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


# ── EI detection ─────────────────────────────────────────────────────────────

def find_ei_index(image3D):
    sums = [float(np.sum(sitk.GetArrayFromImage(f))) for f in image3D]
    ei = int(np.argmax(sums))
    print(f"  Signal sums: {[f'{s:.1f}' for s in sums]}")
    print(f"  → EI auto-detected: frame {ei}")
    return ei


# ── Per-frame ANTs SyNCC registration ────────────────────────────────────────

def _register_single_frame(
    t, cur_img_sitk, cur_rbc_sitk, cur_mem_sitk,
    fixed_img_ants, ants_transform_type,
):
    """
    Register one frame to EI with single-stage ANTs (gas image only).
    ants_transform_type: 'SyNCC' (CC metric) or 'SyN' (Mattes MI metric).
    Returns (img_sitk, rbc_sitk, mem_sitk, jac_arr_ZYX).
    Jacobian is computed from the warp field before transform cleanup.
    """
    a_img = sitk_to_ants(cur_img_sitk)
    a_rbc = sitk_to_ants(cur_rbc_sitk)
    a_mem = sitk_to_ants(cur_mem_sitk)

    reg = ants.registration(
        fixed=fixed_img_ants, moving=a_img, type_of_transform=ants_transform_type,
    )
    fwd = reg['fwdtransforms']   # [Warp.nii.gz, GenericAffine.mat] for SyNCC

    a_img = ants.apply_transforms(fixed=fixed_img_ants, moving=a_img, transformlist=fwd)
    a_rbc = ants.apply_transforms(fixed=fixed_img_ants, moving=a_rbc, transformlist=fwd)
    a_mem = ants.apply_transforms(fixed=fixed_img_ants, moving=a_mem, transformlist=fwd)

    # Jacobian from warp field (deformable component) — before cleanup
    warp_file = fwd[0]
    jac_arr = None
    if os.path.isfile(warp_file):
        jac_ants = ants.create_jacobian_determinant_image(
            domain_image=fixed_img_ants,
            tx=warp_file,
            do_log=False,
            geom=True,
        )
        jac_arr = jac_ants.numpy().astype(np.float32)

    cleanup_ants_files(fwd)
    cleanup_ants_files(reg.get('invtransforms', []))

    def back(a, ref):
        return ants_to_sitk(a, ref)

    return (
        back(a_img, cur_img_sitk),
        back(a_rbc, cur_rbc_sitk),
        back(a_mem, cur_mem_sitk),
        jac_arr,
    )


def register_ants_to_ei(image3D, dp_rbc3D, dp_mem3D, refno, ants_transform_type):
    """
    Register all frames to EI using single-stage ANTs (gas image only).
    ants_transform_type: 'SyNCC' or 'SyN'.
    EI frame passes through unchanged with identity Jacobian (all ones).
    Returns (out_img, out_rbc, out_mem, jac_list) where jac_list[t] is (Z,Y,X) numpy.
    """
    print(f"  Single-stage ANTs {ants_transform_type} to EI (frame {refno})")

    fixed_img_ants = sitk_to_ants(image3D[refno])
    ref_shape = sitk.GetArrayFromImage(image3D[refno]).shape   # (Z, Y, X)

    out_img, out_rbc, out_mem = [], [], []
    jac_list = []

    for t in range(len(image3D)):
        print(f"  Frame {t+1:2d}/{len(image3D)} (t={t})", flush=True)

        if t == refno:
            out_img.append(image3D[t])
            out_rbc.append(dp_rbc3D[t])
            out_mem.append(dp_mem3D[t])
            jac_list.append(np.ones(ref_shape, dtype=np.float32))
            continue

        r_img, r_rbc, r_mem, jac_arr = _register_single_frame(
            t,
            image3D[t], dp_rbc3D[t], dp_mem3D[t],
            fixed_img_ants, ants_transform_type,
        )
        out_img.append(r_img)
        out_rbc.append(r_rbc)
        out_mem.append(r_mem)
        jac_list.append(jac_arr if jac_arr is not None else np.ones(ref_shape, dtype=np.float32))

    return out_img, out_rbc, out_mem, jac_list


# ── Main ─────────────────────────────────────────────────────────────────────

def _choose_metric():
    print("\nChoose ANTs registration metric:")
    for key, (transform, label, desc) in METRIC_OPTIONS.items():
        print(f"  [{key}] {label} ({transform}) — {desc}")
    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice in METRIC_OPTIONS:
            transform, label, desc = METRIC_OPTIONS[choice]
            print(f"  → Using {label} ({transform})\n")
            return transform, label
        print("  Invalid. Enter 1 or 2.")


def main():
    transform_type, metric_label = _choose_metric()
    save_dir = os.path.join(save_base, f'ants_oneshot_gas_{metric_label}/')

    print("Loading files...")
    image4d, dp_rbc4d, dp_mem4d = read_files(input_files)

    image3D  = extract_3d_frames(image4d)
    dp_rbc3D = extract_3d_frames(dp_rbc4d)
    dp_mem3D = extract_3d_frames(dp_mem4d)
    print(f"Loaded {len(image3D)} frames.")

    print("Detecting EI...")
    ei = find_ei_index(image3D)

    os.makedirs(save_dir, exist_ok=True)

    print("Saving registration mask...")
    mask3d = compute_mask(image3D)
    mask_img = sitk.GetImageFromArray(mask3d)
    mask_img.SetSpacing(image3D[0].GetSpacing())
    mask_img.SetOrigin(image3D[0].GetOrigin())
    sitk.WriteImage(mask_img, os.path.join(save_dir, 'mask.nii'))
    _save_mat(save_dir, 'mask', mask3d)

    print(f"Starting ANTs {transform_type} one-shot registration to EI (frame {ei})...")
    t0 = time.time()
    out_img, out_rbc, out_mem, jac_list = register_ants_to_ei(
        image3D, dp_rbc3D, dp_mem3D, refno=ei,
        ants_transform_type=transform_type,
    )
    print(f"Registration done in {(time.time()-t0)/60:.1f} min.")

    # Assemble Jacobian 4D array (T, Z, Y, X)
    jac_arr = np.stack(jac_list, axis=0).astype(np.float32)  # (16, Z, Y, X)
    # jac_arr[ei] = 1 (identity); all others are deformation Jacobians relative to EI
    # jac_from_EI = jac_arr (EI is the reference — Jacobian of transform TO EI)
    jac_cons = jac_arr[1:] / jac_arr[:-1]           # (15, Z, Y, X): bin t → t+1
    jac_EE   = jac_arr / jac_arr[0:1]               # (16, Z, Y, X): relative to EE bin 0

    print("Saving...")
    save_channels(
        save_dir,
        gas=join_3d_to_4d(out_img),
        rbc=join_3d_to_4d(out_rbc),
        mem=join_3d_to_4d(out_mem),
    )
    # Jacobians — jac_det here is jac_from_EI (EI = reference frame)
    _save_mat(save_dir, 'jac_det',     jac_arr)
    _save_mat(save_dir, 'jac_cons',    jac_cons.astype(np.float32))
    _save_mat(save_dir, 'jac_from_EE', jac_EE.astype(np.float32))

    print(f"Done → {save_dir}")
    print(f"EI frame: {ei}  |  jac_det (=jac_from_EI): {jac_arr.shape}  |  jac_cons: {jac_cons.shape}  |  jac_from_EE: {jac_EE.shape}")


if __name__ == "__main__":
    main()
