import numpy as np
from scipy.fft import dct, idct

# =========================
# Functions
# =========================

def dct_compress(X, n_keep, axis=1, dct_type=2, norm="ortho"):
    """
    Compress descriptors by applying a DCT along the feature axis
    and keeping the first n_keep low-frequency coefficients.

    Parameters
    ----------
    X : np.ndarray
        Descriptor array.
    n_keep : int
        Number of DCT coefficients to retain.
    axis : int
        Axis corresponding to descriptor features.
    dct_type : int
        DCT type.
    norm : str
        DCT normalization.

    Returns
    -------
    X_dct_keep : np.ndarray
        Compressed DCT coefficients.
    """
    X = np.asarray(X, dtype=np.float64)

    n_features = X.shape[axis]
    if n_keep > n_features:
        raise ValueError(
            f"n_keep={n_keep} cannot exceed descriptor length {n_features}."
        )

    X_dct = dct(X, type=dct_type, norm=norm, axis=axis)

    slicer = [slice(None)] * X_dct.ndim
    slicer[axis] = slice(0, n_keep)

    return X_dct[tuple(slicer)]


def dct_reconstruct(X_dct_keep, original_length, axis=1, dct_type=2, norm="ortho"):
    """
    Reconstruct approximate descriptors from truncated DCT coefficients.

    Parameters
    ----------
    X_dct_keep : np.ndarray
        Truncated DCT coefficients.
    original_length : int
        Original descriptor length before compression.
    axis : int
        Axis corresponding to descriptor features.
    dct_type : int
        DCT type.
    norm : str
        DCT normalization.

    Returns
    -------
    X_recon : np.ndarray
        Approximate reconstructed descriptors.
    """
    X_dct_keep = np.asarray(X_dct_keep, dtype=np.float64)

    n_keep = X_dct_keep.shape[axis]
    if n_keep > original_length:
        raise ValueError(
            f"Compressed length {n_keep} cannot exceed original length {original_length}."
        )

    full_shape = list(X_dct_keep.shape)
    full_shape[axis] = original_length

    X_dct_full = np.zeros(full_shape, dtype=np.float64)

    slicer = [slice(None)] * X_dct_keep.ndim
    slicer[axis] = slice(0, n_keep)

    X_dct_full[tuple(slicer)] = X_dct_keep

    return idct(X_dct_full, type=dct_type, norm=norm, axis=axis)


def relative_reconstruction_error(X, X_recon):
    """
    Compute relative Frobenius reconstruction error.
    """
    numerator = np.linalg.norm(X - X_recon)
    denominator = np.linalg.norm(X)

    if denominator == 0:
        return np.nan

    return numerator / denominator


# =========================
# Main script
# =========================

# =========================
# User settings
# =========================

N = 10
idx = [i for i in range(0, N+1)]
# Number of DCT coefficients to keep.
# Example: original descriptor length 600 -> compressed length 100
N_KEEP = 100

# Axis corresponding to the descriptor dimension.
# Usually:
#   X.shape = (n_samples, n_features) -> FEATURE_AXIS = 1
#   X.shape = (n_features,)          -> FEATURE_AXIS = 0
FEATURE_AXIS = 1

# DCT type. Type 2 with orthonormal scaling is the standard choice.
DCT_TYPE = 2
NORM = "ortho"

# Optional reconstruction check
DO_RECONSTRUCTION_CHECK = True

dcts = []
for i in idx:
    INPUT_NPY = f"sample/QM7/QM7_overlap_leg3_{i}.npy"
    OUTPUT_NPY = f"sample/QM7/QM7_overlap_leg3_dct_{i}.npy"
    X = np.load(INPUT_NPY)

    print(f"Loaded: {INPUT_NPY}")
    print(f"Original shape: {X.shape}")

    original_length = X.shape[FEATURE_AXIS]

    X_dct_keep = dct_compress(
        X,
        n_keep=N_KEEP,
        axis=FEATURE_AXIS,
        dct_type=DCT_TYPE,
        norm=NORM,
    )

    np.save(OUTPUT_NPY, X_dct_keep)
    dcts.append(X_dct_keep)

    print(f"Saved compressed descriptors to: {OUTPUT_NPY}")
    print(f"Compressed shape: {X_dct_keep.shape}")

    if DO_RECONSTRUCTION_CHECK:
        X_recon = dct_reconstruct(
            X_dct_keep,
            original_length=original_length,
            axis=FEATURE_AXIS,
            dct_type=DCT_TYPE,
            norm=NORM,
        )

        rel_err = relative_reconstruction_error(X, X_recon)

        print(f"Relative reconstruction error: {rel_err:.6e}")


megadct = np.concatenate(dcts, axis=1)

print(np.shape(megadct))

np.save("sample/QM7/QM7_overlap_dct_concat.npy", megadct)
