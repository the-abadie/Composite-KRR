import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from ase.io import iread
from collections import defaultdict


# =========================
# User settings
# =========================

XYZ_PATH = "sample/QM7b/QM7b.xyz"
MAT_PATH = "sample/QM7b/QM7b.mat"

MAX_ATOMS = 23

# QM7b.mat Coulomb off-diagonals appear to use Bohr distances.
# XYZ coordinates are normally in Angstrom.
#
# C_bohr = Z_i Z_j / r_bohr
# C_ang  = Z_i Z_j / r_ang
# r_ang  = BOHR_TO_ANGSTROM * r_bohr
# Therefore:
# C_ang = C_bohr / BOHR_TO_ANGSTROM
BOHR_TO_ANGSTROM = 0.529177210903

GOOD_RMS_TOL = 1e-4
GOOD_MAX_TOL = 1e-3

PRINT_WORST_N = 30

# Output files
MAPPING_ALL_CSV = "sample/QM7b/qm7b_xyz_to_mat_mapping_all.csv"
MAPPING_GOOD_CSV = "sample/QM7b/qm7b_xyz_to_mat_mapping_good.csv"
MAPPING_BAD_CSV = "sample/QM7b/qm7b_xyz_to_mat_mapping_bad.csv"

GOOD_XYZ_INDICES_NPY = "sample/QM7b/qm7b_good_xyz_indices.npy"
GOOD_MAT_INDICES_NPY = "sample/QM7b/qm7b_good_mat_indices.npy"
TARGETS_GOOD_NPY = "sample/QM7b/qm7b_T_good_xyz_order.npy"

SAVE_TARGET_NAMES = True
TARGET_NAMES_TXT = "sample/QM7b/qm7b_target_names.txt"


# =========================
# Loading helpers
# =========================

def load_qm7b_mat(path):
    mat = loadmat(path)

    if "X" not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Could not find key 'X'. Available keys: {available}")

    if "T" not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Could not find key 'T'. Available keys: {available}")

    X = np.asarray(mat["X"], dtype=float)
    T = np.asarray(mat["T"], dtype=float)

    # Common layout is (7211, 23, 23), but handle possible MATLAB layout.
    if X.shape == (23, 23, 7211):
        X = np.moveaxis(X, -1, 0)

    if X.ndim != 3 or X.shape[1:] != (23, 23):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    # Common layout is (7211, 14), but handle transposed target matrix.
    if T.shape[0] != X.shape[0] and T.shape[-1] == X.shape[0]:
        T = T.T

    if T.shape[0] != X.shape[0]:
        raise ValueError(f"X and T disagree: X={X.shape}, T={T.shape}")

    names = mat.get("names", None)

    return X, T, names


def decode_matlab_names(names):
    """
    Best-effort decoder for the QM7b names array.

    Different scipy/MATLAB versions may load the char array differently.
    This is only for convenience; matching does not depend on it.
    """
    if names is None:
        return None

    arr = np.asarray(names)

    try:
        if arr.dtype.kind in {"U", "S"}:
            flat = arr.ravel()
            joined = "".join(str(x) for x in flat)
            return joined
    except Exception:
        pass

    try:
        return str(arr)
    except Exception:
        return None


# =========================
# Coulomb matrix helpers
# =========================

def convert_mat_coulomb_bohr_to_angstrom(X):
    """
    Convert QM7b.mat Coulomb matrices from Bohr-distance convention to
    Angstrom-distance convention.

    Only off-diagonal entries are converted. Diagonal entries are elemental:
        C_ii = 0.5 * Z_i ** 2.4
    and must not be scaled.
    """
    X_ang = np.array(X, dtype=float, copy=True)

    n, m1, m2 = X_ang.shape
    if m1 != m2:
        raise ValueError(f"Expected square matrices, got {X_ang.shape}")

    offdiag = ~np.eye(m1, dtype=bool)
    X_ang[:, offdiag] /= BOHR_TO_ANGSTROM

    return X_ang


def sort_coulomb_matrix(C):
    """
    Sort rows/columns by descending row norm.
    """
    C = np.asarray(C, dtype=float)
    row_norms = np.linalg.norm(C, axis=1)
    order = np.argsort(-row_norms)
    return C[order][:, order]


def coulomb_matrix_from_atoms_angstrom(atoms, max_atoms=23):
    """
    Compute a padded sorted Coulomb matrix from ASE Atoms.

    Assumes atoms.positions are in Angstrom.

    Diagonal:
        C_ii = 0.5 * Z_i ** 2.4

    Off-diagonal:
        C_ij = Z_i * Z_j / r_ij
    """
    numbers = np.asarray(atoms.numbers, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)

    n_atoms = len(numbers)

    if n_atoms > max_atoms:
        raise ValueError(f"Found molecule with {n_atoms} atoms > max_atoms={max_atoms}")

    C = np.zeros((max_atoms, max_atoms), dtype=float)

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    for i in range(n_atoms):
        C[i, i] = 0.5 * numbers[i] ** 2.4

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if dist[i, j] == 0:
                raise ValueError(f"Zero interatomic distance in molecule at atoms {i}, {j}")
            C[i, j] = numbers[i] * numbers[j] / dist[i, j]
            C[j, i] = C[i, j]

    return sort_coulomb_matrix(C)


def composition_key_from_atoms(atoms):
    return tuple(sorted(map(int, atoms.numbers), reverse=True))


def composition_key_from_coulomb(C, diag_tol=1e-8):
    """
    Recover composition from Coulomb diagonal values.
    """
    diag = np.diag(C)
    zs = []

    for value in diag:
        if abs(value) <= diag_tol:
            continue

        z = int(round((2.0 * value) ** (1.0 / 2.4)))
        expected = 0.5 * z ** 2.4

        if abs(expected - value) > 1e-4:
            raise ValueError(
                f"Could not recover atomic number from diagonal value {value}. "
                f"Nearest Z={z}, expected diagonal={expected}."
            )

        zs.append(z)

    return tuple(sorted(zs, reverse=True))


def active_offdiag_mask(n_atoms, max_atoms=23):
    mask = np.zeros((max_atoms, max_atoms), dtype=bool)
    mask[:n_atoms, :n_atoms] = True
    mask[np.eye(max_atoms, dtype=bool)] = False
    return mask


def pair_errors(C_xyz, C_mat_ang, n_atoms):
    """
    Compare two sorted Coulomb matrices in the same Angstrom convention.
    """
    full_diff = C_xyz - C_mat_ang

    mask = active_offdiag_mask(n_atoms, max_atoms=C_xyz.shape[0])
    offdiag_diff = full_diff[mask]

    return {
        "full_rms": float(np.sqrt(np.mean(full_diff**2))),
        "full_max": float(np.max(np.abs(full_diff))),
        "offdiag_rms": float(np.sqrt(np.mean(offdiag_diff**2))),
        "offdiag_max": float(np.max(np.abs(offdiag_diff))),
    }


# =========================
# Analysis helpers
# =========================

def same_index_analysis(C_xyz_all, C_mat_ang_all, molecules):
    max_abs_errors = []
    rms_errors = []
    diag_errors = []
    formulas = []
    natoms = []

    for i, atoms in enumerate(molecules):
        C_xyz = C_xyz_all[i]
        C_mat = C_mat_ang_all[i]

        diff = C_xyz - C_mat

        max_abs_errors.append(np.max(np.abs(diff)))
        rms_errors.append(np.sqrt(np.mean(diff**2)))

        d_xyz = np.sort(np.diag(C_xyz))[::-1]
        d_mat = np.sort(np.diag(C_mat))[::-1]
        diag_errors.append(np.max(np.abs(d_xyz - d_mat)))

        formulas.append(atoms.get_chemical_formula())
        natoms.append(len(atoms))

    max_abs_errors = np.asarray(max_abs_errors)
    rms_errors = np.asarray(rms_errors)
    diag_errors = np.asarray(diag_errors)
    natoms = np.asarray(natoms)

    print()
    print("Same-index comparison after MAT Angstrom conversion")
    print("---------------------------------------------------")
    print(f"Median full max abs error:       {np.median(max_abs_errors):.8e}")
    print(f"Mean full max abs error:         {np.mean(max_abs_errors):.8e}")
    print(f"Max full max abs error:          {np.max(max_abs_errors):.8e}")
    print(f"Median full RMS error:           {np.median(rms_errors):.8e}")
    print(f"Mean full RMS error:             {np.mean(rms_errors):.8e}")
    print(f"Max full RMS error:              {np.max(rms_errors):.8e}")
    print(f"Median diagonal-only error:      {np.median(diag_errors):.8e}")
    print(f"Max diagonal-only error:         {np.max(diag_errors):.8e}")
    print(f"Same-index composition matches:  {np.sum(diag_errors < 1e-6)} / {len(diag_errors)}")

    worst = np.argsort(-max_abs_errors)[:PRINT_WORST_N]

    print()
    print(f"Worst {PRINT_WORST_N} same-index pairs")
    print("--------------------------------------")
    print("xyz_idx  one_idx  formula         natoms  full_max_error    full_rms_error    diag_error")

    for i in worst:
        print(
            f"{i:7d}  {i + 1:7d}  "
            f"{formulas[i]:14s}  "
            f"{natoms[i]:6d}  "
            f"{max_abs_errors[i]:14.8e}  "
            f"{rms_errors[i]:14.8e}  "
            f"{diag_errors[i]:14.8e}"
        )


def build_composition_groups(molecules, C_mat_ang_all):
    xyz_groups = defaultdict(list)
    mat_groups = defaultdict(list)

    for i, atoms in enumerate(molecules):
        xyz_groups[composition_key_from_atoms(atoms)].append(i)

    for j, C in enumerate(C_mat_ang_all):
        mat_groups[composition_key_from_coulomb(C)].append(j)

    xyz_keys = set(xyz_groups)
    mat_keys = set(mat_groups)

    missing_in_mat = xyz_keys - mat_keys
    missing_in_xyz = mat_keys - xyz_keys

    if missing_in_mat or missing_in_xyz:
        print()
        print("Composition mismatch between XYZ and MAT")
        print("----------------------------------------")
        print(f"Compositions in XYZ but not MAT: {len(missing_in_mat)}")
        print(f"Compositions in MAT but not XYZ: {len(missing_in_xyz)}")

        if missing_in_mat:
            print("Examples in XYZ but not MAT:")
            for key in list(missing_in_mat)[:10]:
                print(f"  {key}: count_xyz={len(xyz_groups[key])}")

        if missing_in_xyz:
            print("Examples in MAT but not XYZ:")
            for key in list(missing_in_xyz)[:10]:
                print(f"  {key}: count_mat={len(mat_groups[key])}")

        raise RuntimeError("Cannot make reliable mapping due to composition mismatch.")

    for key in sorted(xyz_keys):
        if len(xyz_groups[key]) != len(mat_groups[key]):
            raise RuntimeError(
                f"Composition group count mismatch for key={key}: "
                f"XYZ={len(xyz_groups[key])}, MAT={len(mat_groups[key])}"
            )

    return xyz_groups, mat_groups


def match_within_composition_groups(C_xyz_all, C_mat_ang_all, molecules):
    xyz_groups, mat_groups = build_composition_groups(molecules, C_mat_ang_all)

    n_mols = len(molecules)

    xyz_to_mat = np.full(n_mols, -1, dtype=int)
    match_cost = np.full(n_mols, np.nan, dtype=float)
    group_size = np.full(n_mols, -1, dtype=int)

    full_rms = np.full(n_mols, np.nan, dtype=float)
    full_max = np.full(n_mols, np.nan, dtype=float)
    offdiag_rms = np.full(n_mols, np.nan, dtype=float)
    offdiag_max = np.full(n_mols, np.nan, dtype=float)

    sorted_keys = sorted(xyz_groups.keys())

    print()
    print("Solving one-to-one assignments within composition groups")
    print("--------------------------------------------------------")

    for group_counter, key in enumerate(sorted_keys, start=1):
        xyz_idx = np.asarray(xyz_groups[key], dtype=int)
        mat_idx = np.asarray(mat_groups[key], dtype=int)

        n_group = len(xyz_idx)
        n_atoms = len(key)

        cost_matrix = np.empty((n_group, n_group), dtype=float)

        mask = active_offdiag_mask(n_atoms, max_atoms=MAX_ATOMS)

        for r, i in enumerate(xyz_idx):
            C_xyz_off = C_xyz_all[i][mask]

            for c, j in enumerate(mat_idx):
                diff = C_xyz_off - C_mat_ang_all[j][mask]
                cost_matrix[r, c] = np.sqrt(np.mean(diff**2))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            i = xyz_idx[r]
            j = mat_idx[c]

            xyz_to_mat[i] = j
            match_cost[i] = cost_matrix[r, c]
            group_size[i] = n_group

            errs = pair_errors(C_xyz_all[i], C_mat_ang_all[j], n_atoms=n_atoms)

            full_rms[i] = errs["full_rms"]
            full_max[i] = errs["full_max"]
            offdiag_rms[i] = errs["offdiag_rms"]
            offdiag_max[i] = errs["offdiag_max"]

        if group_counter % 100 == 0:
            print(f"  processed {group_counter} composition groups")

    if np.any(xyz_to_mat < 0):
        raise RuntimeError("Some XYZ molecules were not assigned.")

    if len(np.unique(xyz_to_mat)) != n_mols:
        raise RuntimeError("Mapping is not one-to-one.")

    return {
        "xyz_to_mat": xyz_to_mat,
        "match_cost": match_cost,
        "group_size": group_size,
        "full_rms": full_rms,
        "full_max": full_max,
        "offdiag_rms": offdiag_rms,
        "offdiag_max": offdiag_max,
    }


def save_outputs(match, molecules, T, names=None):
    n_mols = len(molecules)

    xyz_idx = np.arange(n_mols)
    mat_idx = match["xyz_to_mat"]

    formulas = [atoms.get_chemical_formula() for atoms in molecules]
    natoms = [len(atoms) for atoms in molecules]

    good = (
        (match["offdiag_rms"] < GOOD_RMS_TOL)
        & (match["offdiag_max"] < GOOD_MAX_TOL)
    )

    df = pd.DataFrame(
        {
            "xyz_idx": xyz_idx,
            "xyz_one_idx": xyz_idx + 1,
            "mat_idx": mat_idx,
            "mat_one_idx": mat_idx + 1,
            "formula": formulas,
            "natoms": natoms,
            "group_size": match["group_size"],
            "match_cost_offdiag_rms": match["match_cost"],
            "full_rms": match["full_rms"],
            "full_max": match["full_max"],
            "offdiag_rms": match["offdiag_rms"],
            "offdiag_max": match["offdiag_max"],
            "good_match": good,
        }
    )

    df.to_csv(MAPPING_ALL_CSV, index=False)
    df.loc[good].to_csv(MAPPING_GOOD_CSV, index=False)
    df.loc[~good].to_csv(MAPPING_BAD_CSV, index=False)

    good_xyz_indices = xyz_idx[good]
    good_mat_indices = mat_idx[good]

    T_good_xyz_order = T[good_mat_indices]

    np.save(GOOD_XYZ_INDICES_NPY, good_xyz_indices)
    np.save(GOOD_MAT_INDICES_NPY, good_mat_indices)
    np.save(TARGETS_GOOD_NPY, T_good_xyz_order)

    if SAVE_TARGET_NAMES:
        decoded = decode_matlab_names(names)
        if decoded is not None:
            with open(TARGET_NAMES_TXT, "w") as f:
                f.write(decoded)
                f.write("\n")

    print()
    print("Saved outputs")
    print("-------------")
    print(f"All mapping rows:       {MAPPING_ALL_CSV}")
    print(f"Good mapping rows:      {MAPPING_GOOD_CSV}")
    print(f"Bad mapping rows:       {MAPPING_BAD_CSV}")
    print(f"Good XYZ indices:       {GOOD_XYZ_INDICES_NPY}")
    print(f"Good MAT indices:       {GOOD_MAT_INDICES_NPY}")
    print(f"Good reordered targets: {TARGETS_GOOD_NPY}")

    if SAVE_TARGET_NAMES and decoded is not None:
        print(f"Target names:           {TARGET_NAMES_TXT}")

    return df, good, T_good_xyz_order


def print_matching_summary(match, molecules, good):
    n_mols = len(molecules)
    xyz_to_mat = match["xyz_to_mat"]

    print()
    print("Matched comparison summary")
    print("--------------------------")
    print(f"Number of molecules:                {n_mols}")
    print(f"Unique MAT rows assigned:            {len(np.unique(xyz_to_mat))}")
    print(f"XYZ rows already at same MAT index:  {np.sum(xyz_to_mat == np.arange(n_mols))}")
    print(f"Good matches:                        {np.sum(good)} / {n_mols}")
    print(f"Good RMS tolerance:                  {GOOD_RMS_TOL:.1e}")
    print(f"Good max tolerance:                  {GOOD_MAX_TOL:.1e}")
    print()
    print(f"Median offdiag RMS:                  {np.median(match['offdiag_rms']):.8e}")
    print(f"Mean offdiag RMS:                    {np.mean(match['offdiag_rms']):.8e}")
    print(f"Max offdiag RMS:                     {np.max(match['offdiag_rms']):.8e}")
    print(f"Median offdiag max:                  {np.median(match['offdiag_max']):.8e}")
    print(f"Mean offdiag max:                    {np.mean(match['offdiag_max']):.8e}")
    print(f"Max offdiag max:                     {np.max(match['offdiag_max']):.8e}")
    print()
    print(f"Median full RMS:                     {np.median(match['full_rms']):.8e}")
    print(f"Mean full RMS:                       {np.mean(match['full_rms']):.8e}")
    print(f"Max full RMS:                        {np.max(match['full_rms']):.8e}")
    print(f"Median full max:                     {np.median(match['full_max']):.8e}")
    print(f"Mean full max:                       {np.mean(match['full_max']):.8e}")
    print(f"Max full max:                        {np.max(match['full_max']):.8e}")

    worst = np.argsort(-match["offdiag_rms"])[:PRINT_WORST_N]

    print()
    print(f"Worst {PRINT_WORST_N} matched pairs by offdiag RMS")
    print("--------------------------------------------------")
    print(
        "xyz_idx  xyz_one_idx  mat_idx  mat_one_idx  formula         "
        "natoms  group  offdiag_rms      offdiag_max      good"
    )

    for i in worst:
        atoms = molecules[i]
        print(
            f"{i:7d}  {i + 1:11d}  "
            f"{xyz_to_mat[i]:7d}  {xyz_to_mat[i] + 1:11d}  "
            f"{atoms.get_chemical_formula():14s}  "
            f"{len(atoms):6d}  "
            f"{match['group_size'][i]:5d}  "
            f"{match['offdiag_rms'][i]:14.8e}  "
            f"{match['offdiag_max'][i]:14.8e}  "
            f"{bool(good[i])}"
        )


# =========================
# Main
# =========================

def main():
    print("Loading data")
    print("------------")

    X_mat_bohr, T, names = load_qm7b_mat(MAT_PATH)
    molecules = list(iread(XYZ_PATH, index=":"))

    print(f"Loaded XYZ molecules:       {len(molecules)}")
    print(f"Loaded MAT Coulomb matrices: {X_mat_bohr.shape}")
    print(f"Loaded MAT targets T:        {T.shape}")

    if len(molecules) != X_mat_bohr.shape[0]:
        raise ValueError(
            f"XYZ molecule count ({len(molecules)}) does not match "
            f"MAT molecule count ({X_mat_bohr.shape[0]})"
        )

    print()
    print("Converting MAT Coulomb matrices to Angstrom convention")
    print("------------------------------------------------------")
    X_mat_ang = convert_mat_coulomb_bohr_to_angstrom(X_mat_bohr)
    print(f"Used BOHR_TO_ANGSTROM = {BOHR_TO_ANGSTROM:.12f}")
    print("Converted only off-diagonal entries; diagonals were left unchanged.")

    print()
    print("Computing sorted Coulomb matrices from XYZ")
    print("------------------------------------------")
    C_xyz_all = np.asarray(
        [coulomb_matrix_from_atoms_angstrom(atoms, max_atoms=MAX_ATOMS) for atoms in molecules]
    )

    print("Sorting converted MAT Coulomb matrices")
    print("--------------------------------------")
    C_mat_ang_all = np.asarray([sort_coulomb_matrix(C) for C in X_mat_ang])

    same_index_analysis(C_xyz_all, C_mat_ang_all, molecules)

    match = match_within_composition_groups(C_xyz_all, C_mat_ang_all, molecules)

    # Save first so the CSV exists even if later reporting is interrupted.
    df, good, T_good_xyz_order = save_outputs(match, molecules, T, names=names)

    print_matching_summary(match, molecules, good)

    print()
    print("Training alignment")
    print("------------------")
    print("For descriptors computed directly from QM7b.xyz in XYZ order:")
    print()
    print("    good_idx = np.load('qm7b_good_xyz_indices.npy')")
    print("    T_good = np.load('qm7b_T_good_xyz_order.npy')")
    print("    X_good = X_desc[good_idx]")
    print("    y_ae_pbe0 = T_good[:, 0]")
    print()
    print(f"T_good shape: {T_good_xyz_order.shape}")


if __name__ == "__main__":
    main()
