# region Import Packages
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
# endregion

BOHR_TO_ANGSTROM = 0.529177210903

# region User Parameters
N_CORES = 8
RESOLUTION = 0
VERBOSE = 1
XYZ_UNITS = "bohr"
ANGULAR_L_MAX = 7


DB:str = "QM7"
RHO_PATH = f"sample/{DB}/RCD_3.0_300.npy"
MOL_PATH = f"sample/{DB}/{DB}.xyz"
OUTPUT_FILE = f"sample/{DB}/{DB}_overlap_leg{ANGULAR_L_MAX}.npy"



DERIVATIVES = False
NORMALIZE_PER_ATOM = False

CUTOFF = 3.0
DR = 3.0 / 300
# endregion


# region Class Definitions
@dataclass(frozen=True)
class Molecule:
    atom_ids: np.ndarray
    positions: np.ndarray
# endregion


# region Function Definitions
def getZ(symbol: str) -> int:
    """
    Returns the atomic number for a_ev given atomic symbol, up to Z = 102, Nobelium.
    :param symbol: ``str`` representation of the atomic symbol.
    :return: ``int`` atomic number corresponding to ``symbol``
    """
    if symbol == "":
        raise ValueError("Label must not be empty.")

    elements: list[str] = (
        "H  He "
        "Li  Be  B   C   N   O   F   Ne "
        "Na  Mg  Al  Si  P   S   Cl  Ar "
        "K   Ca  Sc  Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr "
        "Rb  Sr  Y   Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe "
        "Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb "
        "Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn "
        "Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No".split()
    )

    try:
        return elements.index(symbol) + 1
    except ValueError as exc:
        raise ValueError(f"Unsupported atomic symbol: {symbol!r}") from exc


def getDensity(Z: int, data: np.ndarray) -> np.ndarray:
    if Z < 1 or Z > len(data):
        raise IndexError(f"Atomic number Z={Z} is out of bounds for density array of length {len(data)}.")
    return data[Z - 1]


def resolve_output_path(path: str) -> str:
    path = path.strip()
    if path == "":
        raise ValueError("`OUTPUT_FILE` must not be empty.")
    return path if path.endswith(".npy") else f"{path}.npy"


def validate_settings() -> None:
    if N_CORES <= 0:
        raise ValueError(f"`N_CORES` must be positive, got {N_CORES}.")
    if RESOLUTION < 0:
        raise ValueError(f"`RESOLUTION` must be >= 0, got {RESOLUTION}.")
    if XYZ_UNITS not in {"angstrom", "bohr"}:
        raise ValueError(f"`XYZ_UNITS` must be either 'angstrom' or 'bohr', got {XYZ_UNITS!r}.")
    if not os.path.exists(RHO_PATH):
        raise FileNotFoundError(f"Density file not found: {RHO_PATH}")
    if not os.path.exists(MOL_PATH):
        raise FileNotFoundError(f"XYZ file not found: {MOL_PATH}")


def validate_density_array(rhos: np.ndarray) -> None:
    if not isinstance(rhos, np.ndarray):
        raise TypeError("Loaded densities must be a_ev NumPy array.")
    if rhos.ndim != 2:
        raise ValueError(f"Density array must be 2D, got shape {rhos.shape}.")
    if rhos.shape[0] == 0 or rhos.shape[1] < 2:
        raise ValueError(f"Density array shape is invalid: {rhos.shape}.")
    if not np.isfinite(rhos).all():
        raise ValueError("Density array contains non-finite values.")


def getMols(filepath: str, xyz_units: str = "angstrom") -> list[Molecule]:
    with open(filepath, "r", encoding="utf-8") as f:
        structures = f.readlines()

    scale = BOHR_TO_ANGSTROM if xyz_units == "bohr" else 1.0
    molecules: list[Molecule] = []

    line = 0
    n_lines = len(structures)

    while line < n_lines:
        header = structures[line].strip()
        if header == "":
            line += 1
            continue

        parts = header.split()
        if len(parts) != 1:
            raise ValueError(
                f"Expected atom-count line with one field at line {line + 1}, got: {header!r}"
            )

        try:
            n_atoms = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"Expected integer atom count at line {line + 1}, got: {header!r}"
            ) from exc

        if n_atoms <= 0:
            raise ValueError(f"Atom count must be positive at line {line + 1}, got {n_atoms}.")
        if line + 1 >= n_lines:
            raise ValueError(f"Missing comment line after atom count at line {line + 1}.")
        if line + 2 + n_atoms > n_lines:
            raise ValueError(
                f"File ends early for structure starting at line {line + 1}; expected {n_atoms} atom lines."
            )

        Zs = np.empty(n_atoms, dtype=int)
        xyzs = np.empty((n_atoms, 3), dtype=float)

        for atom_index in range(n_atoms):
            atom_line_num = line + 2 + atom_index
            atom_data = structures[atom_line_num].split()
            if len(atom_data) < 4:
                raise ValueError(
                    f"Malformed atom line {atom_line_num + 1}: expected symbol and 3 coordinates, got {atom_data!r}"
                )

            Zs[atom_index] = getZ(atom_data[0])
            try:
                xyzs[atom_index] = np.array(atom_data[1:4], dtype=float) * scale
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse coordinates on line {atom_line_num + 1}: {atom_data[1:4]!r}"
                ) from exc

        molecules.append(Molecule(atom_ids=Zs, positions=xyzs))
        line += n_atoms + 2

    return molecules


def get_channel_length(N: int, resolution: int) -> int:
    return 2 * N if resolution == 0 else resolution


def build_symmetric_densities(rhos: np.ndarray) -> np.ndarray:
    return np.concatenate((rhos[:, ::-1], rhos), axis=1) / 2.0


def get_region_names(derivatives: bool, angular_l_max: int) -> list[str]:
    n_ang = angular_l_max + 1

    region_names = [
        "i",
        "ij_sym",
        "ij_antisym",
    ]
    region_names.extend([f"ijk_sym_P{ell}" for ell in range(n_ang)])
    region_names.extend([f"ijk_antisym_P{ell}" for ell in range(n_ang)])

    if derivatives:
        deriv_names = [f"D_{name}" for name in region_names]
        region_names.extend(deriv_names)

    return region_names


def legendre_values(x: float, l_max: int) -> np.ndarray:
    """
    Return [P0(x), P1(x), ..., P_lmax(x)] using the standard recurrence.
    """
    x = float(np.clip(x, -1.0, 1.0))

    vals = np.empty(l_max + 1, dtype=float)
    vals[0] = 1.0
    if l_max == 0:
        return vals

    vals[1] = x
    for ell in range(1, l_max):
        vals[ell + 1] = ((2 * ell + 1) * x * vals[ell] - ell * vals[ell - 1]) / (ell + 1)

    return vals


def generateDescriptor(
    mol: Molecule,
    distribution: np.ndarray,
    N: int,
    cutoff: float,
    dR: float,
    derivatives: bool,
    normalize_per_atom: bool,
    resolution: int = 0,
    angular_l_max: int = 3,
) -> np.ndarray:
    Z = mol.atom_ids
    R = mol.positions
    nAtoms = len(Z)

    full_channel_len = 2 * N
    n_ang = angular_l_max + 1

    radial_grid = np.arange(N, dtype=float) * dR

    density_cache = {int(z): getDensity(int(z), distribution) for z in np.unique(Z)}
    positive_tail_cache = {z: density_cache[z][N:] for z in density_cache}

    deltas = R[:, None, :] - R[None, :, :]
    distance_matrix = np.linalg.norm(deltas, axis=2)

    i_FP = np.zeros(full_channel_len, dtype=float)
    ij_sym_FP = np.zeros(full_channel_len, dtype=float)
    ij_antisym_FP = np.zeros(full_channel_len, dtype=float)

    # One angular 3-body channel per Legendre order
    ijk_sym_FP = np.zeros((n_ang, full_channel_len), dtype=float)
    ijk_antisym_FP = np.zeros((n_ang, full_channel_len), dtype=float)

    for i in range(nAtoms):
        Zi = int(Z[i])
        rho_i = density_cache[Zi]
        i_FP += rho_i

        for j in range(i + 1, nAtoms):
            Zj = int(Z[j])

            R_ij = R[j] - R[i]
            d_ij = distance_matrix[i, j]

            if d_ij <= 0.0:
                continue
            if d_ij > 2.0 * cutoff:
                continue

            N_ij = int(d_ij / dR)
            if N_ij >= full_channel_len:
                continue

            rho_j = density_cache[Zj]

            # Ordered pair overlap profile, same idea as your current implementation
            ij_overlap = rho_i[N_ij:] * rho_j[: full_channel_len - N_ij]

            base_sym = 0.5 * (ij_overlap + ij_overlap[::-1])
            base_asym = 0.5 * np.abs(ij_overlap - ij_overlap[::-1])

            ij_sym_FP[N_ij:] += base_sym
            ij_antisym_FP[N_ij:] += base_asym

            # Accumulate angular coefficients for this pair:
            # coeffs[ell] = sum_k rho_k(d_ik) * P_ell(cos theta_jik)
            ang_coeffs = np.zeros(n_ang, dtype=float)

            for k in range(nAtoms):
                if k == i or k == j:
                    continue

                Zk = int(Z[k])

                d_ik = distance_matrix[i, k]
                d_jk = distance_matrix[j, k]

                # Keep a_ev locality screen similar to your current 3-body logic
                if d_ik <= 0.0:
                    continue
                if d_ik >= 2.0 * cutoff or d_jk >= 2.0 * cutoff:
                    continue

                R_ik = R[k] - R[i]

                cos_theta = np.dot(R_ij, R_ik) / (d_ij * d_ik)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)

                # Third-atom radial weight: rho_k(d_ik)
                rho_k_pos = positive_tail_cache[Zk]
                wk = np.interp(
                    d_ik,
                    radial_grid,
                    rho_k_pos,
                    left=rho_k_pos[0],
                    right=0.0,
                )

                if wk == 0.0:
                    continue

                ang_coeffs += wk * legendre_values(cos_theta, angular_l_max)

            # Convert the angular coefficients into overlap profiles
            for ell, coeff in enumerate(ang_coeffs):
                if coeff == 0.0:
                    continue

                # Symmetric channel keeps the Legendre sign
                ijk_sym_FP[ell, N_ij:] += coeff * base_sym

                # Antisymmetric channel mirrors your current abs-difference philosophy
                ijk_antisym_FP[ell, N_ij:] += abs(coeff) * base_asym

    descriptor_blocks = [
        i_FP,
        ij_sym_FP,
        ij_antisym_FP,
        *[ijk_sym_FP[ell] for ell in range(n_ang)],
        *[ijk_antisym_FP[ell] for ell in range(n_ang)],
    ]

    if derivatives:
        descriptor_blocks = descriptor_blocks + [np.gradient(block, dR) for block in descriptor_blocks]

    norm_factor = float(nAtoms) if normalize_per_atom else 1.0

    if resolution == 0:
        return np.asarray(descriptor_blocks, dtype=float).ravel() / norm_factor

    if resolution <= 0:
        raise ValueError(f"`resolution` must be 0 or a_ev positive integer, got {resolution}.")
    if full_channel_len % resolution != 0:
        raise ValueError(
            f"`RESOLUTION` ({resolution}) must divide the channel length ({full_channel_len})."
        )

    bin_size = full_channel_len // resolution
    rebinned_blocks = [block.reshape(resolution, bin_size).sum(axis=1) for block in descriptor_blocks]
    return np.asarray(rebinned_blocks, dtype=float).ravel() / norm_factor


def compute_descriptor(
    mol: Molecule,
    distribution: np.ndarray,
    N: int,
    cutoff: float,
    dR: float,
    resolution: int,
    derivatives: bool,
    normalize_per_atom: bool,
    angular_l_max: int,
) -> np.ndarray:
    return generateDescriptor(
        mol=mol,
        distribution=distribution,
        N=N,
        cutoff=cutoff,
        dR=dR,
        derivatives=derivatives,
        normalize_per_atom=normalize_per_atom,
        resolution=resolution,
        angular_l_max=angular_l_max,
    )


def save_region_slices(
    descriptors: np.ndarray,
    base_output_path: str,
    channel_length: int,
    derivatives: bool,
    verbose: int,
) -> None:
    if descriptors.ndim != 2:
        raise ValueError(f"`descriptors` must be 2D, got shape {descriptors.shape}.")

    region_names = get_region_names(derivatives=derivatives, angular_l_max=ANGULAR_L_MAX)
    n_regions = len(region_names)

    if descriptors.shape[1] != n_regions * channel_length:
        raise ValueError(
            f"Descriptor width ({descriptors.shape[1]}) does not match expected n_regions * channel_length "
            f"({n_regions} * {channel_length} = {n_regions * channel_length})."
        )

    output_path = resolve_output_path(base_output_path)
    base = output_path[:-4]

    if verbose > 0:
        print(f"Saving {n_regions} descriptor regions.")

    for idx, region_name in enumerate(region_names):
        region_slice = descriptors[:, idx * channel_length : (idx + 1) * channel_length]
        np.save(file=f"{base}_{idx}.npy", arr=region_slice)

        if verbose > 1:
            print(f"  region {idx}: {region_name} -> {base}_{idx}.npy")


def main() -> None:
    validate_settings()

    rhos = np.load(RHO_PATH)
    validate_density_array(rhos)

    mols = getMols(MOL_PATH, xyz_units=XYZ_UNITS)
    if len(mols) == 0:
        raise ValueError("No molecules were parsed from the XYZ file.")

    max_Z = int(max(np.max(mol.atom_ids) for mol in mols))
    if max_Z > len(rhos):
        raise ValueError(
            f"Density array only contains {len(rhos)} elements, but molecule data requires Z={max_Z}."
        )

    N = rhos.shape[1]
    if VERBOSE > 0:
        print(f"Cutoff: {CUTOFF} A")
        print(f"dr: {DR} A")

    rhos_sym = build_symmetric_densities(rhos)
    channel_length = get_channel_length(N=N, resolution=RESOLUTION)
    output_path = resolve_output_path(OUTPUT_FILE)

    start_time = time.time()

    descriptors = Parallel(n_jobs=N_CORES, prefer="processes")(
        delayed(compute_descriptor)(
            mol=mol,
            distribution=rhos_sym,
            N=N,
            cutoff=CUTOFF,
            dR=DR,
            resolution=RESOLUTION,
            derivatives=DERIVATIVES,
            normalize_per_atom=NORMALIZE_PER_ATOM,
            angular_l_max=ANGULAR_L_MAX,
        )
        for mol in mols
    )

    duration = time.time() - start_time
    descriptors = np.asarray(descriptors, dtype=float)

    np.save(file=output_path, arr=descriptors)
    save_region_slices(
        descriptors=descriptors,
        base_output_path=output_path,
        channel_length=channel_length,
        derivatives=DERIVATIVES,
        verbose=VERBOSE,
    )

    if VERBOSE > 0:
        print(
            " ===========================================================\n",
            "RCD DESCRIPTORS COMPLETE\n",
            f"XYZ INPUT UNITS: {XYZ_UNITS} ({'converted to angstrom' if XYZ_UNITS == 'bohr' else 'no conversion'})\n",
            f"DESCRIPTOR ARRAY SHAPE: {descriptors.shape}\n",
            f"DESCRIPTORS WRITTEN TO {output_path}\n",
            f"{len(mols)} DESCRIPTORS COMPUTED IN {round(duration, 2)} SECONDS ({round(duration / 60.0, 2)} MINUTES).\n",
            f"{round(len(mols) / duration, 2)} DESCRIPTORS/SECOND ({round(3600 * len(mols) / duration, 2)} DESCRIPTORS/HOUR)\n",
            f"{round(len(mols) / duration / N_CORES, 2)} DESCRIPTORS/SECOND/CORE ({round(3600 * len(mols) / duration / N_CORES, 2)} DESCRIPTORS/HOUR/CORE)\n",
            "===========================================================\n",
        )


if __name__ == "__main__":
    main()
# endregion
