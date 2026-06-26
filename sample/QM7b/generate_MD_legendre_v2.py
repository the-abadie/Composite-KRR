# region Import Packages
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
# endregion

BOHR_TO_ANGSTROM = 0.529177210903


# ============================================================
# User Parameters
# ============================================================

# Parallelism / IO
N_CORES = 8
VERBOSE = 1

DB: str = "QM7"
RHO_PATH = "sample/QM7/RCD_3.0_300.npy"
MOL_PATH = f"sample/{DB}/{DB}.xyz"

# Units of the XYZ coordinates
XYZ_UNITS = "bohr"  # "angstrom" or "bohr"

# Descriptor radial settings
CUTOFF = 3.0
DR = 3.0 / 300
RESOLUTION = 0  # 0 keeps full 2*N channel length; otherwise must divide 2*N

# Descriptor options
NORMALIZE_PER_ATOM = False

# Angular basis options:
#   "legendre"  -> P_l(cos theta), l = 0 ... ANGULAR_L_MAX
#   "rbf_cos"   -> Gaussian RBFs centered in cos(theta) space, [-1, 1]
#   "rbf_theta" -> Gaussian RBFs centered in theta space, [0, pi]
ANGULAR_BASIS = "rbf_cos"

# Used only when ANGULAR_BASIS == "legendre"
ANGULAR_L_MAX = 5

# Used only when ANGULAR_BASIS is "rbf_cos" or "rbf_theta"
N_ANGULAR_RBF = 9

# For "rbf_cos", this width is in cos(theta) units.
# Reasonable values: 0.20, 0.25, 0.30, 0.40
ANGULAR_RBF_SIGMA_COS = 0.10

# For "rbf_theta", this width is in degrees.
# Reasonable values: 15.0, 20.0, 30.0
ANGULAR_RBF_SIGMA_DEG = 30.0

# If True, each third atom distributes the same total angular weight
# across nearby RBFs. If False, total contribution depends on basis overlap.
NORMALIZE_ANGULAR_RBF = False

# Output
OUTPUT_FILE = f"sample/{DB}/desc/{DB}_overlap_{ANGULAR_BASIS}13sharp.npy"


# ============================================================
# Class Definitions
# ============================================================

@dataclass(frozen=True)
class Molecule:
    atom_ids: np.ndarray
    positions: np.ndarray


# ============================================================
# Basic Utilities
# ============================================================

def getZ(symbol: str) -> int:
    """
    Returns the atomic number for a given atomic symbol, up to Z = 102, Nobelium.
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
        raise IndexError(
            f"Atomic number Z={Z} is out of bounds for density array of length {len(data)}."
        )
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
        raise ValueError(
            f"`XYZ_UNITS` must be either 'angstrom' or 'bohr', got {XYZ_UNITS!r}."
        )

    if ANGULAR_BASIS not in {"legendre", "rbf_cos", "rbf_theta"}:
        raise ValueError(
            "`ANGULAR_BASIS` must be one of: 'legendre', 'rbf_cos', 'rbf_theta'. "
            f"Got {ANGULAR_BASIS!r}."
        )

    if ANGULAR_L_MAX < 0:
        raise ValueError(f"`ANGULAR_L_MAX` must be >= 0, got {ANGULAR_L_MAX}.")

    if N_ANGULAR_RBF <= 0:
        raise ValueError(f"`N_ANGULAR_RBF` must be positive, got {N_ANGULAR_RBF}.")

    if ANGULAR_RBF_SIGMA_COS <= 0.0:
        raise ValueError(
            f"`ANGULAR_RBF_SIGMA_COS` must be positive, got {ANGULAR_RBF_SIGMA_COS}."
        )

    if ANGULAR_RBF_SIGMA_DEG <= 0.0:
        raise ValueError(
            f"`ANGULAR_RBF_SIGMA_DEG` must be positive, got {ANGULAR_RBF_SIGMA_DEG}."
        )

    if not os.path.exists(RHO_PATH):
        raise FileNotFoundError(f"Density file not found: {RHO_PATH}")

    if not os.path.exists(MOL_PATH):
        raise FileNotFoundError(f"XYZ file not found: {MOL_PATH}")


def validate_density_array(rhos: np.ndarray) -> None:
    if not isinstance(rhos, np.ndarray):
        raise TypeError("Loaded densities must be a NumPy array.")

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
                f"Expected atom-count line with one field at line {line + 1}, "
                f"got: {header!r}"
            )

        try:
            n_atoms = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"Expected integer atom count at line {line + 1}, got: {header!r}"
            ) from exc

        if n_atoms <= 0:
            raise ValueError(
                f"Atom count must be positive at line {line + 1}, got {n_atoms}."
            )

        if line + 1 >= n_lines:
            raise ValueError(f"Missing comment line after atom count at line {line + 1}.")

        if line + 2 + n_atoms > n_lines:
            raise ValueError(
                f"File ends early for structure starting at line {line + 1}; "
                f"expected {n_atoms} atom lines."
            )

        Zs = np.empty(n_atoms, dtype=int)
        xyzs = np.empty((n_atoms, 3), dtype=float)

        for atom_index in range(n_atoms):
            atom_line_num = line + 2 + atom_index
            atom_data = structures[atom_line_num].split()

            if len(atom_data) < 4:
                raise ValueError(
                    f"Malformed atom line {atom_line_num + 1}: expected symbol and "
                    f"3 coordinates, got {atom_data!r}"
                )

            Zs[atom_index] = getZ(atom_data[0])

            try:
                xyzs[atom_index] = np.array(atom_data[1:4], dtype=float) * scale
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse coordinates on line {atom_line_num + 1}: "
                    f"{atom_data[1:4]!r}"
                ) from exc

        molecules.append(Molecule(atom_ids=Zs, positions=xyzs))
        line += n_atoms + 2

    return molecules


def get_channel_length(N: int, resolution: int) -> int:
    return 2 * N if resolution == 0 else resolution


def build_symmetric_densities(rhos: np.ndarray) -> np.ndarray:
    return np.concatenate((rhos[:, ::-1], rhos), axis=1) / 2.0


# ============================================================
# Angular Basis Functions
# ============================================================

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
        vals[ell + 1] = (
            ((2 * ell + 1) * x * vals[ell] - ell * vals[ell - 1])
            / (ell + 1)
        )

    return vals


def angular_rbf_values_cos(
    cos_theta: float,
    centers: np.ndarray,
    sigma: float,
    normalize: bool,
) -> np.ndarray:
    """
    Gaussian RBF angular basis in cos(theta) space.

    Domain:
        cos(theta) in [-1, 1]

    Example centers for N_ANGULAR_RBF = 9:
        [-1.00, -0.75, ..., 0.75, 1.00]
    """
    if sigma <= 0.0:
        raise ValueError(f"`sigma` must be positive, got {sigma}.")

    x = float(np.clip(cos_theta, -1.0, 1.0))
    values = np.exp(-0.5 * ((x - centers) / sigma) ** 2)

    if normalize:
        total = np.sum(values)
        if total > 0.0:
            values = values / total

    return values


def angular_rbf_values_theta(
    cos_theta: float,
    centers_rad: np.ndarray,
    sigma_rad: float,
    normalize: bool,
) -> np.ndarray:
    """
    Gaussian RBF angular basis in theta space.

    Domain:
        theta in [0, pi]

    This is often more chemically interpretable than rbf_cos because
    centers are evenly spaced in actual angle instead of cos(angle).
    """
    if sigma_rad <= 0.0:
        raise ValueError(f"`sigma_rad` must be positive, got {sigma_rad}.")

    x = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(x)

    values = np.exp(-0.5 * ((theta - centers_rad) / sigma_rad) ** 2)

    if normalize:
        total = np.sum(values)
        if total > 0.0:
            values = values / total

    return values


def get_n_angular_channels(
    angular_basis: str,
    angular_l_max: int,
    n_angular_rbf: int,
) -> int:
    if angular_basis == "legendre":
        return angular_l_max + 1

    if angular_basis in {"rbf_cos", "rbf_theta"}:
        return n_angular_rbf

    raise ValueError(f"Unsupported angular basis: {angular_basis!r}")


def get_angular_channel_names(
    angular_basis: str,
    angular_l_max: int,
    n_angular_rbf: int,
) -> list[str]:
    if angular_basis == "legendre":
        return [f"P{ell}" for ell in range(angular_l_max + 1)]

    if angular_basis in {"rbf_cos", "rbf_theta"}:
        return [f"RBF{idx}" for idx in range(n_angular_rbf)]

    raise ValueError(f"Unsupported angular basis: {angular_basis!r}")


def angular_basis_values(
    cos_theta: float,
    angular_basis: str,
    angular_l_max: int,
    rbf_centers_cos: np.ndarray,
    rbf_centers_theta_rad: np.ndarray,
    rbf_sigma_cos: float,
    rbf_sigma_theta_rad: float,
    normalize_rbf: bool,
) -> np.ndarray:
    if angular_basis == "legendre":
        return legendre_values(cos_theta, angular_l_max)

    if angular_basis == "rbf_cos":
        return angular_rbf_values_cos(
            cos_theta=cos_theta,
            centers=rbf_centers_cos,
            sigma=rbf_sigma_cos,
            normalize=normalize_rbf,
        )

    if angular_basis == "rbf_theta":
        return angular_rbf_values_theta(
            cos_theta=cos_theta,
            centers_rad=rbf_centers_theta_rad,
            sigma_rad=rbf_sigma_theta_rad,
            normalize=normalize_rbf,
        )

    raise ValueError(f"Unsupported angular basis: {angular_basis!r}")


def get_region_names(
    angular_basis: str,
    angular_l_max: int,
    n_angular_rbf: int,
) -> list[str]:
    angular_names = get_angular_channel_names(
        angular_basis=angular_basis,
        angular_l_max=angular_l_max,
        n_angular_rbf=n_angular_rbf,
    )

    region_names = [
        "i",
        "ij_sym",
        "ij_antisym",
    ]

    region_names.extend([f"ijk_sym_{name}" for name in angular_names])
    region_names.extend([f"ijk_antisym_{name}" for name in angular_names])


    return region_names


# ============================================================
# Descriptor Generation
# ============================================================

def generateDescriptor(
    mol: Molecule,
    distribution: np.ndarray,
    N: int,
    cutoff: float,
    dR: float,
    normalize_per_atom: bool,
    resolution: int = 0,
    angular_basis: str = "legendre",
    angular_l_max: int = 3,
    n_angular_rbf: int = 9,
    angular_rbf_sigma_cos: float = 0.30,
    angular_rbf_sigma_deg: float = 20.0,
    normalize_angular_rbf: bool = True,
) -> np.ndarray:
    Z = mol.atom_ids
    R = mol.positions
    nAtoms = len(Z)

    full_channel_len = 2 * N

    n_ang = get_n_angular_channels(
        angular_basis=angular_basis,
        angular_l_max=angular_l_max,
        n_angular_rbf=n_angular_rbf,
    )

    radial_grid = np.arange(N, dtype=float) * dR

    rbf_centers_cos = np.linspace(-1.0, 1.0, n_angular_rbf, dtype=float)
    rbf_centers_theta_rad = np.linspace(0.0, np.pi, n_angular_rbf, dtype=float)
    rbf_sigma_theta_rad = np.deg2rad(angular_rbf_sigma_deg)

    density_cache = {
        int(z): getDensity(int(z), distribution)
        for z in np.unique(Z)
    }

    positive_tail_cache = {
        z: density_cache[z][N:]
        for z in density_cache
    }

    deltas = R[:, None, :] - R[None, :, :]
    distance_matrix = np.linalg.norm(deltas, axis=2)

    i_FP = np.zeros(full_channel_len, dtype=float)
    ij_sym_FP = np.zeros(full_channel_len, dtype=float)
    ij_antisym_FP = np.zeros(full_channel_len, dtype=float)

    ijk_sym_FP = np.zeros((n_ang, full_channel_len), dtype=float)
    ijk_antisym_FP = np.zeros((n_ang, full_channel_len), dtype=float)

    for i in range(nAtoms):
        Zi = int(Z[i])
        rho_i = density_cache[Zi]

        i_FP += rho_i

        for j in range(i + 1, nAtoms):
            Zj = int(Z[j])

            R_ij = R[j] - R[i]
            R_ji = -R_ij

            d_ij = distance_matrix[i, j]

            if d_ij <= 0.0:
                continue

            # Because each atom has density support out to cutoff,
            # two atoms can overlap if their centers are within 2*cutoff.
            if d_ij > 2.0 * cutoff:
                continue

            N_ij = int(d_ij / dR)

            if N_ij >= full_channel_len:
                continue

            rho_j = density_cache[Zj]

            # Ordered pair overlap profile.
            ij_overlap = rho_i[N_ij:] * rho_j[: full_channel_len - N_ij]

            base_sym = 0.5 * (ij_overlap + ij_overlap[::-1])
            base_asym = 0.5 * np.abs(ij_overlap - ij_overlap[::-1])

            ij_sym_FP[N_ij:] += base_sym
            ij_antisym_FP[N_ij:] += base_asym

            # ------------------------------------------------------------
            # Symmetrized angular coefficients for unordered pair {i, j}
            #
            # Center at i:
            #     sum_k rho_k(d_ik) * basis(angle j-i-k)
            #
            # Center at j:
            #     sum_k rho_k(d_jk) * basis(angle i-j-k)
            #
            # Final:
            #     0.5 * (center_i + center_j)
            #
            # This removes the previous atom-order / i-j-k asymmetry.
            # ------------------------------------------------------------
            ang_coeffs_i = np.zeros(n_ang, dtype=float)
            ang_coeffs_j = np.zeros(n_ang, dtype=float)

            for k in range(nAtoms):
                if k == i or k == j:
                    continue

                Zk = int(Z[k])
                rho_k_pos = positive_tail_cache[Zk]

                d_ik = distance_matrix[i, k]
                d_jk = distance_matrix[j, k]

                # third atom k must be local to both i and j.
                if d_ik <= 0.0 or d_jk <= 0.0:
                    continue

                if d_ik >= 2.0 * cutoff or d_jk >= 2.0 * cutoff:
                    continue

                # --------------------------------------------------------
                # Center at i: angle j-i-k
                # --------------------------------------------------------
                R_ik = R[k] - R[i]

                cos_jik = np.dot(R_ij, R_ik) / (d_ij * d_ik)
                cos_jik = np.clip(cos_jik, -1.0, 1.0)

                wk_i = np.interp(
                    d_ik,
                    radial_grid,
                    rho_k_pos,
                    left=rho_k_pos[0],
                    right=0.0,
                )

                if wk_i != 0.0:
                    ang_coeffs_i += wk_i * angular_basis_values(
                        cos_theta=cos_jik,
                        angular_basis=angular_basis,
                        angular_l_max=angular_l_max,
                        rbf_centers_cos=rbf_centers_cos,
                        rbf_centers_theta_rad=rbf_centers_theta_rad,
                        rbf_sigma_cos=angular_rbf_sigma_cos,
                        rbf_sigma_theta_rad=rbf_sigma_theta_rad,
                        normalize_rbf=normalize_angular_rbf,
                    )

                # --------------------------------------------------------
                # Center at j: angle i-j-k
                # --------------------------------------------------------
                R_jk = R[k] - R[j]

                cos_ijk = np.dot(R_ji, R_jk) / (d_ij * d_jk)
                cos_ijk = np.clip(cos_ijk, -1.0, 1.0)

                wk_j = np.interp(
                    d_jk,
                    radial_grid,
                    rho_k_pos,
                    left=rho_k_pos[0],
                    right=0.0,
                )

                if wk_j != 0.0:
                    ang_coeffs_j += wk_j * angular_basis_values(
                        cos_theta=cos_ijk,
                        angular_basis=angular_basis,
                        angular_l_max=angular_l_max,
                        rbf_centers_cos=rbf_centers_cos,
                        rbf_centers_theta_rad=rbf_centers_theta_rad,
                        rbf_sigma_cos=angular_rbf_sigma_cos,
                        rbf_sigma_theta_rad=rbf_sigma_theta_rad,
                        normalize_rbf=normalize_angular_rbf,
                    )

            ang_coeffs = 0.5 * (ang_coeffs_i + ang_coeffs_j)

            # Convert angular coefficients into overlap profiles.
            for idx, coeff in enumerate(ang_coeffs):
                if coeff == 0.0:
                    continue

                ijk_sym_FP[idx, N_ij:] += coeff * base_sym
                ijk_antisym_FP[idx, N_ij:] += abs(coeff) * base_asym

    descriptor_blocks = [
        i_FP,
        ij_sym_FP,
        ij_antisym_FP,
        *[ijk_sym_FP[idx] for idx in range(n_ang)],
        *[ijk_antisym_FP[idx] for idx in range(n_ang)],
    ]

    norm_factor = float(nAtoms) if normalize_per_atom else 1.0

    if resolution == 0:
        return np.asarray(descriptor_blocks, dtype=float).ravel() / norm_factor

    if resolution <= 0:
        raise ValueError(
            f"`resolution` must be 0 or a positive integer, got {resolution}."
        )

    if full_channel_len % resolution != 0:
        raise ValueError(
            f"`RESOLUTION` ({resolution}) must divide the channel length "
            f"({full_channel_len})."
        )

    bin_size = full_channel_len // resolution

    rebinned_blocks = [
        block.reshape(resolution, bin_size).sum(axis=1)
        for block in descriptor_blocks
    ]

    return np.asarray(rebinned_blocks, dtype=float).ravel() / norm_factor


def compute_descriptor(
    mol: Molecule,
    distribution: np.ndarray,
    N: int,
    cutoff: float,
    dR: float,
    resolution: int,
    normalize_per_atom: bool,
    angular_basis: str,
    angular_l_max: int,
    n_angular_rbf: int,
    angular_rbf_sigma_cos: float,
    angular_rbf_sigma_deg: float,
    normalize_angular_rbf: bool,
) -> np.ndarray:
    return generateDescriptor(
        mol=mol,
        distribution=distribution,
        N=N,
        cutoff=cutoff,
        dR=dR,
        normalize_per_atom=normalize_per_atom,
        resolution=resolution,
        angular_basis=angular_basis,
        angular_l_max=angular_l_max,
        n_angular_rbf=n_angular_rbf,
        angular_rbf_sigma_cos=angular_rbf_sigma_cos,
        angular_rbf_sigma_deg=angular_rbf_sigma_deg,
        normalize_angular_rbf=normalize_angular_rbf,
    )


# ============================================================
# Output Helpers
# ============================================================

def save_region_slices(
    descriptors: np.ndarray,
    base_output_path: str,
    channel_length: int,
    verbose: int,
) -> None:
    if descriptors.ndim != 2:
        raise ValueError(f"`descriptors` must be 2D, got shape {descriptors.shape}.")

    region_names = get_region_names(
        angular_basis=ANGULAR_BASIS,
        angular_l_max=ANGULAR_L_MAX,
        n_angular_rbf=N_ANGULAR_RBF,
    )

    n_regions = len(region_names)

    expected_width = n_regions * channel_length

    if descriptors.shape[1] != expected_width:
        raise ValueError(
            f"Descriptor width ({descriptors.shape[1]}) does not match expected "
            f"n_regions * channel_length "
            f"({n_regions} * {channel_length} = {expected_width})."
        )

    output_path = resolve_output_path(base_output_path)
    base = output_path[:-4]

    if verbose > 0:
        print(f"Saving {n_regions} descriptor regions.")

    for idx, region_name in enumerate(region_names):
        region_slice = descriptors[
            :,
            idx * channel_length : (idx + 1) * channel_length,
        ]

        np.save(file=f"{base}_{idx}.npy", arr=region_slice)

        if verbose > 1:
            print(f"  region {idx}: {region_name} -> {base}_{idx}.npy")


def print_descriptor_settings(
    n_mols: int,
    n_radial: int,
    channel_length: int,
    n_ang: int,
    n_regions: int,
    descriptor_width: int,
) -> None:
    if VERBOSE <= 0:
        return

    print("===========================================================")
    print("RCD overlap descriptor settings")
    print("-----------------------------------------------------------")
    print(f"DB:                         {DB}")
    print(f"RHO_PATH:                   {RHO_PATH}")
    print(f"MOL_PATH:                   {MOL_PATH}")
    print(f"OUTPUT_FILE:                {resolve_output_path(OUTPUT_FILE)}")
    print(f"XYZ_UNITS:                  {XYZ_UNITS}")
    print(f"N_MOLECULES:                {n_mols}")
    print(f"N_RADIAL:                   {n_radial}")
    print(f"CHANNEL_LENGTH:             {channel_length}")
    print(f"CUTOFF:                     {CUTOFF} A")
    print(f"DR:                         {DR} A")
    print(f"RESOLUTION:                 {RESOLUTION}")
    print(f"NORMALIZE_PER_ATOM:         {NORMALIZE_PER_ATOM}")
    print(f"ANGULAR_BASIS:              {ANGULAR_BASIS}")
    print(f"ANGULAR_L_MAX:              {ANGULAR_L_MAX}")
    print(f"N_ANGULAR_RBF:              {N_ANGULAR_RBF}")
    print(f"ANGULAR_RBF_SIGMA_COS:      {ANGULAR_RBF_SIGMA_COS}")
    print(f"ANGULAR_RBF_SIGMA_DEG:      {ANGULAR_RBF_SIGMA_DEG}")
    print(f"NORMALIZE_ANGULAR_RBF:      {NORMALIZE_ANGULAR_RBF}")
    print(f"N_ANGULAR_CHANNELS:         {n_ang}")
    print(f"N_REGIONS:                  {n_regions}")
    print(f"DESCRIPTOR_WIDTH:           {descriptor_width}")
    print("ANGULAR SYMMETRIZATION:     enabled")
    print("===========================================================")


# ============================================================
# Main
# ============================================================

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
            f"Density array only contains {len(rhos)} elements, "
            f"but molecule data requires Z={max_Z}."
        )

    N = rhos.shape[1]
    rhos_sym = build_symmetric_densities(rhos)

    channel_length = get_channel_length(N=N, resolution=RESOLUTION)

    n_ang = get_n_angular_channels(
        angular_basis=ANGULAR_BASIS,
        angular_l_max=ANGULAR_L_MAX,
        n_angular_rbf=N_ANGULAR_RBF,
    )

    region_names = get_region_names(
        angular_basis=ANGULAR_BASIS,
        angular_l_max=ANGULAR_L_MAX,
        n_angular_rbf=N_ANGULAR_RBF,
    )

    n_regions = len(region_names)
    descriptor_width = n_regions * channel_length

    print_descriptor_settings(
        n_mols=len(mols),
        n_radial=N,
        channel_length=channel_length,
        n_ang=n_ang,
        n_regions=n_regions,
        descriptor_width=descriptor_width,
    )

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
            normalize_per_atom=NORMALIZE_PER_ATOM,
            angular_basis=ANGULAR_BASIS,
            angular_l_max=ANGULAR_L_MAX,
            n_angular_rbf=N_ANGULAR_RBF,
            angular_rbf_sigma_cos=ANGULAR_RBF_SIGMA_COS,
            angular_rbf_sigma_deg=ANGULAR_RBF_SIGMA_DEG,
            normalize_angular_rbf=NORMALIZE_ANGULAR_RBF,
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
        verbose=VERBOSE,
    )

    if VERBOSE > 0:
        descriptors_per_second = len(mols) / duration if duration > 0.0 else np.inf
        descriptors_per_hour = 3600.0 * descriptors_per_second
        descriptors_per_second_per_core = descriptors_per_second / N_CORES
        descriptors_per_hour_per_core = descriptors_per_hour / N_CORES

        print(
            " ===========================================================\n",
            "RCD DESCRIPTORS COMPLETE\n",
            f"XYZ INPUT UNITS: {XYZ_UNITS} "
            f"({'converted to angstrom' if XYZ_UNITS == 'bohr' else 'no conversion'})\n",
            f"ANGULAR BASIS: {ANGULAR_BASIS}\n",
            f"ANGULAR SYMMETRIZATION: enabled\n",
            f"DESCRIPTOR ARRAY SHAPE: {descriptors.shape}\n",
            f"DESCRIPTORS WRITTEN TO {output_path}\n",
            f"{len(mols)} DESCRIPTORS COMPUTED IN {round(duration, 2)} SECONDS "
            f"({round(duration / 60.0, 2)} MINUTES).\n",
            f"{round(descriptors_per_second, 2)} DESCRIPTORS/SECOND "
            f"({round(descriptors_per_hour, 2)} DESCRIPTORS/HOUR)\n",
            f"{round(descriptors_per_second_per_core, 2)} DESCRIPTORS/SECOND/CORE "
            f"({round(descriptors_per_hour_per_core, 2)} DESCRIPTORS/HOUR/CORE)\n",
            "===========================================================\n",
        )


if __name__ == "__main__":
    main()
