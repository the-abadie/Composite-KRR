# region Import Packages
import os
import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy import interpolate
# endregion

BOHR_TO_ANGSTROM = 0.529177210903

# region User Parameters
DB:str = "QM7"
DISTRIBUTION_PATH = "RCD_3.0_300.npy"
MOL_PATH = f"{DB}.xyz"

OUTPUT_FILE = f"{DB}_overlap.npy"

N_CORES = 16
RESOLUTION = 0
VERBOSE = 1
XYZ_UNITS = "bohr"

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


@dataclass(frozen=True)
class DescriptorBlock:
    label: str
    save_positive_half: bool = False


BASE_DESCRIPTOR_BLOCKS = (
    DescriptorBlock("1B", save_positive_half=True),
    DescriptorBlock("2Bs"),
    DescriptorBlock("2Ba"),
    DescriptorBlock("3Bs"),
    DescriptorBlock("3Ba"),
)
BASE_DERIVATIVE_DESCRIPTOR_BLOCKS = (
    DescriptorBlock("D_1B", save_positive_half=True),
    DescriptorBlock("D_2Bs"),
    DescriptorBlock("D_2Ba"),
    DescriptorBlock("D_3Bs"),
    DescriptorBlock("D_3Ba"),
)
EXTRA_DESCRIPTOR_BLOCKS = (
    DescriptorBlock("2B_raw"),
    DescriptorBlock("3B_raw"),
)
EXTRA_DERIVATIVE_DESCRIPTOR_BLOCKS = (
    DescriptorBlock("D_2B_raw"),
    DescriptorBlock("D_3B_raw"),
)
# endregion


# region Function Definitions
def getZ(symbol:str) -> int:
    """
    Returns the atomic number for a_ev given atomic symbol, up to Z = 102, Nobelium.
    :param symbol: ``str`` representation of the atomic symbol.
    :return: ``int`` atomic number corresponding to ``symbol``
    """
    assert symbol != "", "Label must not be empty."

    elements = (
        "H  He "
        "Li  Be  B   C   N   O   F   Ne "
        "Na  Mg  Al  Si  P   S   Cl  Ar "
        "K   Ca  Sc  Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr "
        "Rb  Sr  Y   Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe "
        "Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb "
        "Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn "
        "Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No".split())

    return elements.index(symbol) + 1


def getDensity(Z: int, data: np.ndarray) -> np.ndarray:
    return data[Z - 1]


def get_descriptor_block_infos(derivatives: bool) -> tuple[DescriptorBlock, ...]:
    if derivatives:
        return (
            *BASE_DESCRIPTOR_BLOCKS,
            *BASE_DERIVATIVE_DESCRIPTOR_BLOCKS,
            *EXTRA_DESCRIPTOR_BLOCKS,
            *EXTRA_DERIVATIVE_DESCRIPTOR_BLOCKS,
        )

    return (*BASE_DESCRIPTOR_BLOCKS, *EXTRA_DESCRIPTOR_BLOCKS)


def getMols(filepath: str, xyz_units: str = "angstrom") -> list[Molecule]:
    with open(filepath, "r", encoding="utf-8") as f:
        structures = f.readlines()

    scale = BOHR_TO_ANGSTROM if xyz_units == "bohr" else 1.0
    molecules: list[Molecule] = []

    for line in range(len(structures)):
        x = structures[line].split()

        if len(x) == 1 and not x[0].startswith("QM7"):
            n_atoms = int(x[0])

            Zs = np.zeros(n_atoms, dtype=int)
            xyzs = np.zeros((n_atoms, 3), dtype=float)

            atom_index = 0
            for j in range(line + 2, line + 2 + n_atoms):
                atom_data = structures[j].split()
                Zs[atom_index] = getZ(atom_data[0])
                xyzs[atom_index] = np.array([float(val) for val in atom_data[1:4]], dtype=float) * scale
                atom_index += 1

            molecules.append(Molecule(atom_ids=Zs, positions=xyzs))

    return molecules


def generateDescriptor(
    mol: Molecule,
    distribution: np.ndarray,
    N: int,
    cutoff: float,
    dR: float,
    derivatives: bool,
    normalize_per_atom: bool,
    resolution: int = 0,
) -> np.ndarray:
    Z = mol.atom_ids
    R = mol.positions
    nAtoms = len(Z)

    i_FP = np.zeros(2 * N)
    ij_FP = np.zeros(2 * N)
    ij_sym_FP = np.zeros(2 * N)
    ij_antisym_FP = np.zeros(2 * N)
    ijk_FP = np.zeros(2 * N)
    ijk_sym_FP = np.zeros(2 * N)
    ijk_antisym_FP = np.zeros(2 * N)

    for i in range(nAtoms):
        rho_i = getDensity(Z[i], distribution)
        i_FP += rho_i

        for j in range(i + 1, nAtoms):
            R_ij = R[i] - R[j]
            d_ij = np.linalg.norm(R_ij)

            if d_ij > 2 * cutoff:
                continue

            rho_j = getDensity(Z[j], distribution)

            N_ij = int(d_ij / dR)
            ij_overlap = rho_i[N_ij:] * rho_j[: 2 * N - N_ij]

            ij_FP[N_ij:] += ij_overlap
            ij_sym_FP    [N_ij:] += 0.5 * (ij_overlap + ij_overlap[::-1])
            ij_antisym_FP[N_ij:] += 0.5 * np.abs(ij_overlap - ij_overlap[::-1])

            ij_k_interactions = np.zeros(2 * N - N_ij)
            for k in range(nAtoms):
                if k == i or k == j:
                    continue

                d_ik = np.linalg.norm(R[i] - R[k])
                d_jk = np.linalg.norm(R[j] - R[k])
                h_k = np.linalg.norm(np.cross(R[k] - R[i], R[k] - R[j])) / d_ij

                if d_ik >= 2 * cutoff or d_jk >= 2 * cutoff or h_k >= cutoff:
                    continue

                rho_k = getDensity(Z[k], distribution)
                k_interp = interpolate.interp1d(np.arange(N) * dR, rho_k[N:])

                R_k0 = R[i] - R[k] - R_ij * (cutoff / d_ij)
                xs = np.linalg.norm((np.arange(2 * N) * dR)[:, None] * R_ij / d_ij + R_k0, axis=1)
                xs = np.clip(xs, 0, (N - 2) * dR)

                ij_k_interactions += k_interp(xs[N_ij:])

            ijk_overlap = ij_overlap * ij_k_interactions

            ijk_FP[N_ij:] += ijk_overlap
            ijk_sym_FP[N_ij:] += 0.5 * (ijk_overlap + ijk_overlap[::-1])
            ijk_antisym_FP[N_ij:] += 0.5 * np.abs(ijk_overlap - ijk_overlap[::-1])

    if derivatives:
        D_i_FP = np.gradient(i_FP, dR)
        D_ij_FP = np.gradient(ij_FP, dR)
        D_ij_sym_FP = np.gradient(ij_sym_FP, dR)
        D_ij_antisym_FP = np.gradient(ij_antisym_FP, dR)
        D_ijk_FP = np.gradient(ijk_FP, dR)
        D_ijk_sym_FP = np.gradient(ijk_sym_FP, dR)
        D_ijk_antisym_FP = np.gradient(ijk_antisym_FP, dR)

        descriptor_blocks = [
            i_FP,
            ij_sym_FP, ij_antisym_FP,
            ijk_sym_FP, ijk_antisym_FP,
            D_i_FP,
            D_ij_sym_FP, D_ij_antisym_FP,
            D_ijk_sym_FP, D_ijk_antisym_FP,
            ij_FP, ijk_FP,
            D_ij_FP, D_ijk_FP,
        ]
    else:
        descriptor_blocks = [
            i_FP,
            ij_sym_FP, ij_antisym_FP,
            ijk_sym_FP, ijk_antisym_FP,
            ij_FP, ijk_FP,
        ]

    norm_factor = float(nAtoms) if normalize_per_atom else 1.0

    if resolution == 0:
        return np.array(descriptor_blocks).ravel() / norm_factor

    full_channel_len = 2 * N
    if resolution <= 0:
        raise ValueError(f"`resolution` must be 0 or a positive integer, got {resolution}.")
    if full_channel_len % resolution != 0:
        print(
            f"`RESOLUTION` ({resolution}) does not divide the channel length ({full_channel_len}). "
            "Descriptor size will not be reduced."
        )
        return np.array(descriptor_blocks).ravel() / norm_factor

    bin_size = full_channel_len // resolution
    rebinned_blocks = [block.reshape(resolution, bin_size).sum(axis=1) for block in descriptor_blocks]
    return np.array(rebinned_blocks).ravel() / norm_factor


def compute_descriptor(i: int, mol: Molecule, distribution: np.ndarray, N: int, cutoff: float, dR: float,
                       resolution: int, derivatives: bool, normalize_per_atom: bool):
    desc = generateDescriptor(
        mol=mol,
        distribution=distribution,
        N=N,
        cutoff=cutoff,
        dR=dR,
        derivatives=derivatives,
        normalize_per_atom=normalize_per_atom,
        resolution=resolution,
    )
    return i, desc


def get_channel_length(N: int, resolution: int) -> int:
    full_channel_len = 2 * N
    if resolution == 0 or full_channel_len % resolution != 0:
        return full_channel_len

    return resolution


def save_region_slices(descriptors: np.ndarray, base_output_path: str, channel_length: int, derivatives: bool, verbose: int):
    block_infos = get_descriptor_block_infos(derivatives)
    n_regions = len(block_infos)

    if descriptors.shape[1] != n_regions * channel_length:
        raise ValueError(
            f"Descriptor width ({descriptors.shape[1]}) does not match expected n_regions * channel_length "
            f"({n_regions} * {channel_length} = {n_regions * channel_length})."
        )

    root, _ = os.path.splitext(base_output_path)

    if verbose > 0:
        print(f"{n_regions} regions.")

    for i, block_info in enumerate(block_infos):
        region_slice = descriptors[:, i * channel_length : (i + 1) * channel_length]
        assert region_slice.shape[1] == channel_length

        region_path = f"{root}_{int(i)}.npy"
        np.save(file=region_path, arr=region_slice)

        if verbose > 0:
            print(f"  {i}: {block_info.label} -> {region_path} {region_slice.shape}")

        if block_info.save_positive_half:
            if channel_length % 2 != 0:
                raise ValueError(
                    f"Cannot save positive half for {block_info.label}: channel length "
                    f"({channel_length}) is not even."
                )

            positive_half = region_slice[:, channel_length // 2 :]
            positive_half_path = f"{root}_{int(i)}_nosym.npy"
            np.save(file=positive_half_path, arr=positive_half)

            if verbose > 0:
                print(f"     positive half -> {positive_half_path} {positive_half.shape}")


def collect_descriptors_with_progress(
    mols: list[Molecule],
    distribution: np.ndarray,
    N: int,
    cutoff: float,
    dR: float,
    resolution: int,
    derivatives: bool,
    normalize_per_atom: bool,
    n_cores: int,
    verbose: int,
):
    total = len(mols)
    descriptors = [None] * total

    results = Parallel(
        n_jobs=n_cores,
        prefer="processes",
        return_as="generator_unordered",
    )(
        delayed(compute_descriptor)(
            i=i,
            mol=mol,
            distribution=distribution,
            N=N,
            cutoff=cutoff,
            dR=dR,
            resolution=resolution,
            derivatives=derivatives,
            normalize_per_atom=normalize_per_atom,
        )
        for i, mol in enumerate(mols)
    )

    completed = 0
    progress_step = max(1, total // 20) if total > 0 else 1
    last_update_time = time.time()

    for i, desc in results:
        descriptors[i] = desc
        completed += 1

        should_report = (
            verbose > 0
            and (
                completed == 1
                or completed == total
                or completed % progress_step == 0
                or time.time() - last_update_time >= 30.0
            )
        )
        if should_report:
            percent = 100.0 * completed / total
            print(f"Progress: {completed}/{total} descriptors ({percent:.1f}%)")
            last_update_time = time.time()

    return np.asarray(descriptors, dtype=float)
# endregion


def main() -> None:
    # region Validation
    if N_CORES <= 0:
        raise ValueError(f"`N_CORES` must be positive, got {N_CORES}.")
    if RESOLUTION < 0:
        raise ValueError(f"`RESOLUTION` must be >= 0, got {RESOLUTION}.")
    if XYZ_UNITS not in {"angstrom", "bohr"}:
        raise ValueError(f"`XYZ_UNITS` must be either 'angstrom' or 'bohr', got {XYZ_UNITS!r}.")
    if not os.path.exists(DISTRIBUTION_PATH):
        raise FileNotFoundError(f"Density file not found: {DISTRIBUTION_PATH}")
    if not os.path.exists(MOL_PATH):
        raise FileNotFoundError(f"XYZ file not found: {MOL_PATH}")
    # endregion

    # region Pre-Processing
    rhos = np.load(DISTRIBUTION_PATH)
    mols = getMols(MOL_PATH, xyz_units=XYZ_UNITS)

    distribution_length = rhos.shape[1]
    print(f"Cutoff: {CUTOFF} A")
    print(f"dr: {DR} A")

    rhos_sym = np.zeros((len(rhos), 2 * distribution_length))
    for i in range(len(rhos)):
        rhos_sym[i] = np.concatenate((rhos[i][::-1], rhos[i])) / 2.0

    channel_length = get_channel_length(N=distribution_length, resolution=RESOLUTION)
    # endregion

    # region Program Start
    start_time = time.time()

    descriptors = collect_descriptors_with_progress(
        mols=mols,
        distribution=rhos_sym,
        N=distribution_length,
        cutoff=CUTOFF,
        dR=DR,
        resolution=RESOLUTION,
        derivatives=DERIVATIVES,
        normalize_per_atom=NORMALIZE_PER_ATOM,
        n_cores=N_CORES,
        verbose=VERBOSE,
    )

    duration = time.time() - start_time
    # endregion

    # region Save Data
    np.save(file=OUTPUT_FILE, arr=descriptors)
    save_region_slices(
        descriptors=descriptors,
        base_output_path=OUTPUT_FILE,
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
            f"DESCRIPTORS WRITTEN TO {OUTPUT_FILE}\n",
            f"{len(mols)} DESCRIPTORS COMPUTED IN {round(duration, 2)} SECONDS ({round(duration / 60., 2)} MINUTES).\n",
            f"{round(len(mols) / duration, 2)} DESCRIPTORS/SECOND ({round(3600 * len(mols) / duration, 2)} DESCRIPTORS/HOUR)\n",
            f"{round(len(mols) / duration / N_CORES, 2)} DESCRIPTORS/SECOND/CORE ({round(3600 * len(mols) / duration / N_CORES, 2)} DESCRIPTORS/HOUR/CORE)\n",
            "===========================================================\n",
        )
    # endregion


if __name__ == "__main__":
    main()
