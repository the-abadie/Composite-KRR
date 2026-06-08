from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


# =========================
# user settings
# =========================
DB:str = "QM7"

XYZ_PATH  = Path(f"sample/{DB}/{DB}.xyz")
SAVE_PATH = Path(f"sample/{DB}/{DB}_CM_ev_sorted.npy")
COORDS_IN_BOHR = True

BOHR_TO_ANG = 0.529177210903

# sort eigenvalues descending
SORT_DESCENDING = True

# padding value for shorter molecules, so final saved array is 2D
PAD_VALUE = 0.0


# =========================
# xyz reader
# =========================
def read_multixyz_robust(path: Path):
    with open(path, "r") as f:
        lines = [line.rstrip() for line in f]

    i = 0
    nlines = len(lines)

    while i < nlines:
        while i < nlines and not lines[i].strip():
            i += 1
        if i >= nlines:
            break

        n_atoms = int(lines[i].strip())
        i += 1

        comment = lines[i] if i < nlines else ""
        i += 1

        symbols = []
        positions = []

        for _ in range(n_atoms):
            parts = lines[i].split()
            if len(parts) < 4:
                raise ValueError(f"Malformed atom line at line {i+1}: {lines[i]!r}")
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            symbols.append(sym)
            positions.append([x, y, z])
            i += 1

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info["comment"] = comment
        yield atoms


# =========================
# Coulomb matrix + eigs
# =========================
def get_coulomb_matrix(atoms: Atoms, coords_in_bohr: bool = True) -> np.ndarray:
    Z = np.array([atomic_numbers[s] for s in atoms.get_chemical_symbols()], dtype=float)
    R = atoms.get_positions().astype(float).copy()

    if coords_in_bohr:
        R *= BOHR_TO_ANG

    n = len(Z)
    M = np.zeros((n, n), dtype=float)

    # diagonal
    np.fill_diagonal(M, 0.5 * Z**2.4)

    # off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(R[i] - R[j])
            if dist <= 0.0:
                raise ValueError(f"Zero interatomic distance encountered for atoms {i} and {j}.")
            val = (Z[i] * Z[j]) / dist
            M[i, j] = val
            M[j, i] = val

    return M


def coulomb_eigenvalue_descriptor(atoms: Atoms, coords_in_bohr: bool = True) -> np.ndarray:
    M = get_coulomb_matrix(atoms, coords_in_bohr=coords_in_bohr)

    # symmetric matrix -> use eigh
    eigvals = np.linalg.eigvalsh(M)

    eigvals.sort()
    if SORT_DESCENDING:
        eigvals = eigvals[::-1]

    return eigvals


def pad_descriptors(descriptors: list[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    max_len = max(len(d) for d in descriptors)
    out = np.full((len(descriptors), max_len), pad_value, dtype=float)

    for i, d in enumerate(descriptors):
        out[i, :len(d)] = d

    return out


def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    descriptors = []
    comments = []
    natoms_list = []

    for idx, atoms in enumerate(read_multixyz_robust(XYZ_PATH)):
        d = coulomb_eigenvalue_descriptor(atoms, coords_in_bohr=COORDS_IN_BOHR)
        descriptors.append(d)
        comments.append(atoms.info.get("comment", ""))
        natoms_list.append(len(atoms))

        print(
            f"{idx:4d} | {atoms.info.get('comment', ''):<20} | "
            f"N_atoms = {len(atoms):2d} | n_eigs = {len(d):2d}"
        )

    X = pad_descriptors(descriptors, pad_value=PAD_VALUE)

    np.save(SAVE_PATH, X)

    print()
    print(f"Saved descriptor array to: {SAVE_PATH}")
    print(f"Shape: {X.shape}")
    print(f"Max atoms: {max(natoms_list)}")
    print(f"Max eigenvalue count: {X.shape[1]}")


if __name__ == "__main__":
    main()
