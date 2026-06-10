# region Import Packages
from dataclasses import dataclass

import numpy as np

# region User Parameters
DB:str = "QM7"
MOL_PATH = f"sample/{DB}/{DB}.xyz"

OUTPUT_FILE = f"sample/{DB}/{DB}_atomic_IDs.npy"

N_CORES = 8
RESOLUTION = 0
VERBOSE = 1
XYZ_UNITS = "bohr"

# endregion


# region Class Definitions
@dataclass(frozen=True)
class Molecule:
    atom_ids: np.ndarray
    positions: np.ndarray
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


def getMols(filepath: str) -> list[Molecule]:
    with open(filepath, "r", encoding="utf-8") as f:
        structures = f.readlines()

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
                xyzs[atom_index] = np.array([float(val) for val in atom_data[1:4]], dtype=float)
                atom_index += 1

            molecules.append(Molecule(atom_ids=Zs, positions=xyzs))

    return molecules


molecules = getMols(MOL_PATH)
Z_symbols = ["H", "C", "N", "O", "S"]

atomic_IDs = np.zeros((len(molecules), len(Z_symbols)))

for i in range(len(molecules)):
    mol = molecules[i]

    Zs = mol.atom_ids

    for z in Zs:
        if z == 1:
            atomic_IDs[i, 0] += 1
        elif z == 6:
            atomic_IDs[i, 1] += 1
        elif z == 7:
            atomic_IDs[i, 2] += 1
        elif z == 8:
            atomic_IDs[i, 3] += 1
        elif z == 16:
            atomic_IDs[i, 4] += 1
        else: raise ValueError("bruh moment")


print(atomic_IDs)

np.save(OUTPUT_FILE, atomic_IDs)
