import scipy.io as io
import numpy as np


qm7 = np.load("sample/QM7/atomization_energy.npy")
qm7b = np.load("sample/QM7b/ae_pbe0.npy")

print(qm7[0])
print(qm7b[0])
# qm7b = io.loadmat("sample/QM7b/qm7b.mat")
# import numpy as np

# target_names = [
#     "ae_pbe0",
#     "zindo_excitation_energy_with_the_most_absorption",
#     "zindo_highest_absorption",
#     "zindo_homo",
#     "zindo_lumo",
#     "zindo_1st_excitation_energy",
#     "zindo_ionization_potential",
#     "zindo_electron_affinity",
#     "ks_homo",
#     "ks_lumo",
#     "gw_homo",
#     "gw_lumo",
#     "polarizability_pbe",
#     "polarizability_scs",
# ]

# print(qm7b["T"].shape)


# for i in range(len(target_names)):
#     print(f"{target_names[i]}")
#     target = qm7b["T"][:,i]
#     np.save(f"sample/QM7b/{target_names[i]}.npy", target)
