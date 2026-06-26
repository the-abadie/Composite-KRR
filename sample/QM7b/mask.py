import numpy as np

mask = np.load("sample/QM7b/qm7b_good_xyz_indices.npy")

X = np.load("sample/QM7b/QM7b_overlap_leg5_V2.npy")


X_good = X[mask]

np.save("sample/QM7b/QM7b_overlap_leg5_V2_good.npy", X_good)
