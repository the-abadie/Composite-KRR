SEED = 1

X_PATHS = ["sample/desc1.npy", "sample/desc2.npy", "sample/desc3.npy"]
X_NAMES = ["desc1", "desc2", "desc3"]
X_NORMS = ["standard", "standard", "standard"]

Y_PATH = "sample/target1.npy"
Y_NAME = "my_target"
Y_NORM = "standard"

N_SAMPLES = 1000
TRAIN_VAL_SPLIT = 0.70
N_KFOLD = 5
STRATIFY = True
N_STRATA = 5

KRR_KERNEL = "rbf"
KRR_ALPHA_BOUNDS = (1e-8, 1e2)
KRR_GAMMA_BOUNDS = (1e-6, 1e2)
KRR_RANDOM_SEARCH_STAGE1 = 25
KRR_RANDOM_SEARCH_STAGE2 = 25
KRR_RANDOM_SEARCH_STAGE3 = 25
KRR_SCORE_METRIC = "rmse"

VERBOSITY = 2
