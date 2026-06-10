SEED = 1
VERBOSITY = 2

RUN_NAME = "QM7_OVERLAP_SEP"

# X_PATHS = [
#     "sample/QM7/QM7_overlap_N_0_nosym.npy",
#     "sample/QM7/QM7_overlap_N_1.npy", "sample/QM7/QM7_overlap_N_2.npy",
#     "sample/QM7/QM7_overlap_N_3.npy", "sample/QM7/QM7_overlap_N_4.npy",
# ]

# X_NAMES = [
#     "leg_7",
#     "1B ",
#     "2Bs", "2Ba",
#     "3Bs", "3Ba",
# ]

# X_NORMS = [
#     "standard",
#     "standard", "standard",
#     "standard", "standard",
# ]

X_MAX = 4
X_PATHS = [f"sample/QM7/QM7_overlap_{i}.npy" for i in range(0,  X_MAX+1)]
X_NAMES = [f"overlap_mk_i_{i:02}" for i in range(0,  X_MAX+1)]
X_NORMS = ["standard" for _ in X_PATHS]
# X_PCA_COMPONENTS = [0.99 for _ in X_PATHS]
# X_PCA_WHITEN = [False for _ in X_PATHS]


Y_PATH = "sample/QM7/atomization_energy.npy"
Y_NAME = "atomization energy"
Y_NORM = "standard"

USE_PREDEFINED_SPLITS = False
# PREDEF_TRAINING_IDX_PATH = "predef_training_idx.npy"
# PREDEF_VAL_KFOLD_IDX_PATH = "predef_val_kfold_idx.npy"
# PREDEF_TESTING_IDX_PATH  = "predef_testing_idx.npy"

N_SAMPLES = 7165
TRAIN_VAL_SPLIT = 0.15
N_KFOLD = 5
STRATIFY = True
N_STRATA = 5

KRR_KERNEL = "rbf"

KRR_ALPHA_BOUNDS = (1e-9, 1e2)
KRR_GAMMA_BOUNDS = (1e-9, 1e2)
KRR_RANDOM_SEARCH_STAGE1 = 5000
KRR_RANDOM_SEARCH_STAGE2 = 5000
KRR_RANDOM_SEARCH_STAGE3 = 0
KRR_TOP_K_FRACTION = 0.25
KRR_TOP_K_MIN_CANDIDATES = 5
KRR_BAYESIAN_SEARCH_TRIALS = 2000
KRR_BAYESIAN_SEARCH_TIMEOUT = None
KRR_BAYESIAN_SEARCH_PATIENCE = 1000

KRR_EVALUATE_KERNEL_CONTRIBUTIONS = True
KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS = None

# None follows sklearn's default; -1 uses all available cores.
KRR_RANDOM_SEARCH_N_JOBS = -1
KRR_RANDOM_SEARCH_BLAS_THREADS = 1
KRR_COMPUTE_DTYPE = "float64"
KRR_USE_DISTANCE_CACHE = True
KRR_DISTANCE_BLOCK_SIZE = 2048
KRR_DISTANCE_CACHE_DTYPE = KRR_COMPUTE_DTYPE
KRR_DISTANCE_CACHE_N_JOBS = -1
KRR_DISTANCE_CACHE_MEMORY_FRACTION = 0.80

KRR_SCORE_METRIC = "neg_mean_absolute_error"


OUTPUT_DIR = f"sample/output/{SEED}/{RUN_NAME}"
OVERWRITE_OK = True
