SEED = 1
N_TRAIN = 100
VERBOSITY = 2

RUN_NAME = "QM7_OVERLAP_CONCAT_DCT_100_0.70frac"

X_MAX = 10
# X_PATHS = [f"sample/QM7/QM7_overlap_leg3_dct_{i}.npy" for i in range(0,  X_MAX+1)]
# X_NAMES = [f"overlap_leg3_dct_{i:02}" for i in range(0,  X_MAX+1)]

X_PATHS = ["sample/QM7/QM7_overlap_leg3_dct_concat.npy"]
X_NAMES = ["LEG3 Overlap DCT Concatenated"]
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
TRAIN_VAL_SPLIT = N_TRAIN/N_SAMPLES
N_KFOLD = 5
STRATIFY = True
N_STRATA = 5

KRR_KERNEL = "rbf"
KRR_BACKEND = "exact"  # "exact" or "nystrom"

KRR_ALPHA_BOUNDS = (1e-9, 1e2)
KRR_GAMMA_BOUNDS = (1e-9, 1e2)
KRR_RANDOM_SEARCH_STAGE1 = 75
KRR_RANDOM_SEARCH_STAGE2 = 75
KRR_RANDOM_SEARCH_STAGE3 = 0
KRR_TOP_K_FRACTION = 0.25
KRR_TOP_K_MIN_CANDIDATES = 5
KRR_BAYESIAN_SEARCH_TRIALS = 50
KRR_BAYESIAN_SEARCH_TIMEOUT = None
KRR_BAYESIAN_SEARCH_PATIENCE = 25
KRR_BAYESIAN_BATCH_SIZE = 1  # 0 or 1 uses Optuna's default sequential optimize loop.

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
KRR_GAMMA_PRIOR_MAX_SAMPLES = 5000
KRR_CACHED_SCORING_BACKEND = "numpy"  # "numpy" or "pytorch"
KRR_PYTORCH_DEVICE = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
KRR_PYTORCH_DEVICES = None  # None uses KRR_PYTORCH_DEVICE; "auto" uses one GPU per CV fold when available.
KRR_PYTORCH_CANDIDATE_BATCH_SIZE = 1
KRR_PYTORCH_PREDICT_BATCH_SIZE = 2048
KRR_NYSTROM_N_LANDMARKS = 2048
KRR_NYSTROM_LANDMARK_SELECTION = "random"  # "random" or "first"
KRR_NYSTROM_BATCH_SIZE = 2048
KRR_NYSTROM_EIGENVALUE_FLOOR = 1e-12

KRR_SCORE_METRIC = "neg_mean_absolute_error"


OUTPUT_DIR = f"sample/output/{SEED}/{RUN_NAME}"
OVERWRITE_OK = True
