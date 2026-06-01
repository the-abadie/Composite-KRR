SEED = 1
VERBOSITY = 1

X_PATHS = ["sample/desc1.npy", "sample/desc2.npy", "sample/desc3.npy"]
X_NAMES = ["desc1", "desc2", "desc3"]
X_NORMS = ["standard", "standard", "standard"]

Y_PATH = "sample/target1.npy"
Y_NAME = "my_target"
Y_NORM = "standard"

N_SAMPLES = 1000
USE_PREDEFINED_SPLITS = False

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

# None follows sklearn's default; -1 uses all available cores.
KRR_RANDOM_SEARCH_N_JOBS = None
KRR_RANDOM_SEARCH_PLOT_PATH = "random_search_validation_error.png"
KRR_BAYESIAN_SEARCH_TRIALS = 50
KRR_BAYESIAN_SEARCH_TIMEOUT = None
KRR_BAYESIAN_SEARCH_PATIENCE = 10
KRR_SCORE_METRIC = "neg_mean_absolute_error"
