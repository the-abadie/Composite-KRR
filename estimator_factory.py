from search_random import CompositeKRREstimator
from nystrom_krr import CompositeNystromKRREstimator
from torch_cg_krr import CompositeTorchCGKRREstimator


def normalize_krr_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend in {"exact", "dense"}:
        return "exact"
    if backend in {"nystrom", "nyström"}:
        return "nystrom"
    if backend in {"cg", "exact_cg", "matrix_free_cg", "torch_cg"}:
        return "cg"
    raise ValueError('KRR backend must be "exact", "nystrom", or "cg".')


def make_composite_krr_regressor(
    *,
    krr_backend: str,
    names,
    kernel_types,
    normalizations,
    pca_components,
    pca_whiten,
    normalize_kernel_weights: bool,
    compute_dtype,
    nystrom_n_landmarks: int = 2048,
    nystrom_landmark_selection: str = "random",
    random_state=None,
    nystrom_backend: str = "numpy",
    pytorch_device: str | None = "auto",
    pytorch_devices="auto",
    nystrom_batch_size: int = 2048,
    nystrom_eigenvalue_floor: float = 1e-12,
    cg_tol: float = 1e-6,
    cg_max_iter: int = 1000,
    cg_block_size: int = 2048,
    cg_log_interval: int = 25,
):
    backend = normalize_krr_backend(krr_backend)
    if backend == "exact":
        return CompositeKRREstimator(
            names=names,
            kernel_types=kernel_types,
            normalizations=normalizations,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            normalize_kernel_weights=normalize_kernel_weights,
            compute_dtype=compute_dtype,
        )

    if backend == "nystrom":
        return CompositeNystromKRREstimator(
            names=names,
            kernel_types=kernel_types,
            normalizations=normalizations,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            normalize_kernel_weights=normalize_kernel_weights,
            compute_dtype=compute_dtype,
            n_landmarks=nystrom_n_landmarks,
            landmark_selection=nystrom_landmark_selection,
            random_state=random_state,
            backend=nystrom_backend,
            pytorch_device=pytorch_device,
            batch_size=nystrom_batch_size,
            eigenvalue_floor=nystrom_eigenvalue_floor,
        )

    return CompositeTorchCGKRREstimator(
        names=names,
        kernel_types=kernel_types,
        normalizations=normalizations,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
        normalize_kernel_weights=normalize_kernel_weights,
        compute_dtype=compute_dtype,
        pytorch_device=pytorch_device,
        pytorch_devices=pytorch_devices,
        cg_tol=cg_tol,
        cg_max_iter=cg_max_iter,
        cg_block_size=cg_block_size,
        cg_log_interval=cg_log_interval,
    )
