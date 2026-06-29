from search_random import CompositeKRREstimator
from nystrom_krr import CompositeNystromKRREstimator


def normalize_krr_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend in {"exact", "dense"}:
        return "exact"
    if backend in {"nystrom", "nyström"}:
        return "nystrom"
    raise ValueError('KRR backend must be "exact" or "nystrom".')


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
    nystrom_batch_size: int = 2048,
    nystrom_eigenvalue_floor: float = 1e-12,
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
