"""Inference from saved exact-KRR coefficients, without solving or searching again."""
import numpy as np

from class_CompositeKRR import CompositeKRR, KernelComponent
from kernel_mixing import resolve_kernel_products
from preprocess import make_data_preprocessor, make_target_preprocessor, descriptor_preprocessing


class SavedExactModel:
    """Rebuild training-only transforms around an existing exact solution.

    Callers must verify the source artifacts and reproduce saved predictions before
    treating external predictions as valid. PCA and linear baselines are rejected
    because their fitted state is not fully captured by the coefficient artifact.
    """

    def __init__(self, spec, train_blocks, train_targets, coefficients, gammas,
                 weights, alpha, *, backend="numpy", device="cpu"):
        if spec.model.backend != "exact" or spec.model.linear_baseline is not None:
            raise ValueError("Saved inference requires exact KRR without a linear baseline")
        if any(d.pca_components is not None for d in spec.descriptors):
            raise ValueError("Saved inference requires descriptors without PCA")
        if backend not in {"numpy", "pytorch"}:
            raise ValueError("backend must be numpy or pytorch")
        self.dtype = np.dtype(spec.execution.dtype)
        y = np.asarray(train_targets, dtype=self.dtype)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError("Saved inference currently requires finite scalar targets")
        coef = np.asarray(coefficients, dtype=self.dtype).reshape(-1, 1)
        if coef.shape != (len(y), 1) or not np.isfinite(coef).all():
            raise ValueError("Invalid saved coefficients")
        names = [d.id for d in spec.descriptors]
        products = resolve_kernel_products(
            [[names.index(n) for n in term] for term in spec.kernel_products], len(names)
        )
        gammas = np.asarray(gammas, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if gammas.shape != (len(names),) or weights.shape != (len(names) + len(products),):
            raise ValueError("Saved parameter counts differ from the configuration")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid saved kernel weights")
        # runner.py always uses normalize_kernel_weights=True.
        weights = weights / weights.sum()
        components = [KernelComponent(d.id, float(g), float(w), d.kernel)
                      for d, g, w in zip(spec.descriptors, gammas, weights[:len(names)], strict=True)]
        self.model = CompositeKRR(components, float(alpha), self.dtype, products,
                                  weights[len(names):].tolist())
        if len(train_blocks) != len(names):
            raise ValueError("Training block count differs from the configuration")
        self.transforms = []
        self.train_blocks = []
        for d, block in zip(spec.descriptors, train_blocks, strict=True):
            block = np.asarray(block, dtype=self.dtype).reshape(len(y), -1)
            if not np.isfinite(block).all():
                raise ValueError("Non-finite training descriptors")
            transform = make_data_preprocessor(descriptor_preprocessing(d))
            self.train_blocks.append(np.asarray(transform.fit_transform(block), dtype=self.dtype))
            self.transforms.append(transform)
        self.target_transform = make_target_preprocessor(spec.target.normalization)
        self.target_transform.fit(y.reshape(-1, 1))
        self.coefficients = coef
        self.backend = backend
        if backend == "pytorch":
            from torch_exact_krr import _resolve_torch
            self.torch, self.torch_dtype, self.device = _resolve_torch(dtype=self.dtype, device=device)
            self.train_tensors = [self.torch.as_tensor(b, dtype=self.torch_dtype, device=self.device)
                                  for b in self.train_blocks]
            self.coef_tensor = self.torch.as_tensor(coef, dtype=self.torch_dtype, device=self.device)

    def predict(self, blocks):
        if len(blocks) != len(self.transforms):
            raise ValueError("Evaluation block count differs from the configuration")
        n = len(blocks[0])
        transformed = []
        for transform, block, train in zip(self.transforms, blocks, self.train_blocks, strict=True):
            block = np.asarray(block, dtype=self.dtype)
            if len(block) != n or not n:
                raise ValueError("Evaluation blocks must have the same nonzero sample count")
            block = block.reshape(n, -1)
            if not np.isfinite(block).all():
                raise ValueError("Invalid evaluation descriptors")
            values = np.asarray(transform.transform(block), dtype=self.dtype)
            if values.shape[1] != train.shape[1]:
                raise ValueError("Invalid transformed evaluation descriptor width")
            transformed.append(values)
        if self.backend == "numpy":
            pred = self.model._composite_kernel(transformed, self.train_blocks) @ self.coefficients
        else:
            from torch_exact_krr import _composite_kernel_torch
            with self.torch.no_grad():
                tensors = [self.torch.as_tensor(b, dtype=self.torch_dtype, device=self.device)
                           for b in transformed]
                kernel = _composite_kernel_torch(
                    tensors, self.train_tensors, self.model.components, torch=self.torch,
                    kernel_products=self.model.kernel_products, product_weights=self.model.product_weights,
                )
                pred = (kernel @ self.coef_tensor).cpu().numpy()
        pred = self.target_transform.inverse_transform(pred).ravel()
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite saved-model predictions")
        return pred
