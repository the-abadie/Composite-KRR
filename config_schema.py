"""The versioned configuration model; defaults and JSON Schema live here."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
UnitFraction = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
Text = Annotated[str, Field(min_length=1)]
Bounds = Annotated[list[PositiveFloat], Field(min_length=2, max_length=2)]
PCAComponents = PositiveInt | Fraction | Literal["mle"] | None
KernelType = Literal["rbf", "laplacian", "linear", "poly", "polynomial", "sigmoid", "cosine", "chi2", "additive_chi2"]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True, allow_inf_nan=False)


class DescriptorConfig(ConfigModel):
    id: Text = Field(description="Unique descriptor name used by kernel_products.")
    path: Text = Field(description="NPY file; relative paths are relative to this configuration file.")
    normalization: Literal["none", "passthrough", "standard", "log_standard"] = "standard"
    kernel: KernelType = "rbf"
    pca_components: PCAComponents = None
    pca_whiten: bool = False


class TargetConfig(ConfigModel):
    path: Text
    name: Text | Annotated[list[Text], Field(min_length=1)] = "target"
    normalization: Literal["none", "standard", "log", "log1p", "log_standard", "log1p_standard"] = "standard"


class RandomSplit(ConfigModel):
    mode: Literal["random"] = "random"
    n_samples: PositiveInt | None = Field(default=None, description="Total sample pool; null uses all samples.")
    n_train: PositiveInt | None = Field(default=None, description="Training plus CV pool size; exclusive with train_fraction.")
    train_fraction: Fraction | None = None
    n_folds: Annotated[StrictInt, Field(ge=2)] = 5
    stratify: bool = False
    n_strata: PositiveInt = 5

    @model_validator(mode="after")
    def consistent_counts(self):
        if (self.n_train is None) == (self.train_fraction is None):
            raise ValueError("Specify exactly one of split.n_train or split.train_fraction.")
        n_train = self.training_count(self.n_samples) if self.n_samples is not None else self.n_train
        if n_train is not None:
            if n_train < self.n_folds:
                raise ValueError("split training count must be at least n_folds.")
            if self.n_samples is not None and n_train >= self.n_samples:
                raise ValueError("split must leave at least one held-out test sample.")
            if self.stratify and self.n_strata > n_train:
                raise ValueError("split.n_strata cannot exceed the training count.")
        return self

    def training_count(self, pool_size: int) -> int:
        return self.n_train if self.n_train is not None else int(pool_size * self.train_fraction)


class PredefinedSplitConfig(ConfigModel):
    mode: Literal["predefined"]
    train_indices: Text
    validation_folds: Text
    test_indices: Text


class NystromConfig(ConfigModel):
    n_landmarks: PositiveInt = 2048
    landmark_selection: Literal["random", "first"] = "random"
    batch_size: PositiveInt = 2048
    eigenvalue_floor: Annotated[float, Field(ge=0)] = 1e-12


class ModelConfig(ConfigModel):
    backend: Literal["exact", "nystrom"] = "exact"
    nystrom: NystromConfig = Field(default_factory=NystromConfig)


class RandomSearchConfig(ConfigModel):
    stage1: PositiveInt = 75
    stage2: PositiveInt = 75
    stage3: NonnegativeInt = 0


class BayesianConfig(ConfigModel):
    trials: NonnegativeInt = 50
    timeout: PositiveFloat | None = None
    patience: PositiveInt | None = 25
    batch_size: NonnegativeInt = 1


class SearchConfig(ConfigModel):
    alpha_bounds: Bounds = Field(default_factory=lambda: [1e-9, 1e2])
    gamma_bounds: Bounds | Annotated[list[Bounds], Field(min_length=1)] = Field(default_factory=lambda: [1e-9, 1e2])
    random: RandomSearchConfig = Field(default_factory=RandomSearchConfig)
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    top_k_fraction: UnitFraction = 0.25
    top_k_min_candidates: PositiveInt = 5
    gamma_prior_max_samples: PositiveInt | None = 5000
    scoring: Text = "neg_mean_absolute_error"

    @model_validator(mode="after")
    def increasing_bounds(self):
        gamma_bounds = self.gamma_bounds if isinstance(self.gamma_bounds[0], list) else [self.gamma_bounds]
        for bounds in [self.alpha_bounds, *gamma_bounds]:
            if bounds[0] >= bounds[1]:
                raise ValueError("Search bounds must satisfy 0 < low < high.")
        return self


class DistanceCacheConfig(ConfigModel):
    enabled: bool = True
    block_size: PositiveInt = 2048
    dtype: Literal["float32", "float64"] | None = Field(default=None, description="null inherits execution.dtype.")
    n_jobs: StrictInt | None = -1
    memory_fraction: UnitFraction = 0.8

    @model_validator(mode="after")
    def valid_jobs(self):
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be null or a non-zero integer.")
        return self


class TorchConfig(ConfigModel):
    device: Text | None = "auto"
    devices: Text | Annotated[list[Text], Field(min_length=1)] | None = None
    candidate_batch_size: PositiveInt = 1
    predict_batch_size: PositiveInt = 2048


class ExecutionConfig(ConfigModel):
    backend: Literal["numpy", "pytorch"] = "numpy"
    dtype: Literal["float32", "float64"] = "float64"
    n_jobs: StrictInt | None = -1
    blas_threads: PositiveInt | None = 1
    distance_cache: DistanceCacheConfig = Field(default_factory=DistanceCacheConfig)
    torch: TorchConfig = Field(default_factory=TorchConfig)

    @model_validator(mode="after")
    def valid_jobs(self):
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be null or a non-zero integer.")
        return self


class ReportingConfig(ConfigModel):
    kernel_contributions: bool = False
    contribution_bayesian_trials: NonnegativeInt | None = None


class OutputConfig(ConfigModel):
    directory: Text
    overwrite: bool = False


class RunConfig(ConfigModel):
    schema_version: Literal[1] = 1
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] | None = 1
    run_name: Text = "krr-run"
    verbosity: Annotated[StrictInt, Field(ge=0, le=2)] = 1
    descriptors: Annotated[list[DescriptorConfig], Field(min_length=1)]
    target: TargetConfig
    split: Annotated[RandomSplit | PredefinedSplitConfig, Field(discriminator="mode")]
    kernel_products: list[Annotated[list[Text], Field(min_length=2)]] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    output: OutputConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1.")
        return value

    @model_validator(mode="after")
    def consistent_run(self):
        names = [descriptor.id for descriptor in self.descriptors]
        if len(set(names)) != len(names):
            raise ValueError("Descriptor ids must be unique.")
        seen = set()
        for factors in self.kernel_products:
            unknown = set(factors) - set(names)
            if unknown:
                raise ValueError(f"Unknown kernel product descriptor ids: {sorted(unknown)}")
            key = tuple(sorted(factors))
            if key in seen:
                raise ValueError(f"Duplicate kernel product: {factors}")
            seen.add(key)
        bounds = self.search.gamma_bounds
        if isinstance(bounds[0], list) and len(bounds) != len(names):
            raise ValueError("Per-descriptor gamma bounds must match the descriptor count.")
        if self.execution.backend == "pytorch" or self.model.backend == "nystrom":
            if any(d.kernel not in {"rbf", "laplacian"} for d in self.descriptors):
                raise ValueError("PyTorch and Nystrom require rbf or laplacian base kernels.")
        return self
