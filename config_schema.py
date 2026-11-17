"""The versioned configuration model; defaults and JSON Schema live here."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Seed = Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
Fraction = Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
UnitFraction = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
Text = Annotated[str, Field(min_length=1)]
Bounds = Annotated[list[PositiveFloat], Field(min_length=2, max_length=2)]
PCAComponents = PositiveInt | Fraction | Literal["mle"] | None
KernelType = Literal["rbf", "laplacian", "linear", "poly", "polynomial", "sigmoid", "cosine", "chi2", "additive_chi2"]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True, allow_inf_nan=False)


class SpatialDCTConfig(ConfigModel):
    block_size: PositiveInt = Field(description="Spatial bins per contiguous channel before DCT-II.")
    n_components: PositiveInt = Field(description="Retained orthonormal DCT-II modes per channel.")
    post_normalization: Literal["none", "standard"] = Field(default="none", description="Optional training-fold coefficient standardization after DCT truncation.")

    @model_validator(mode="after")
    def valid_width(self):
        if self.n_components > self.block_size:
            raise ValueError("spatial_dct.n_components cannot exceed block_size.")
        return self


class SpatialFFTConfig(ConfigModel):
    block_size: PositiveInt = Field(description="Spatial bins per contiguous channel before orthonormal real FFT.")
    n_frequencies: PositiveInt = Field(description="Retained nonnegative frequencies per channel, including DC.")
    post_normalization: Literal["none", "standard"] = Field(default="none", description="Optional training-fold scaling of retained packed Fourier coordinates.")

    @model_validator(mode="after")
    def valid_width(self):
        if self.n_frequencies > self.block_size // 2 + 1:
            raise ValueError("spatial_fft.n_frequencies exceeds the real FFT frequency count.")
        return self


class DescriptorConfig(ConfigModel):
    id: Text = Field(description="Unique descriptor name used by kernel_products.")
    path: Text = Field(description="NPY file or compressed-shard JSON manifest; relative paths are relative to this configuration file.")
    archive_key: Text | None = Field(default=None, description="Array or virtual concatenation key in a descriptor shard manifest.")
    normalization: Literal["none", "passthrough", "standard", "log_standard"] = "standard"
    kernel: KernelType = "rbf"
    pca_components: PCAComponents = None
    pca_whiten: bool = False
    spatial_dct: SpatialDCTConfig | None = Field(default=None, description="Fixed channel-wise DCT after fold-fitted normalization, optional fold-fitted coefficient scaling, then optional PCA.")
    spatial_fft: SpatialFFTConfig | None = Field(default=None, description="Fixed channel-wise orthonormal FFT after fold-fitted normalization; complex coordinates share the descriptor kernel.")

    @model_validator(mode="after")
    def one_spatial_transform(self):
        if self.spatial_dct is not None and self.spatial_fft is not None:
            raise ValueError("Choose only one of spatial_dct and spatial_fft.")
        return self


class TargetConfig(ConfigModel):
    path: Text
    name: Text | Annotated[list[Text], Field(min_length=1)] = "target"
    normalization: Literal["none", "standard", "log", "log1p", "log_standard", "log1p_standard", "boxcox_standard", "yeojohnson_standard", "yeojohnson_bounded_standard", "negate_log_standard", "negate_boxcox_standard", "asinh_iqr_standard", "signed_sqrt_standard", "cbrt_standard", "signed_log_iqr_standard", "asinh_median_iqr_standard"] = "standard"


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
    linear_baseline: Text | None = Field(default=None, description="Descriptor ID for a fold-fitted affine baseline; exact KRR fits residuals.")


class RandomSearchConfig(ConfigModel):
    stage1: PositiveInt = 200
    stage2: PositiveInt = 200
    stage3: NonnegativeInt = 50


class TPESamplerConfig(ConfigModel):
    multivariate: bool = True
    group: bool = False
    n_startup_trials: NonnegativeInt = 10
    n_ei_candidates: PositiveInt = 24
    constant_liar: bool = False
    consider_endpoints: bool = False
    consider_magic_clip: bool = True
    prior_weight: PositiveFloat = 1.0

    @model_validator(mode="after")
    def grouping_requires_multivariate(self):
        if self.group and not self.multivariate:
            raise ValueError("TPE group requires multivariate=true.")
        return self


class BayesianConfig(ConfigModel):
    trials: NonnegativeInt = 200
    rounds: PositiveInt = Field(default=4, description="Split the total trial cap across fresh local Bayesian rounds; one reproduces the historical single-box search.")
    recenter: bool = Field(default=True, description="Move each new round's search box to the current-run incumbent. False restarts samplers in the original fixed box.")
    reuse_observations: bool = Field(default=True, description="Seed TPE with eligible completed CV observations from earlier stages/rounds of this same run; never imports another run.")
    timeout: PositiveFloat | None = None
    patience: PositiveInt | None = 75
    batch_size: NonnegativeInt = 1
    refine_decades: PositiveFloat = 0.6
    weight_logit_radius: PositiveFloat = 3.0
    min_delta_absolute: NonnegativeFloat = 0.0
    min_delta_relative: NonnegativeFloat = 0.0
    sampler: TPESamplerConfig = Field(default_factory=TPESamplerConfig)

    @model_validator(mode="after")
    def valid_round_budget(self):
        if self.trials and self.rounds > self.trials:
            raise ValueError("Bayesian rounds cannot exceed the positive total trial cap.")
        return self


class SearchConfig(ConfigModel):
    seed: Seed | None = Field(default=None, description="Optimizer seed; null inherits the run/split seed.")
    history_batch_size: PositiveInt = Field(default=32, description="Maximum cached candidates per durable history checkpoint; rounded up to the GPU candidate batch size.")
    gamma_prior_decades: PositiveFloat = 2.5
    refine_decades: PositiveFloat = 1.0
    top_k_padding_decades: PositiveFloat = 0.5
    stage3_weight_concentration: PositiveFloat = 25.0
    alpha_bounds: Bounds = Field(default_factory=lambda: [1e-9, 1e2])
    gamma_bounds: Bounds | Annotated[list[Bounds], Field(min_length=1)] = Field(default_factory=lambda: [1e-9, 1e2])
    random: RandomSearchConfig = Field(default_factory=RandomSearchConfig)
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    top_k_fraction: UnitFraction = 0.25
    top_k_min_candidates: PositiveInt = 5
    gamma_prior_max_samples: PositiveInt | None = 5000
    scoring: Text = "neg_mean_absolute_error"
    target_space: Literal["original", "transformed"] = Field(default="original", description="Score in original units, or after the target transform fitted on each CV training fold.")

    @model_validator(mode="after")
    def increasing_bounds(self):
        gamma_bounds = self.gamma_bounds if isinstance(self.gamma_bounds[0], list) else [self.gamma_bounds]
        for bounds in [self.alpha_bounds, *gamma_bounds]:
            if bounds[0] >= bounds[1]:
                raise ValueError("Search bounds must satisfy 0 < low < high.")
        return self


class DistanceCacheConfig(ConfigModel):
    enabled: bool = True
    storage: Literal["device", "host"] = Field(
        default="device", exclude_if=lambda value: value == "device",
        description="For exact PyTorch CV, host keeps pairwise distances in CPU RAM and streams each block to the scoring device.",
    )
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
    warmup_size: NonnegativeInt = Field(default=0, description="Optional deterministic matrix warm-up in this run's process, before the training timer; zero disables it.")


class ExecutionConfig(ConfigModel):
    backend: Literal["numpy", "pytorch"] = "numpy"
    dtype: Literal["float32", "float64"] = "float64"
    n_jobs: StrictInt | None = -1
    blas_threads: PositiveInt | None = 1
    prediction_batch_size: PositiveInt = Field(default=2048, description="Held-out rows loaded, preprocessed, and predicted per batch, on either backend.")
    distance_cache: DistanceCacheConfig = Field(default_factory=DistanceCacheConfig)
    torch: TorchConfig = Field(default_factory=TorchConfig)

    @model_validator(mode="after")
    def valid_jobs(self):
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be null or a non-zero integer.")
        return self


class KernelAlignmentConfig(ConfigModel):
    enabled: bool = Field(default=False, description="Report centered alignment of the final fitted model's unweighted kernel terms using training/CV rows only.")
    max_samples: Annotated[StrictInt, Field(ge=3)] = Field(default=1000, description="Maximum uniformly sampled training/CV rows; sampled deterministically with the run seed. Kernel memory grows quadratically in this limit.")
    include_products: bool = Field(default=True, description="Include configured elementwise product terms after the base kernels, including terms with zero fitted weight.")
    include_target: bool = Field(default=False, description="Also report alignment with a linear target kernel for each target column on the same sampled training/CV rows. Requires enabled=true.")
    target_space: Literal["original", "transformed"] = Field(default="original", description="Target labels in original units or after the final fitted target transformer; independent of search.target_space. No baseline residual subtraction.")

    @model_validator(mode="after")
    def target_requires_alignment(self):
        if self.include_target and not self.enabled:
            raise ValueError("kernel_alignment.include_target requires kernel_alignment.enabled=true.")
        return self


class ReportingConfig(ConfigModel):
    kernel_contributions: bool = False
    contribution_bayesian_trials: NonnegativeInt | None = None
    plot_max_samples: PositiveInt = Field(default=20000, description="Maximum held-out rows in diagnostic plots; full predictions and metrics are always saved.")
    kernel_alignment: KernelAlignmentConfig = Field(default_factory=KernelAlignmentConfig)


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
        contribution_trials=self.reporting.contribution_bayesian_trials
        if (self.reporting.kernel_contributions and contribution_trials is not None
                and 0 < contribution_trials < self.search.bayesian.rounds):
            raise ValueError("Contribution Bayesian trial cap must cover the configured number of rounds, or be zero.")
        names = [descriptor.id for descriptor in self.descriptors]
        if len(set(names)) != len(names):
            raise ValueError("Descriptor ids must be unique.")
        if self.model.linear_baseline is not None:
            if self.model.linear_baseline not in names:
                raise ValueError("model.linear_baseline must name a descriptor ID.")
            if self.model.backend != "exact":
                raise ValueError("model.linear_baseline currently requires exact KRR.")
            if self.reporting.kernel_contributions:
                raise ValueError("Kernel contribution ablations currently require model.linear_baseline=null.")
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
        if self.execution.distance_cache.storage == "host" and (
            self.execution.backend != "pytorch" or self.model.backend != "exact"
        ):
            raise ValueError("Host distance-cache storage requires exact PyTorch KRR.")
        if self.execution.backend == "pytorch" or self.model.backend == "nystrom":
            if any(d.kernel not in {"rbf", "laplacian"} for d in self.descriptors):
                raise ValueError("PyTorch and Nystrom require rbf or laplacian base kernels.")
        return self
