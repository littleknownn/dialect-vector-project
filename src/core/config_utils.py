from dataclasses import dataclass
from pathlib import Path
import re
from typing import NamedTuple

from src.core.utils import (
    DifferenceVector,
    PoolingType,
    Q3EditCount,
    Q4CurveFunction,
    Q4EditKind,
    SplitSuffixes,
)


LOGGED_EXPERIMENT_CONFIG_FIELDS = (
    'model_hf_path',
    'attn_implementation',
    'load_in_4bit',
    'bnb_4bit_quant_type',
    'target_layers',
    'primary_layer',
    'debug_max_split_lines',
    'batch_size',
    'max_seq_len',
)


class ModelSpec(NamedTuple):
    model_hf_path: str
    primary_layer: int
    attn_implementation: str | None = None


class Q1Thresholds(NamedTuple):
    positive_projection_ratio: float
    standardized_mean_ci_lower: float


class Q1LabeledThresholds(NamedTuple):
    consistent: Q1Thresholds
    weak: Q1Thresholds


class Q4RegressionSettings(NamedTuple):
    free_shape_breakpoints: tuple[float, ...]
    initial_parameters_by_curve: dict[Q4CurveFunction, tuple[float, ...]]
    initial_distance_decay_exponent: float
    initial_size_exponent: float


class Q4OrthographicThresholds(NamedTuple):
    strict: float
    relaxed: float
    loose: float


@dataclass
class Q1Config:
    labeled_bounds: Q1LabeledThresholds
    fdr_vectors: tuple[DifferenceVector, ...]
    layer_sweep_vectors: tuple[DifferenceVector, ...]
    num_sign_flips: int
    num_shuffled_pairings: int


@dataclass
class Q2Config:
    primary_corpus_names: tuple[str, ...]
    reported_corpus_names: tuple[str, ...]
    midpoint_corpus_names: tuple[str, ...]
    primary_pooling_types: tuple[PoolingType, ...]

    num_negatives_per_positive: int
    num_null_shuffles: int
    min_auroc: float
    power: float
    sensitivity_vectors: tuple[DifferenceVector, ...]


@dataclass
class Q3Config:
    edits_by_count: dict[Q3EditCount, tuple[Q4EditKind, ...]]
    num_dose_bins: int


@dataclass
class TaggingConfig:
    orthographic_thresholds: Q4OrthographicThresholds
    respelling_threshold: float
    single_letter_respelling_threshold: float
    aint_words: tuple[str, ...]
    function_words: tuple[str, ...]
    grammar_words: tuple[str, ...]
    copula_words: tuple[str, ...]


@dataclass
class Q4Config:
    primary_curves: tuple[Q4CurveFunction, ...]
    format_flags: tuple[str, ...]
    regression_settings: Q4RegressionSettings
    min_edits_for_verdict: int
    num_held_out_folds: int
    num_calibration_bins: int
    num_bootstrap_samples: int
    num_robustness_bootstrap_samples: int
    robustness_distance_shift: int
    marginal_effect_distances: tuple[int, ...]


@dataclass
class ExperimentConfig:
    model_hf_path: str
    attn_implementation: str | None
    load_in_4bit: bool
    bnb_4bit_quant_type: str

    corpus_split_size: int
    batch_size: int
    max_seq_len: int
    random_seed: int

    target_layers: list[int]
    primary_layer: int
    primary_pooling_type: PoolingType

    num_cv_folds: int
    num_bootstrap_samples: int
    p_alpha: float

    q1: Q1Config
    q2: Q2Config
    q3: Q3Config
    q3_q4_pooling_types: tuple[PoolingType, ...]
    tagging: TaggingConfig
    q4: Q4Config
    
    debug_max_split_lines: int | None


@dataclass
class CoraalSourceConfig:
    in_dir: Path
    header: bool
    delimiter: str

    text_col_idx: int
    speaker_col_idx: int
    interviewer_marker: str
    pause_pattern: str


@dataclass
class GithubSourceConfig:
    git_url: str
    delimiter: str


@dataclass
class HFSourceConfig:
    hf_path: str
    split: str
    text_col: str
    hf_name: str | None = None


@dataclass
class RealAaeSourceConfig:
    in_dir: Path
    header: bool
    delimiter: str
    
    aae_col_idx: int
    sae_col_idx: int
    aae_back_col_idx: int


@dataclass(kw_only=True)
class DatasetConfig:
    source_config: CoraalSourceConfig | GithubSourceConfig | HFSourceConfig | RealAaeSourceConfig

    min_len_chars: int
    min_len_words: int
    max_len_words: int

    skipped_data_patterns: list[str | re.Pattern]
    skipped_case_patterns: list[str | re.Pattern]
    replacement_patterns: list[tuple[str | re.Pattern, str]]

    sentencize: bool
    pool_size: int | None = None

    def __post_init__(self):
        self.skipped_data_patterns = [re.compile(pattern) for pattern in self.skipped_data_patterns]
        self.skipped_case_patterns = [re.compile(pattern) for pattern in self.skipped_case_patterns]
        self.replacement_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.replacement_patterns
        ]


@dataclass
class BaselineDatasetConfig(DatasetConfig):
    pass


@dataclass(kw_only=True)
class SupervisedDatasetConfig(DatasetConfig):
    boundary_min_previous_words: int
    boundary_min_following_words: int


class SupervisedPair(NamedTuple):
    corpus_name: str
    aae_split: str
    sae_split: str


class ModelConfigs(NamedTuple):
    corpus_config: dict[str, DatasetConfig]
    experiment_config: ExperimentConfig


class EstimatorData(NamedTuple):
    supervised_split_pairs: list[SupervisedPair]
    aae_corpora_split_names: list[str]


def get_dataset_config(
    split_name: str,
    corpus_config: dict[str, DatasetConfig],
) -> DatasetConfig:
    if split_name in corpus_config:
        return corpus_config[split_name]
    for corpus_name, config in corpus_config.items():
        if split_name.startswith(corpus_name):
            return config


def get_estimator_data(corpus_config: dict[str, DatasetConfig]) -> EstimatorData:
    supervised_split_pairs = []
    aae_corpora_split_names = []

    for corpus_name, config in corpus_config.items():
        if isinstance(config, BaselineDatasetConfig):
            continue
        if isinstance(config, SupervisedDatasetConfig):
            supervised_split_pairs.append(
                SupervisedPair(
                    corpus_name = corpus_name,
                    aae_split = f'{corpus_name}{SplitSuffixes.AAE}{SplitSuffixes.SUPERVISED}',
                    sae_split = f'{corpus_name}{SplitSuffixes.SAE}{SplitSuffixes.SUPERVISED}',
                )
            )
        else:
            aae_corpora_split_names.append(corpus_name)

    estimator_data = EstimatorData(
        supervised_split_pairs,
        aae_corpora_split_names,
    )
    return estimator_data
