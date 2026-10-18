from dataclasses import dataclass
from pathlib import Path
import re
from typing import NamedTuple

from src.core.utils import (
    PoolingTypeEnum,
    SplitSuffixes,
)


class Q1Thresholds(NamedTuple):
    positive_projection_ratio: float
    standardized_mean_ci_lower: float


class Q1LabeledThresholds(NamedTuple):
    consistent: Q1Thresholds
    weak: Q1Thresholds


class Q4GroupLabelThresholds(NamedTuple):
    contribution_share_ci_lower: float
    contribution_unique_r_squared_ci_lower: float
    dominant_share: float


class Q4OrthographicThresholds(NamedTuple):
    strict: float
    relaxed: float
    loose: float


class Q4TagsPerGroup(NamedTuple):
    orthographic: list[str]
    function_word: list[str]
    lexical: list[str]
    format: list[str]
    length: list[str]


@dataclass
class ExperimentConfig:
    model_hf_path: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str

    target_layers: list[int]
    target_ranks: list[int]
    target_alphas: list[float]

    primary_layer: int
    primary_pooling_type: PoolingTypeEnum

    debug_max_split_lines: int | None

    h1_corpora: list[str]

    num_equal_n_repeats: int
    num_null_draws: int
    num_cv_folds: int
    q1_labeled_bounds: Q1LabeledThresholds

    q4_orthographic_thresholds: Q4OrthographicThresholds
    q4_function_words: tuple[str, ...]
    q4_tags_per_group: Q4TagsPerGroup
    q4_group_label_thresholds: Q4GroupLabelThresholds
    q4_log_centered_features: tuple[str, ...]

    corpus_split_size: int
    batch_size: int
    max_seq_len: int
    random_seed: int

    num_bootstrap_samples: int
    p_alpha: float


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
