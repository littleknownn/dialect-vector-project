from enum import StrEnum
from pathlib import Path
import json
from typing import (
    Any,
    Iterable,
    NamedTuple,
    TypedDict,
    NotRequired,
)

import torch
import torch.nn as nn


type MetricsJson = dict[
    str, dict[
        str, dict[
            str, dict
        ]
    ]
]


type GeometricSummary = list[GeometricMetricsData]


type HiddenStateData = tuple[torch.Tensor, *tuple[Any, ...]]


class PoolingTypeEnum(StrEnum):
    LAST = 'last'
    MEAN = 'mean'
    BOUNDARY = 'boundary'

    @property
    def is_paired_only(self) -> bool:
        return self is PoolingTypeEnum.BOUNDARY


class SingleEstimatorPrefixes(StrEnum):
    ACT_ADD = 'act_add_'
    H3_CONSENSUS = 'h3_consensus_'
    H3_POOLED = 'h3_pooled_'


class SplitSuffixes(StrEnum):
    AAE = '_aae'
    SAE = '_sae'
    AAE_BACK = '_aae_back'
    PAIR_METADATA = '_pair_metadata'
    SUPERVISED = '_supervised'
    TESTING = '_testing'
    RESERVE = '_reserve'
    STRIPPED = '_stripped'
    POOL = '_pool'
    FEATURE_INDICES = '_feature_indices'


class MetricTypeEnum(StrEnum):
    LLR_SHIFT = 'llr_shift'
    PERPLEXITY = 'perplexity'


class Q1ConsistencyLabel(StrEnum):
    CONSISTENT = 'consistent'
    PARTIAL = 'partial'
    WEAK = 'weak'
    POOLING_DEPENDENT = 'pooling_dependent'
    
class Q4Stratum(StrEnum):
    IDENTICAL = 's0_identical'
    FORMAT_ONLY = 's1_format_only'
    ORTHOGRAPHIC = 's2_orthographic'
    FUNCTION_WORD = 's3_function_word'
    LEXICAL = 's4_lexical'


class PairBoundary(TypedDict):
    aae_idx: int
    sae_idx: int
    aae_back_idx: int | None
    is_exact: bool
    is_aae_back_shared: bool


class PairMetadata(TypedDict):
    row_idx: int
    is_sae_typography_normalized: bool
    is_aae_back_typography_normalized: bool
    is_aae_back_valid: bool
    boundary: PairBoundary | None


class LinesProcessingCounts(TypedDict):
    num_cleaned: int
    num_out_of_bounds: int
    num_duplicates: int
    num_kept: int


class PairedCorpusProcessingCounts(TypedDict):
    num_rows: int
    num_pairs_kept: int
    num_aae_back_invalid: int
    num_dropped_by_split: dict[str, int]


class ProcessingLog(TypedDict):
    paired_corpus_counts: dict[str, PairedCorpusProcessingCounts]
    single_corpus_counts: dict[str, LinesProcessingCounts]
    split_sizes: dict[str, int]


class CIStats(TypedDict):
    estimate: float
    ci_lower: float
    ci_upper: float


class SweepMetric(TypedDict):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int

    corpus_name: str
    estimator_name: str
    alpha: float

    metric: MetricTypeEnum
    value: float

    ci_lower: NotRequired[float]
    ci_upper: NotRequired[float]
    p_value: NotRequired[float]


class GeometricMetricsData(TypedDict):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_a_name: str
    estimator_b_name: str
    principal_angles: float
    chordal_distances: float


class AucSummary(TypedDict):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricTypeEnum
    auc_pos: float
    auc_neg: float


class FdrSummary(TypedDict):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricTypeEnum
    alpha: float
    p_value_fdr: float


class BehavioralSummary(TypedDict):
    auc: list[AucSummary]
    fdr: list[FdrSummary]


class SweepSummary(TypedDict):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricTypeEnum
    alpha: float
    value: float
    p_value: float


class H1Results(TypedDict):
    supervised_vector_norm: float
    floors: dict[str, list[float]]
    pair_distances: dict[str, list[float]]
    null_distances: list[float]
    projection_energies: dict[str, list[float]]


class Q1Results(TypedDict):
    corpus_name: str
    pooling_type: PoolingTypeEnum
    num_pairs: int
    mean_projection: CIStats
    positive_projection_ratio: CIStats
    standardized_projection_mean: CIStats
    label: Q1ConsistencyLabel
    identical_pairs_median_abs_projection: float
    other_pairs_median_abs_projection: float


class Q4Results(TypedDict):
    corpus_name: str
    pooling_type: PoolingTypeEnum
    num_pairs_by_stratum: dict[Q4Stratum, int]
    mean_projection_by_stratum: dict[Q4Stratum, CIStats]
    
    group_shares: dict[str, CIStats]
    r_squared: float
    unique_r_squared: dict[str, CIStats]
    contributing_groups: list[str]
    dominating_groups: list[str]


class PairEditData(TypedDict):
    stratum: Q4Stratum
    num_orthographic_edits: int
    num_function_word_edits: int
    num_lexical_edits: int
    ending_changed: bool
    case_changed: bool
    tag_count_changed: bool
    has_asterisk: bool
    num_words_aae: int
    num_words_sae: int


class ExtractionLog(TypedDict):
    model_hf_path: str
    torch_version: str
    transformers_version: str
    experiment_config: dict[str, Any]
    processed_data_hashes: dict[str, str]
    num_lines_by_split: dict[str, int]
    num_truncated_lines_by_split: dict[str, int]


class GeometricMetricsPair(NamedTuple):
    principal_angles: float
    chordal_distance: float


class SweepKey(NamedTuple):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricTypeEnum


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


class Q4TagsPerGroup(NamedTuple):
    orthographic: list[str]
    function_word: list[str]
    lexical: list[str]
    format: list[str]
    length: list[str]


class EstimatorSteering:
    def __init__(
        self,
        estimator_matrix: torch.Tensor,
        layer_module: int,
        alpha: float,
        act_add: bool = False,
    ):
        self.estimator_matrix = estimator_matrix
        self.layer_module = layer_module
        self.alpha = alpha
        self.act_add = act_add
        self.forward_hook = None

        if act_add:
            assert estimator_matrix.shape[1] == 1, 'ActAdd Steering requires a Rank-1 Estimator'

    def __enter__(self):
        self.forward_hook = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.forward_hook is not None:
            self.forward_hook.remove()

    def _hook(
        self,
        target_layer: nn.Module,
        previous_layer_inputs: HiddenStateData,
        previous_layer_outputs: torch.Tensor | HiddenStateData,
    ) -> torch.Tensor | HiddenStateData:
        is_output_wrapped = isinstance(previous_layer_outputs, tuple)
        hidden_states = previous_layer_outputs[0] if is_output_wrapped else previous_layer_outputs

        if self.act_add:
            steered_hidden_states = hidden_states + self.alpha * self.estimator_matrix.T
        else:
            subspace_projection = (hidden_states @ self.estimator_matrix) @ self.estimator_matrix.T
            steered_hidden_states = hidden_states + (self.alpha * subspace_projection)

        if is_output_wrapped:
            return (steered_hidden_states, *previous_layer_outputs[1:])
        return steered_hidden_states


def save_jsonl(
    text_iter: Iterable[str],
    out_dir: Path,
    out_filename: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename += '.jsonl'
    out_path = out_dir / out_filename
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for text in text_iter:
            json_object = {'text': text}
            out_file.write(f'{json.dumps(json_object, ensure_ascii=False)}\n')


def load_jsonl(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as corpus_file:
        lines = [json.loads(line)['text'] for line in corpus_file]
    return lines


def save_json(
    data: dict | list,
    out_dir: Path,
    out_filename: str,
    json_indent: int = 4,
) -> None:
    out_filepath = out_dir / f'{out_filename}.json'
    with open(out_filepath, 'w') as file:
        json.dump(data, file, indent=json_indent)


def load_json(
    filepath: Path,
) -> dict | list:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_boundary_char_indices_by_split(
    in_dir: Path,
) -> dict[str, list[int | None]]:
    boundary_keys = {
        SplitSuffixes.AAE: 'aae_idx', #
        SplitSuffixes.SAE: 'sae_idx', #
        SplitSuffixes.AAE_BACK: 'aae_back_idx', #
    }
    boundary_char_indices_by_split = {}
    for metadata_filepath in  in_dir.glob(f'*{SplitSuffixes.PAIR_METADATA}*.json'):
        corpus_name, split_suffix = metadata_filepath.stem.split(SplitSuffixes.PAIR_METADATA)
        pair_metadata = load_json(metadata_filepath)
        for dialect_suffix, boundary_key in boundary_keys.items():
            split_name = f'{corpus_name}{dialect_suffix}{split_suffix}'
            boundary_char_indices_by_split[split_name] = [
                metadata['boundary'][boundary_key]
                if metadata['boundary'] else None
                for metadata in pair_metadata
            ]
    return boundary_char_indices_by_split


def get_boundary_token_idx(
    token_char_indices: torch.Tensor,
    boundary_char_idx: int,
) -> int:
    token_end_char_indices = token_char_indices[:, 1]
    is_before_boundary = (
        (token_end_char_indices > 0)
        & (token_end_char_indices <= boundary_char_idx)
    )
    boundary_token_idx = int(is_before_boundary.nonzero().max())
    return boundary_token_idx
