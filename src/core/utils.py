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


class SingleEstimatorPrefixes(StrEnum):
    ACT_ADD = 'act_add_'
    H3_CONSENSUS = 'h3_consensus_'
    H3_POOLED = 'h3_pooled_'


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


def check_text_len_bounds(
    text: str, 
    min_len_chars: int,
    min_len_words: int,
    max_len_words: int | None,
) -> bool:
    num_chars = len(text)
    num_words = len(text.split())
    above_lower_bounds = min_len_chars <= num_chars and min_len_words <= num_words
    below_upper_bounds = num_words <= max_len_words if max_len_words is not None else True
    return above_lower_bounds and below_upper_bounds


def save_jsonl(
    text_iter: Iterable[str],
    out_dir: Path,
    out_filename: str,
    min_len_chars: int = 0,
    min_len_words: int = 0,
    max_len_words: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename += '.jsonl'
    out_path = out_dir / out_filename
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for text in text_iter:
            if not check_text_len_bounds(text, min_len_chars, min_len_words, max_len_words):
                continue
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
    in_dir: Path,
    filename: str,
) -> dict | list:
    with open(in_dir / f'{filename}.json', 'r', encoding='utf-8') as file:
        return json.load(file)