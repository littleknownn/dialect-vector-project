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
    SVD = 'svd_'
    SUPERVISED = 'supervised_'
    ACT_ADD = 'act_add_'


class AggregateEstimators(StrEnum):
    CONTROL = 'control'
    CHORDAL_MEAN_CORPORA = 'chordal_mean_corpora'
    CHORDAL_MEAN_ALL = 'chordal_mean_all'
    LEACE_CORPORA = 'leace_corpora'
    LEACE_ALL = 'leace_all'


class MetricTypeEnum(StrEnum):
    LLR_SHIFT = 'llr_shift'
    PERPLEXITY = 'perplexity'


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
    procrustes_alignment: float
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


class VectorMagnitudeSummary(TypedDict):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str
    norm: float


class EstimatorKey(NamedTuple):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str


class SweepKey(NamedTuple):
    layer: int
    pooling_type: PoolingTypeEnum
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricTypeEnum


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