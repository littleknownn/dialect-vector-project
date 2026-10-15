import json
from pathlib import Path

import torch

from src.core.paths import (
    METRICS_DIR,
    PAIR_EDIT_DATA_PREFIX,
    Q1_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    Q4_RESULTS_PREFIX,
    get_sweep_identifier,
)
from src.core.utils import (
    PairEditData,
    PoolingTypeEnum,
    Q1Results,
    Q4Results,
    load_json,
)


def load_pair_edit_data(
    corpus_name: str,
    metrics_dir: Path = METRICS_DIR,
) -> list[PairEditData]:
    return load_json(metrics_dir, f'{PAIR_EDIT_DATA_PREFIX}{corpus_name}')


def load_q1_projections(
    corpus_name: str,
    layer: int,
    pooling_type: PoolingTypeEnum,
    metrics_dir: Path = METRICS_DIR,
) -> torch.Tensor:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    projections = torch.load(metrics_dir / f'{Q1_PROJECTIONS_PREFIX}{corpus_name}_{sweep_identifier}.pt', weights_only=True)
    return projections


def load_q1_results(
    corpus_name: str,
    layer: int,
    pooling_type: PoolingTypeEnum,
    metrics_dir: Path = METRICS_DIR,
) -> Q1Results:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    results = load_json(metrics_dir, f'{Q1_RESULTS_PREFIX}{corpus_name}_{sweep_identifier}')
    return results


def load_q4_results(
    corpus_name: str,
    layer: int,
    pooling_type: PoolingTypeEnum,
    metrics_dir: Path = METRICS_DIR,
) -> Q4Results:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    results = load_json(metrics_dir, f'{Q4_RESULTS_PREFIX}{corpus_name}_{sweep_identifier}')
    return results