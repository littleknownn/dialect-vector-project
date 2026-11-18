from pathlib import Path

import torch

from src.core.paths import (
    PAIR_EDIT_DATA_PREFIX,
    Q1_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    Q2_RESULTS_FILENAME,
    Q3_RESULTS_PREFIX,
    Q4_RESULTS_PREFIX,
    Q4_ROBUSTNESS_PREFIX,
    get_sweep_identifier,
)
from src.core.utils import (
    DifferenceVector,
    PairEditData,
    PoolingType,
    Q1Results,
    Q2Results,
    Q3Results,
    Q4Results,
    Q4RobustnessResults,
    SplitSuffixes,
    load_json,
)


def load_pair_edit_data(
    corpus_name: str,
    split_suffix: SplitSuffixes,
    metrics_dir: Path,
) -> list[PairEditData]:
    return load_json(metrics_dir / f'{PAIR_EDIT_DATA_PREFIX}{corpus_name}{split_suffix}.json')


def load_q1_projections(
    corpus_name: str,
    split_suffix: SplitSuffixes,
    layer: int,
    pooling_type: PoolingType,
    metrics_dir: Path,
) -> torch.Tensor:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    projections = torch.load(metrics_dir / f'{Q1_PROJECTIONS_PREFIX}{corpus_name}{split_suffix}_{sweep_identifier}.pt', weights_only=True)
    return projections


def load_q1_results(
    corpus_name: str,
    split_suffix: SplitSuffixes,
    metrics_dir: Path,
) -> list[Q1Results]:
    results_list = load_json(metrics_dir / f'{Q1_RESULTS_PREFIX}{corpus_name}{split_suffix}.json')
    return results_list


def select_q1_results(
    q1_results_list: list[Q1Results],
    vector: DifferenceVector,
    layer: int | None = None,
    pooling_type: PoolingType | None = None,
) -> list[Q1Results]:
    selected_results = [
        results for results in q1_results_list
        if results['vector'] == vector
        and (layer is None or results['layer'] == layer)
        and (pooling_type is None or results['pooling_type'] == pooling_type)
    ]
    return selected_results


def load_q2_results(
    metrics_dir: Path,
) -> list[Q2Results]:
    results_list = load_json(metrics_dir / f'{Q2_RESULTS_FILENAME}.json')
    return results_list


def select_q2_results(
    q2_results_list: list[Q2Results],
    **field_values,
) -> list[Q2Results]:
    selected_results = [
        results for results in q2_results_list
        if all(
            results[field] == value
            for field, value
            in field_values.items()
        )
    ]
    return selected_results


def load_q3_results(
    corpus_name: str,
    split_suffix: SplitSuffixes,
    metrics_dir: Path,
) -> list[Q3Results]:
    results_list = load_json(metrics_dir / f'{Q3_RESULTS_PREFIX}{corpus_name}{split_suffix}.json')
    return results_list


def load_q4_results(
    corpus_name: str,
    split_suffix: SplitSuffixes,
    layer: int,
    pooling_type: PoolingType,
    metrics_dir: Path,
) -> Q4Results:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    results = load_json(metrics_dir / f'{Q4_RESULTS_PREFIX}{corpus_name}{split_suffix}_{sweep_identifier}.json')
    return results


def load_q4_robustness_results(
    corpus_name: str,
    split_suffix: SplitSuffixes,
    layer: int,
    pooling_type: PoolingType,
    metrics_dir: Path,
) -> list[Q4RobustnessResults]:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    results_list = load_json(metrics_dir / f'{Q4_ROBUSTNESS_PREFIX}{corpus_name}{split_suffix}_{sweep_identifier}.json')
    return results_list
