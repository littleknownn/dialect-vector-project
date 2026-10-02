from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm
from scipy.stats import false_discovery_control
from scipy.linalg import (
    orthogonal_procrustes,
    subspace_angles,
)
import torch

from src.core.config import (
    ExperimentConfig,
    EXPERIMENT_CONFIG,
)
from src.core.paths import (
    BEHAVIORAL_RESULTS_FILENAME,
    ESTIMATORS_DIR,
    GEOMETRIC_RESULTS_FILENAME,
    METRICS_DIR,
    SWEEPS_DIR,
    SWEEP_RESULTS_FILENAME,
    VECTOR_MAGNITUDES_FILENAME,
    get_sweep_identifier,
)
from src.core.utils import (
    AucSummary,
    BehavioralSummary,
    EstimatorKey,
    FdrSummary,
    GeometricMetricsData,
    GeometricSummary,
    MetricTypeEnum,
    PoolingTypeEnum,
    SweepKey,
    SweepSummary,
    VectorMagnitudeSummary,
    save_json,
)


class SteeringCurvePoint(NamedTuple):
    alpha: float
    value: float
    p_value: float | None


class BehavioralResults(NamedTuple):
    auc_pos: dict[SweepKey, float]
    auc_neg: dict[SweepKey, float]
    p_values_fdr: dict[SweepKey, dict[float, float]]


class GeometricMetricsPair(NamedTuple):
    procrustes_alignment: float
    chordal_distance: float


def load_estimators(
    in_dir: Path,
    experiment_config: ExperimentConfig,
) -> dict[EstimatorKey, torch.Tensor]:
    estimators = {}
    sweep_tuples = itertools.product(
        experiment_config.target_layers,
        PoolingTypeEnum,
        experiment_config.target_ranks,
    )
    for layer, pooling_type, rank in sweep_tuples:
        sweep_identifier = get_sweep_identifier(layer, pooling_type, rank)

        for estimator_path in in_dir.glob(f'*_{sweep_identifier}.pt'):
            estimator_name = estimator_path.name.removesuffix(f'_{sweep_identifier}.pt')
            key = EstimatorKey(
                layer,
                pooling_type,
                rank,
                estimator_name,
            )
            estimators[key] = torch.load(estimator_path, weights_only=True)
    
    return estimators

def load_steering_metric_sweeps(in_dir: Path) -> dict[SweepKey, list[SteeringCurvePoint]]:
    steering_metric_sweeps = defaultdict(list)

    for filepath in in_dir.glob('*.json'):
        with open(filepath, 'r') as file:
            metrics = json.load(file)
            for metric_data in metrics: # deserialization boundary
                metric_data['pooling_type'] = PoolingTypeEnum(metric_data['pooling_type'])
                metric_data['metric'] = MetricTypeEnum(metric_data['metric'])

                sweep_key = SweepKey(
                    **{key: metric_data[key] for key in SweepKey._fields}
                )
                point = SteeringCurvePoint(
                    **{key: metric_data.get(key) for key in SteeringCurvePoint._fields}
                )
                steering_metric_sweeps[sweep_key].append(point)

    for points in steering_metric_sweeps.values():
        points.sort(key=lambda point: point.alpha)

    return steering_metric_sweeps

def compute_geometric_results(
    estimators: dict[EstimatorKey, torch.Tensor],
) -> dict[tuple[EstimatorKey, EstimatorKey], GeometricMetricsPair]:
    results = {}

    estimators_by_sweep = defaultdict(list)
    for key in estimators.keys():
        estimators_by_sweep[(key.layer, key.pooling_type, key.rank)].append(key)

    for estimator_keys in tqdm(estimators_by_sweep.values(), desc='Computing Geometric Results'):
        for key_a, key_b in itertools.combinations(estimator_keys, 2):
            tensor_a = estimators[key_a].cpu().numpy()
            tensor_b = estimators[key_b].cpu().numpy()

            _, procrustes_alignment = orthogonal_procrustes(tensor_a, tensor_b)

            principal_angles = subspace_angles(tensor_a, tensor_b)
            chordal_distance = np.linalg.norm(np.sin(principal_angles))

            results[(key_a, key_b)] = GeometricMetricsPair(
                procrustes_alignment=float(procrustes_alignment),
                chordal_distance=float(chordal_distance),
            )
    return results


def compute_behavioral_results(
        steering_metric_sweeps: dict[SweepKey, list[SteeringCurvePoint]],
    ) -> BehavioralResults:
    auc_pos_results = {}
    auc_neg_results = {}

    llr_keys = []
    llr_p_values = []
    for sweep_key, curve_points in tqdm(steering_metric_sweeps.items(), desc='Computing Behavioral Results'):
        pos_points = [p for p in curve_points if p.alpha >= 0.0]
        neg_points = [p for p in curve_points if p.alpha <= 0.0]

        auc_pos_results[sweep_key] = float(np.trapz(
            y=[point.value for point in pos_points],
            x=[point.alpha for point in pos_points],
        ))
        auc_neg_results[sweep_key] = -float(np.trapz(
            y=[point.value for point in neg_points],
            x=[point.alpha for point in neg_points],
        ))

        if sweep_key.metric == MetricTypeEnum.LLR_SHIFT:
            for point in curve_points:
                if point.p_value is not None:
                    llr_p_values.append(point.p_value)
                    llr_keys.append((sweep_key, point.alpha))

    llr_p_values_fdr = false_discovery_control(llr_p_values)
    fdr_results = defaultdict(dict)
    for (sweep_key, alpha), p_value_fdr in zip(llr_keys, llr_p_values_fdr):
        fdr_results[sweep_key][alpha] = float(p_value_fdr)

    results = BehavioralResults(
        auc_pos=auc_pos_results,
        auc_neg=auc_neg_results,
        p_values_fdr=dict(fdr_results),
    )
    return results


def flatten_geometric_results(
        geometric_results: dict[tuple[EstimatorKey, EstimatorKey], GeometricMetricsPair],
    ) -> GeometricSummary:
    summary: GeometricSummary = []
    for (key_a, key_b), metrics in geometric_results.items():
        summary.append(GeometricMetricsData(
            layer=key_a.layer,
            pooling_type=key_a.pooling_type,
            rank=key_a.rank,
            estimator_a_name=key_a.estimator_name,
            estimator_b_name=key_b.estimator_name,
            procrustes_alignment=metrics.procrustes_alignment,
            chordal_distances=metrics.chordal_distance,
        ))
    return summary


def flatten_behavioral_results(behavioral_results: BehavioralResults) -> BehavioralSummary:
    auc = []
    for sweep_key, auc_pos in behavioral_results.auc_pos.items():
        auc.append(AucSummary(
            layer=sweep_key.layer,
            pooling_type=sweep_key.pooling_type,
            rank=sweep_key.rank,
            estimator_name=sweep_key.estimator_name,
            corpus_name=sweep_key.corpus_name,
            metric=sweep_key.metric,
            auc_pos=auc_pos,
            auc_neg=behavioral_results.auc_neg[sweep_key],
        ))

    fdr = []
    for sweep_key, alpha_dict in behavioral_results.p_values_fdr.items():
        for alpha, p_value_fdr in alpha_dict.items():
            fdr.append(FdrSummary(
                layer=sweep_key.layer,
                pooling_type=sweep_key.pooling_type,
                rank=sweep_key.rank,
                estimator_name=sweep_key.estimator_name,
                corpus_name=sweep_key.corpus_name,
                metric=sweep_key.metric,
                alpha=alpha,
                p_value_fdr=p_value_fdr,
            ))

    summary = BehavioralSummary(auc=auc, fdr=fdr)
    return summary


def flatten_sweep_metrics(
    steering_metric_sweeps: dict[SweepKey, list[SteeringCurvePoint]],
) -> list[SweepSummary]:
    summary = []
    for sweep_key, points in steering_metric_sweeps.items():
        for point in points:
            summary.append(SweepSummary(
                layer=sweep_key.layer,
                pooling_type=sweep_key.pooling_type,
                rank=sweep_key.rank,
                estimator_name=sweep_key.estimator_name,
                corpus_name=sweep_key.corpus_name,
                metric=sweep_key.metric,
                alpha=point.alpha,
                value=point.value,
                p_value=point.p_value,
            ))
    return summary


def compute_vector_magnitudes(
    estimators: dict[EstimatorKey, torch.Tensor], 
) -> list[VectorMagnitudeSummary]:
    summary = []
    for sweep_key, tensor in estimators.items():
        norm = float(torch.linalg.vector_norm(tensor).item())
        summary.append(VectorMagnitudeSummary(
            layer=sweep_key.layer,
            pooling_type=sweep_key.pooling_type,
            rank=sweep_key.rank,
            estimator_name=sweep_key.estimator_name,
            norm=norm,
        ))
    return summary


if __name__ == '__main__':
    tqdm.write('Loading Data...')
    estimators = load_estimators(ESTIMATORS_DIR, EXPERIMENT_CONFIG)
    steering_metric_sweeps = load_steering_metric_sweeps(SWEEPS_DIR)
    
    geometric_results = compute_geometric_results(estimators)
    behavioral_results = compute_behavioral_results(steering_metric_sweeps)

    geometric_summary = flatten_geometric_results(geometric_results)
    behavioral_summary = flatten_behavioral_results(behavioral_results)
    sweep_summary = flatten_sweep_metrics(steering_metric_sweeps)
    act_add_vector_magnitudes = compute_vector_magnitudes(estimators)
    
    save_json(geometric_summary, METRICS_DIR, GEOMETRIC_RESULTS_FILENAME)
    save_json(behavioral_summary, METRICS_DIR, BEHAVIORAL_RESULTS_FILENAME)
    save_json(sweep_summary, METRICS_DIR, SWEEP_RESULTS_FILENAME)
    save_json(act_add_vector_magnitudes, METRICS_DIR, VECTOR_MAGNITUDES_FILENAME)

    tqdm.write('05: Metric Computation Complete\n')
