from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm
from scipy.stats import false_discovery_control

from src.core.config import EXPERIMENT_CONFIG
from src.core.paths import (
    SWEEP_RESULTS_FILENAME,
    BEHAVIORAL_RESULTS_FILENAME,
    RunDirs,
    get_run_dirs,
)
from src.core.utils import (
    AucSummary,
    BehavioralSummary,
    FdrSummary,
    MetricTypeEnum,
    PoolingTypeEnum,
    SweepKey,
    SweepSummary,
    load_json,
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


### Steering Metrics ###
def _load_steering_metric_sweeps(
    in_dir: Path
) -> dict[SweepKey, list[SteeringCurvePoint]]:
    steering_metric_sweeps = defaultdict(list)
    for filepath in in_dir.glob('*.json'):
        metrics = load_json(filepath)
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


def _flatten_sweep_metrics(
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


def _compute_behavioral_results(
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

    fdr_results = defaultdict(dict)
    if llr_p_values:
        llr_p_values_fdr = false_discovery_control(llr_p_values)
        for (sweep_key, alpha), p_value_fdr in zip(llr_keys, llr_p_values_fdr):
            fdr_results[sweep_key][alpha] = float(p_value_fdr)

    results = BehavioralResults(
        auc_pos=auc_pos_results,
        auc_neg=auc_neg_results,
        p_values_fdr=dict(fdr_results),
    )
    return results


def _flatten_behavioral_results(behavioral_results: BehavioralResults) -> BehavioralSummary:
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

### Main ###
def compute_and_save_metrics(
    run_dirs: RunDirs,
    sweep_results_filename: str = SWEEP_RESULTS_FILENAME,
    behavioral_results_filename: str = BEHAVIORAL_RESULTS_FILENAME,
) -> None:
    steering_metric_sweeps = _load_steering_metric_sweeps(run_dirs.sweeps)
    behavioral_results = _compute_behavioral_results(steering_metric_sweeps)
    save_json(_flatten_sweep_metrics(steering_metric_sweeps), run_dirs.steering_results, sweep_results_filename)
    save_json(_flatten_behavioral_results(behavioral_results), run_dirs.steering_results, behavioral_results_filename)


if __name__ == '__main__':
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    compute_and_save_metrics(run_dirs)
    tqdm.write('08: Steering Metrics Computation Complete\n')
