from typing import NamedTuple

from scipy.optimize import least_squares
import torch

from src.core.config_utils import Q4RegressionSettings
import src.core.stats_utils as stats_utils
from src.core.utils import (
    CIStats,
    PairControls,
    PairEditData,
    Q4CurveFunction,
    Q4Edit,
    Q4EditKind,
)


EDIT_KINDS: tuple[Q4EditKind, ...] = tuple(Q4EditKind)
CONTENT_CHANGE_IDX: int = EDIT_KINDS.index(Q4EditKind.CONTENT_CHANGE)
NO_EDIT_IDX: int = -1
_MAX_ABS_LOG_WEIGHT: float = 20.0 #


class Inputs(NamedTuple):
    projections: torch.Tensor
    edit_kind_indices: torch.Tensor
    edit_distances: torch.Tensor
    edit_sizes: torch.Tensor
    controls: torch.Tensor
    num_format_controls: int
    uses_distance_decay: bool


class Parameters(NamedTuple):
    weights: torch.Tensor
    distance_decay_exponent: float
    size_exponent: float
    curve_parameters: torch.Tensor
    control_coefficients: torch.Tensor


class Fit(NamedTuple):
    curve: Q4CurveFunction
    settings: Q4RegressionSettings
    parameter_vector: torch.Tensor


class Predictions(NamedTuple):
    projections: torch.Tensor
    curve_values: torch.Tensor
    control_values: torch.Tensor


class Shares(NamedTuple):
    by_kind: torch.Tensor
    format_flags: float
    pair_controls: float
    unexplained: float


class Draws(NamedTuple):
    weights: torch.Tensor
    distance_decay_exponents: torch.Tensor
    size_exponents: torch.Tensor
    kind_shares: torch.Tensor


def _filter_rows(
    inputs: Inputs,
    rows: torch.Tensor,
) -> Inputs:
    subset_rows = inputs._replace(
        projections=inputs.projections[rows],
        edit_kind_indices=inputs.edit_kind_indices[rows],
        edit_distances=inputs.edit_distances[rows],
        edit_sizes=inputs.edit_sizes[rows],
        controls=inputs.controls[rows],
    )
    return subset_rows


def fit_regression(
    inputs: Inputs,
    curve: Q4CurveFunction,
    settings: Q4RegressionSettings,
    initial_fit: Fit | None = None,
    edit_kinds: tuple[Q4EditKind, ...] = EDIT_KINDS,
) -> Fit:
    initial_parameter_vector = (
        initial_fit.parameter_vector
        if initial_fit
        else torch.cat([
            torch.zeros(len(edit_kinds), dtype=torch.float64),
            torch.tensor(
                [
                    settings.initial_distance_decay_exponent,
                    settings.initial_size_exponent
                ],
                dtype=torch.float64,
            ),
            _get_initial_curve_parameters(curve, settings),
            torch.zeros(inputs.controls.shape[1], dtype=torch.float64)
        ])
    )
    compute_residuals_fn = lambda parameter_vector: (
        predict(
            Fit(curve, settings, torch.from_numpy(parameter_vector)), inputs,
        ).projections
        - inputs.projections
    ).numpy()
    solution = least_squares(compute_residuals_fn, initial_parameter_vector.numpy())
    fit = Fit(curve, settings, torch.from_numpy(solution.x))
    return fit


def predict(
    fit: Fit,
    inputs: Inputs,
) -> Predictions:
    parameters = unpack_parameters(fit, inputs.controls.shape[1])
    doses = compute_edit_contributions(parameters, inputs).sum(dim=1)
    curve_values = _apply_curve(doses, parameters.curve_parameters, fit.curve, fit.settings)
    control_values = inputs.controls * parameters.control_coefficients
    predictions = Predictions(
        projections=curve_values + control_values.sum(dim=1),
        curve_values=curve_values,
        control_values=control_values,
    )
    return predictions


def unpack_parameters(
    fit: Fit,
    num_controls: int,
    edit_kinds: tuple[Q4EditKind, ...] = EDIT_KINDS,
    max_abs_log_weight: float = _MAX_ABS_LOG_WEIGHT,
    content_change_idx: int = CONTENT_CHANGE_IDX,
) -> Parameters:
    num_kinds = len(edit_kinds)
    curve_start = num_kinds + 2
    curve_end = curve_start + len(_get_initial_curve_parameters(fit.curve, fit.settings))
    log_weights = torch.clamp(fit.parameter_vector[:num_kinds], -max_abs_log_weight, max_abs_log_weight)

    weights = torch.exp(log_weights - log_weights[content_change_idx])
    distance_decay_exponent = float(fit.parameter_vector[num_kinds])
    size_exponent = float(fit.parameter_vector[num_kinds + 1])
    curve_parameters = fit.parameter_vector[curve_start:curve_end]
    control_coefficients = fit.parameter_vector[curve_end:curve_end + num_controls]
    parameters = Parameters(
        weights=weights,
        distance_decay_exponent=distance_decay_exponent,
        size_exponent=size_exponent,
        curve_parameters=curve_parameters,
        control_coefficients=control_coefficients,
    )
    return parameters


def compute_edit_contributions(
    parameters: Parameters,
    inputs: Inputs,
    no_edit_idx: int = NO_EDIT_IDX,
) -> torch.Tensor:
    distance_decay_exponent = (
        parameters.distance_decay_exponent
        if inputs.uses_distance_decay
        else 0.0
    )
    contributions = (
        parameters.weights[inputs.edit_kind_indices]
        * inputs.edit_distances ** -distance_decay_exponent
        * inputs.edit_sizes ** parameters.size_exponent
    )
    filtered_contributions = torch.where(
        inputs.edit_kind_indices != no_edit_idx,
        contributions,
        0.0,
    )
    return filtered_contributions


def _apply_curve(
    doses: torch.Tensor,
    curve_parameters: torch.Tensor,
    curve: Q4CurveFunction,
    settings: Q4RegressionSettings,
) -> torch.Tensor:
    match curve:
        case Q4CurveFunction.LOG:
            [scale] = curve_parameters
            value = scale * torch.log(doses + 1)
            return value
        case Q4CurveFunction.CEILING:
            top, log_half_effect_dose = curve_parameters
            value = top * doses / (torch.exp(log_half_effect_dose) + doses)
            return value
        case Q4CurveFunction.STRAIGHT:
            slope, intercept = curve_parameters
            value = slope * doses + intercept
            return value
        case Q4CurveFunction.FREE_SHAPE:
            breaks = [
                torch.clamp(doses - breakpoint, min=0.0)
                for breakpoint in settings.free_shape_breakpoints
            ]
            value = torch.stack([doses, *breaks], dim=1) @ curve_parameters
            return value


def _get_initial_curve_parameters(
    curve: Q4CurveFunction,
    settings: Q4RegressionSettings,
) -> torch.Tensor:
    initial_curve_parameters = list(settings.initial_parameters_by_curve[curve])
    if curve == Q4CurveFunction.FREE_SHAPE:
        initial_curve_parameters.extend([0.0] * len(settings.free_shape_breakpoints))
    parameters = torch.tensor(initial_curve_parameters, dtype=torch.float64)
    return parameters


def build_regression_inputs(
    projections: torch.Tensor,
    pair_edits_list: list[list[Q4Edit]],
    pair_edit_data_list: list[PairEditData],
    pair_controls_list: list[PairControls],
    format_flags: tuple[str, ...],
    uses_distance_decay: bool,
    edit_kinds: tuple[Q4EditKind, ...] = EDIT_KINDS,
    no_edit_idx: int = NO_EDIT_IDX,
) -> Inputs:
    num_pairs = len(pair_edits_list)
    max_num_edits = max(1, max(len(pair_edits) for pair_edits in pair_edits_list))
    edit_kind_indices = torch.full((num_pairs, max_num_edits), no_edit_idx)
    edit_distances = torch.ones((num_pairs, max_num_edits), dtype=torch.float64)
    edit_sizes = torch.ones((num_pairs, max_num_edits), dtype=torch.float64)
    for pair_idx, pair_edits in enumerate(pair_edits_list):
        for edit_idx, edit in enumerate(pair_edits):
            edit_kind_indices[pair_idx, edit_idx] = edit_kinds.index(edit['kind'])
            edit_distances[pair_idx, edit_idx] = edit['distance']
            edit_sizes[pair_idx, edit_idx] = edit['size']

    controls = torch.tensor([
        [float(pair_edit_data[flag]) for flag in format_flags]
        + [float(value) for value in pair_controls.values()]
        for pair_edit_data, pair_controls
        in zip(pair_edit_data_list, pair_controls_list, strict=True)
    ], dtype=torch.float64)
    inputs = Inputs(
        projections=projections.double(),
        edit_kind_indices=edit_kind_indices,
        edit_distances=edit_distances,
        edit_sizes=edit_sizes,
        controls=controls,
        num_format_controls=len(format_flags),
        uses_distance_decay=uses_distance_decay,
    )
    return inputs


def compute_shares(
    fit: Fit,
    inputs: Inputs,
    edit_kinds: tuple[Q4EditKind, ...] = EDIT_KINDS,
) -> Shares:
    parameters = unpack_parameters(fit, inputs.controls.shape[1])
    edit_contributions = compute_edit_contributions(parameters, inputs)
    doses = edit_contributions.sum(dim=1)
    
    predictions = predict(fit, inputs)
    curve_value_per_dose = torch.where(
        doses > 0,
        predictions.curve_values / doses,
        0.0,
    )
    edit_shares = edit_contributions * curve_value_per_dose[:, None]
    
    total_projection = inputs.projections.sum()
    format_control_values = predictions.control_values[:, :inputs.num_format_controls]
    pair_control_values = predictions.control_values[:, inputs.num_format_controls:]
    
    shares_by_kind = torch.tensor([
        edit_shares[inputs.edit_kind_indices == kind_idx].sum()
        for kind_idx in range(len(edit_kinds))
    ], dtype=torch.float64) / total_projection
    format_flags = float(format_control_values.sum() / total_projection)
    pair_controls = float(pair_control_values.sum() / total_projection)
    unexplained = float(
        (inputs.projections - predictions.projections).sum() / total_projection
    )
    shares = Shares(
        by_kind=shares_by_kind,
        format_flags=format_flags,
        pair_controls=pair_controls,
        unexplained=unexplained,
    )
    return shares


def compute_mean_marginal_effects(
    fit: Fit,
    inputs: Inputs,
    distance: int,
) -> torch.Tensor:
    parameters = unpack_parameters(fit, inputs.controls.shape[1])
    apply_curve_fn = lambda pair_doses: _apply_curve(pair_doses, parameters.curve_parameters, fit.curve, fit.settings)
    distance_decay_exponent = (
        parameters.distance_decay_exponent
        if inputs.uses_distance_decay
        else 0.0
    )
    doses = compute_edit_contributions(parameters, inputs).sum(dim=1)
    added_doses = parameters.weights * distance ** -distance_decay_exponent
    mean_marginal_effects = torch.stack([
        (apply_curve_fn(doses + added_dose) - apply_curve_fn(doses)).mean()
        for added_dose in added_doses
    ])
    return mean_marginal_effects


def compute_held_out_predictions(
    fit: Fit,
    inputs: Inputs,
    num_folds: int,
    seed: int,
) -> torch.Tensor:
    fold_ids = stats_utils.make_fold_ids(len(inputs.projections), num_folds, seed)
    held_out_predictions = torch.empty_like(inputs.projections)
    for fold_id in range(num_folds):
        is_held_out = fold_ids == fold_id
        fold_fit = fit_regression(
            _filter_rows(inputs, ~is_held_out),
            fit.curve,
            fit.settings,
            initial_fit=fit,
        )
        held_out_predictions[is_held_out] = predict(
            fold_fit, _filter_rows(inputs, is_held_out),
        ).projections
    return held_out_predictions


def compute_mean_residual_by_kind(
    predictions: torch.Tensor,
    inputs: Inputs,
    edit_kinds: tuple[Q4EditKind, ...] = EDIT_KINDS,
) -> torch.Tensor:
    residuals = inputs.projections - predictions
    mean_residuals = torch.stack([
        residuals[(inputs.edit_kind_indices == kind_idx).any(dim=1)].mean()
        for kind_idx in range(len(edit_kinds))
    ])
    return mean_residuals


def compute_shuffled_r_squared(
    fit: Fit,
    inputs: Inputs,
    seed: int,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    shuffled_projections = inputs.projections[torch.randperm(len(inputs.projections), generator=generator)]
    shuffled_inputs = inputs._replace(projections=shuffled_projections)
    shuffled_fit = fit_regression(shuffled_inputs, fit.curve, fit.settings)
    shuffled_predictions = predict(shuffled_fit, shuffled_inputs).projections
    shuffled_r_squared = float(stats_utils.compute_r_squared(shuffled_predictions, shuffled_projections))
    return shuffled_r_squared


def bootstrap_regression(
    fit: Fit,
    inputs: Inputs,
    num_draws: int,
    seed: int,
) -> Draws:
    generator = torch.Generator().manual_seed(seed)
    num_pairs = len(inputs.projections)
 
    weights_list, distance_decay_exponents_list, size_exponent_list, kind_shares_list = [], [], [], []
    for _ in range(num_draws):
        rows = torch.randint(num_pairs, (num_pairs,), generator=generator)
        resampled_inputs = _filter_rows(inputs, rows)
        draw_fit = fit_regression(resampled_inputs, fit.curve, fit.settings, initial_fit=fit)
        draw_parameters = unpack_parameters(draw_fit, inputs.controls.shape[1])
        weights_list.append(draw_parameters.weights)
        distance_decay_exponents_list.append(draw_parameters.distance_decay_exponent)
        size_exponent_list.append(draw_parameters.size_exponent)
        kind_shares_list.append(compute_shares(draw_fit, resampled_inputs).by_kind)
 
    weights=torch.stack(weights_list)
    distance_decay_exponents=torch.tensor(distance_decay_exponents_list, dtype=torch.float64)
    size_exponents=torch.tensor(size_exponent_list, dtype=torch.float64)
    kind_shares=torch.stack(kind_shares_list)
    draws = Draws(
        weights=weights,
        distance_decay_exponents=distance_decay_exponents,
        size_exponents=size_exponents,
        kind_shares=kind_shares,
    )    
    return draws


def compute_single_edit_weight_cis(
    inputs: Inputs,
    num_draws: int,
    p_alpha: float,
    seed: int,
    edit_kinds: tuple[Q4EditKind, ...] = EDIT_KINDS,
    no_edit_idx: int = NO_EDIT_IDX,
    content_change_idx: int = CONTENT_CHANGE_IDX,
) -> dict[Q4EditKind, CIStats]:
    def _compute_weights(
        rows: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = stats_utils.compute_regression_coefficients(feature_matrix[rows], projections[rows])
        kind_coefficients = coefficients[:len(present_kind_indices)]
        weights = kind_coefficients / kind_coefficients[content_change_position]
        return weights

    num_edits = (inputs.edit_kind_indices != no_edit_idx).sum(dim=1)
    is_single_edit = num_edits == 1
    kind_indices = inputs.edit_kind_indices[is_single_edit, 0]
    projections = inputs.projections[is_single_edit]
    log_distances = torch.log(inputs.edit_distances[is_single_edit, 0])
    sizes = inputs.edit_sizes[is_single_edit, 0]
    present_kind_indices = kind_indices.unique().tolist()
    feature_matrix = torch.column_stack([
        (kind_indices[:, None] == torch.tensor(present_kind_indices)).double(),
        log_distances - log_distances.mean(),
        sizes - sizes.mean(),
    ])
    content_change_position = present_kind_indices.index(content_change_idx)

    num_pairs = len(projections)
    estimates = _compute_weights(torch.arange(num_pairs))
    generator = torch.Generator().manual_seed(seed)
    draws = torch.stack([
        _compute_weights(torch.randint(num_pairs, (num_pairs,), generator=generator))
        for _ in range(num_draws)
    ])
    single_edit_weight_cis = {
        edit_kinds[kind_idx]: stats_utils.compute_percentile_ci(
            draws[:, position], p_alpha, float(estimates[position]),
        )
        for position, kind_idx in enumerate(present_kind_indices)
    }
    return single_edit_weight_cis
