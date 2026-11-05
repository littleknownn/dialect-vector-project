from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from src.analysis.pair_data import get_pooling_annotations_list
from src.analysis.q1 import Q1Projections
from src.analysis.q4 import (
    build_q4_regression_inputs,
    compute_q4_parameter_cis,
    get_q4_parameter_estimates,
)
from src.core.config_utils import (
    ExperimentConfig,
    SupervisedPair,
)
from src.core.paths import (
    Q4_ROBUSTNESS_PREFIX,
    get_sweep_identifier,
)
import src.core.edit_regression as edit_regression
from src.core.utils import (
    DifferenceVector,
    PairAnnotations,
    Q4CurveFunction,
    Q4EditKind,
    Q4RobustnessResults,
    Q4RobustnessVariant,
    SplitSuffixes,
    TextPairAnnotations,
    save_json,
)


_SINGLE_WORD_EDIT_SIZE: int = 1
_VECTOR_BY_ROBUSTNESS_VARIANT: dict[Q4RobustnessVariant, DifferenceVector] = {
    Q4RobustnessVariant.NO_PROFANITY_VECTOR: DifferenceVector.NO_PROFANITY,
    Q4RobustnessVariant.NO_TYPOGRAPHY_VECTOR: DifferenceVector.NO_TYPOGRAPHY,
    Q4RobustnessVariant.SAME_TERMINAL_PUNCTUATION_VECTOR: DifferenceVector.SAME_TERMINAL_PUNCTUATION,
}


class Q4RobustnessInputs(NamedTuple):
    variant: Q4RobustnessVariant
    removed_word: str | None
    regression_inputs: edit_regression.Inputs


def _build_q4_robustness_inputs_list(
    q1_projections: Q1Projections,
    pooling_annotations_list: list[TextPairAnnotations],
    experiment_config: ExperimentConfig,
    single_word_edit_size: int = _SINGLE_WORD_EDIT_SIZE,
    vector_by_robustness_variant: dict[Q4RobustnessVariant, DifferenceVector] = _VECTOR_BY_ROBUSTNESS_VARIANT,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> list[Q4RobustnessInputs]:
    build_fn = lambda projections, pair_edits_list: build_q4_regression_inputs(
        projections, pooling_annotations_list, pair_edits_list, experiment_config.primary_pooling_type, experiment_config,
    )
    relabeled_build_fn = lambda kind_fn: build_fn(
        q1_projections.primary_pooling_projections_by_vector[DifferenceVector.DIALECT],
        [
            [
                edit | {'kind': kind_fn(edit)}
                for edit in annotations['edits']
            ]
            for annotations in pooling_annotations_list
        ],
    )

    pair_edits_list = [annotations['edits'] for annotations in pooling_annotations_list]
    primary_inputs = build_fn(
        q1_projections.primary_pooling_projections_by_vector[DifferenceVector.DIALECT],
        pair_edits_list,
    )
    multi_word_marker_mask = (
        (primary_inputs.edit_kind_indices == edit_kinds.index(Q4EditKind.GRAMMAR_WORD))
        & (primary_inputs.edit_sizes > single_word_edit_size)
    )
    robustness_inputs_list = [
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.SINGLE_WORD_MARKERS,
            removed_word=None,
            regression_inputs=edit_regression.relabel_edit_kinds(
                primary_inputs, multi_word_marker_mask, Q4EditKind.CONTENT_CHANGE,
            ),
        ),
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.AINT_EXCLUDED,
            removed_word=None,
            regression_inputs=relabeled_build_fn(
                lambda edit: edit['kind_without_aint']
            ),
        ),
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.DISTANCE_FROM_TWO,
            removed_word=None,
            regression_inputs=primary_inputs._replace(
                edit_distances=primary_inputs.edit_distances + experiment_config.q4.robustness_distance_shift,
            ),
        ),
    ]
    robustness_inputs_list += [
        Q4RobustnessInputs(variant, removed_word=None, regression_inputs=build_fn(
            q1_projections.primary_pooling_projections_by_vector[vector],
            pair_edits_list,
        ))
        for variant, vector in vector_by_robustness_variant.items()
    ]
    present_marker_words = sorted({
        word
        for pair_edits in pair_edits_list
        for edit in pair_edits
        for word in edit['kind_by_removed_marker_word']
    })
    robustness_inputs_list += [
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.LEAVE_ONE_OUT,
            removed_word=word,
            regression_inputs=relabeled_build_fn(
                lambda edit:
                edit['kind_by_removed_marker_word'].get(word, edit['kind']),
            )
        )
        for word in present_marker_words
    ]
    return robustness_inputs_list


def _compute_q4_robustness_results(
    pair: SupervisedPair,
    robustness_inputs: Q4RobustnessInputs,
    curve: Q4CurveFunction,
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4RobustnessResults:
    fit = edit_regression.fit_regression(
        robustness_inputs.regression_inputs, curve, experiment_config.q4.regression_settings,
    )
    parameter_cis = (
        get_q4_parameter_estimates(fit, robustness_inputs.regression_inputs)
        if robustness_inputs.variant == Q4RobustnessVariant.LEAVE_ONE_OUT
        else compute_q4_parameter_cis(
            fit, robustness_inputs.regression_inputs, experiment_config.q4.num_robustness_bootstrap_samples, experiment_config,
        )
    )
    num_edits_by_kind = edit_regression.count_edits_by_kind(robustness_inputs.regression_inputs)
    num_marker_edits = int(num_edits_by_kind[edit_kinds.index(Q4EditKind.GRAMMAR_WORD)])
    results = Q4RobustnessResults(
        corpus_name=pair.corpus_name,
        pooling_type=experiment_config.primary_pooling_type,
        variant=robustness_inputs.variant,
        removed_word=robustness_inputs.removed_word,
        curve=curve,
        num_marker_edits=num_marker_edits,
        weights=parameter_cis.weights,
    )
    return results


def compute_and_save_q4_robustness_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    q1_projections: Q1Projections,
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    pooling_annotations_list = get_pooling_annotations_list(pair_annotations_list, experiment_config.primary_pooling_type)
    robustness_inputs_list = _build_q4_robustness_inputs_list(q1_projections, pooling_annotations_list, experiment_config)
    robustness_results = [
        _compute_q4_robustness_results(pair, robustness_inputs, curve, experiment_config)
        for robustness_inputs in tqdm(robustness_inputs_list, desc='Q4 Robustness', leave=False)
        for curve in experiment_config.q4.primary_curves
    ]
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, experiment_config.primary_pooling_type)
    save_json(robustness_results, out_dir, f'{Q4_ROBUSTNESS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}')

