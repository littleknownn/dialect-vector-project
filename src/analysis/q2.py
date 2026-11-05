import itertools
from pathlib import Path
from typing import NamedTuple

from scipy.stats import false_discovery_control
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.core.base_model import load_tokenizer
from src.core.config_utils import (
    CoraalSourceConfig,
    DatasetConfig,
    ExperimentConfig,
    SupervisedDatasetConfig,
    get_estimator_data,
)
from src.core.paths import (
    PROCESSED_DATA_DIR,
    Q2_RESULTS_FILENAME,
    RunDirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
)
import src.core.matched_auroc as matched_auroc
import src.core.stats_utils as stats_utils
import src.core.text_rules as text_rules
import src.core.text_utils as text_utils
from src.core.utils import (
    AaeFeature,
    CIStats,
    DifferenceVector,
    LineAnnotations,
    PoolingType,
    Q2Family,
    Q2Results,
    Q2TransferLabel,
    SeedOffset,
    SplitSuffixes,
    load_json,
    load_jsonl,
    save_json,
)


_ALL_PROBES: tuple[AaeFeature, ...] = tuple(AaeFeature)
_SECONDARY_PROBE_SETS: tuple[tuple[AaeFeature, ...], ...] = (
    tuple(
        probe for probe in AaeFeature
        if probe != AaeFeature.UNCONJUGATED_AINT
    ),
    (AaeFeature.NEGATIVE_CONCORD,),
    (AaeFeature.UNCONJUGATED_AINT,),
)


class Q2LineData(NamedTuple):
    texts: list[str]
    annotations_list: list[LineAnnotations]
    cell_ids: torch.Tensor
    cluster_ids: torch.Tensor
    surface_features: dict[str, torch.Tensor]


class Q2TestSpec(NamedTuple):
    pooling_type: PoolingType
    family: Q2Family
    layer: int
    vector: DifferenceVector = DifferenceVector.DIALECT
    positive_probes: tuple[AaeFeature, ...] = _ALL_PROBES
    excludes_filter_flagged_words: bool = False


class Q2Masks(NamedTuple):
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    num_probes_after_cut_lines: int


class Q2Scores(NamedTuple):
    scores: torch.Tensor
    midpoint_char_indices: list[int | None] | None


def _compute_uppercase_share(
    text: str,
) -> float:
    letters = [character for character in text if character.isalpha()]
    uppercase_share = sum(
        character.isupper() for character in letters
    ) / max(len(letters), 1)
    return uppercase_share


def _is_filter_flagged(
    annotations: LineAnnotations,
) -> bool:
    is_filter_flagged = bool(annotations['profanity_matches']) or annotations['has_masked_word']
    return is_filter_flagged


def _get_annotation_tensor(
    annotations_list: list[LineAnnotations],
    key: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    annotation_tensor = torch.tensor(
        [annotations[key] for annotations in annotations_list],
        dtype=dtype,
    )
    return annotation_tensor


def _get_q2_cell_ids(
    annotations_list: list[LineAnnotations],
) -> torch.Tensor:
    cell_keys = torch.stack(
        [
            _get_annotation_tensor(annotations_list, 'num_words'),
            _get_annotation_tensor(annotations_list, 'has_terminal_punctuation', torch.long),
            _get_annotation_tensor(annotations_list, 'has_mention', torch.long),
        ],
        dim=1,
    )
    _, cell_ids = cell_keys.unique(dim=0, return_inverse=True)
    return cell_ids


def _get_q2_split_name(
    corpus_name: str,
    corpus_config: dict[str, DatasetConfig],
) -> str:
    config = corpus_config[corpus_name]
    if isinstance(config, SupervisedDatasetConfig):
        return f'{corpus_name}{SplitSuffixes.AAE}{SplitSuffixes.TESTING}'
    if config.pool_size is not None:
        return f'{corpus_name}{SplitSuffixes.POOL}'
    return corpus_name


def _get_speaker_split_name(
    corpus_name: str,
    corpus_config: dict[str, DatasetConfig],
) -> str | None:
    config = corpus_config[corpus_name]
    if not isinstance(config.source_config, CoraalSourceConfig):
        return None
    pool_suffix = (
        SplitSuffixes.POOL
        if config.pool_size is not None
        else ''
    )
    speaker_split_name = f'{corpus_name}{SplitSuffixes.SPEAKERS}{pool_suffix}'
    return speaker_split_name


def _load_q2_line_data(
    corpus_name: str,
    split_name: str,
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path,
) -> Q2LineData:
    texts = load_jsonl(processed_data_dir / f'{split_name}.jsonl')[:experiment_config.debug_max_split_lines]
    annotations_list = load_json(run_dirs.text_annotations / f'{split_name}.json')[:experiment_config.debug_max_split_lines]
    speaker_split_name = _get_speaker_split_name(corpus_name, corpus_config)
    if speaker_split_name is None:
        cluster_ids = torch.arange(len(texts))
    else:
        speakers = load_json(processed_data_dir / f'{speaker_split_name}.json')[:experiment_config.debug_max_split_lines]
        speaker_ids = {
            speaker: speaker_idx
            for speaker_idx, speaker
            in enumerate(dict.fromkeys(speakers))
        }
        cluster_ids = torch.tensor([speaker_ids[speaker] for speaker in speakers])
    surface_features = {
        'uppercase_share': torch.tensor(
            [_compute_uppercase_share(text) for text in texts],
            dtype=torch.float64,
        ),
        'profanity': torch.tensor(
            [_is_filter_flagged(annotations) for annotations in annotations_list],
            dtype=torch.float64,
        ),
    }
    line_data = Q2LineData(
        texts=texts,
        annotations_list=annotations_list,
        cell_ids=_get_q2_cell_ids(annotations_list),
        cluster_ids=cluster_ids,
        surface_features=surface_features,
    )
    return line_data


def _get_q2_test_specs(
    corpus_name: str,
    experiment_config: ExperimentConfig,
    secondary_probe_sets: tuple[tuple[AaeFeature, ...], ...] = _SECONDARY_PROBE_SETS,
) -> list[Q2TestSpec]:
    if corpus_name not in experiment_config.q2.primary_corpus_names:
        test_specs = [
            Q2TestSpec(pooling_type, Q2Family.DESCRIPTIVE, experiment_config.primary_layer)
            for pooling_type in experiment_config.q2.primary_pooling_types
        ]
        return test_specs

    test_specs = [
        Q2TestSpec(pooling_type, Q2Family.PRIMARY, experiment_config.primary_layer)
        for pooling_type in experiment_config.q2.primary_pooling_types
    ]
    test_specs.extend([
       Q2TestSpec(pooling_type, Q2Family.SECONDARY, experiment_config.primary_layer, positive_probes=probes)
       for pooling_type, probes in itertools.product(
           experiment_config.q2.primary_pooling_types,
           secondary_probe_sets,
       )
    ])
    test_specs.extend([
        Q2TestSpec(
            pooling_type,
            Q2Family.SECONDARY,
            experiment_config.primary_layer,
            excludes_filter_flagged_words=True,
        )
        for pooling_type in experiment_config.q2.primary_pooling_types
    ])
    if corpus_name in experiment_config.q2.midpoint_corpus_names:
        test_specs.append(Q2TestSpec(
            PoolingType.BOUNDARY, Q2Family.SECONDARY, experiment_config.primary_layer,
        ))
    test_specs.extend([
        Q2TestSpec(pooling_type, Q2Family.DESCRIPTIVE, experiment_config.primary_layer, vector=vector)
        for pooling_type, vector in itertools.product(
            experiment_config.q2.primary_pooling_types,
            experiment_config.q2.sensitivity_vectors,
        )
    ])
    test_specs.extend([
        Q2TestSpec(pooling_type, Q2Family.DESCRIPTIVE, layer)
        for pooling_type, layer in itertools.product(
            experiment_config.q2.primary_pooling_types,
            experiment_config.target_layers,
        )
        if layer != experiment_config.primary_layer
    ])
    return test_specs


def _load_unit_vector(
    pooling_type: PoolingType,
    layer: int,
    vector: DifferenceVector,
    paired_corpus_name: str,
    run_dirs: RunDirs,
) -> torch.Tensor:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    difference_vector = torch.load(
        get_sweep_tensor_filepath(
            run_dirs.difference_vectors, sweep_identifier, f'{paired_corpus_name}{vector}',
        ),
        weights_only=True,
    )
    unit_vector = F.normalize(difference_vector.float().squeeze(1), dim=0)
    return unit_vector


def _compute_midpoint_scores(
    split_name: str,
    texts: list[str],
    unit_vector: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    pair_config: SupervisedDatasetConfig,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> Q2Scores:
    saved_token_stats = torch.load(run_dirs.token_states / f'{split_name}.pt', weights_only=True)
    line_start_indices = torch.cumsum(saved_token_stats['num_tokens_by_line'], dim=0) - saved_token_stats['num_tokens_by_line']

    midpoint_char_indices = [
        text_utils.compute_line_midpoint_char_idx(text, pair_config)
        for text in texts
    ]
    read_line_indices, read_state_indices = [], []
    for line_idx, (text, midpoint_char_idx) in enumerate(zip(texts, midpoint_char_indices)):
        if midpoint_char_idx is None:
            continue
        token_char_indices = tokenizer(
            text, truncation=True, max_length=experiment_config.max_seq_len, return_offsets_mapping=True, return_tensors='pt',
        )['offset_mapping'][0]
        token_idx = text_utils.get_boundary_token_idx(token_char_indices, midpoint_char_idx)
        if token_idx >= saved_token_stats['num_tokens_by_line'][line_idx]:
            continue
        read_line_indices.append(line_idx)
        read_state_indices.append(int(line_start_indices[line_idx]) + token_idx)

    scores = torch.full((len(texts),), float('nan'))
    read_states = saved_token_stats['token_states'][read_state_indices].float()
    scores[read_line_indices] = read_states @ unit_vector
    q2_scores = Q2Scores(scores=scores, midpoint_char_indices=midpoint_char_indices)
    return q2_scores


def _compute_q2_scores(
    split_name: str,
    test_spec: Q2TestSpec,
    line_data: Q2LineData,
    tokenizer: PreTrainedTokenizerBase | None,
    paired_corpus_name: str,
    pair_config: SupervisedDatasetConfig,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> Q2Scores:
    unit_vector = _load_unit_vector(
        test_spec.pooling_type, test_spec.layer, test_spec.vector, paired_corpus_name, run_dirs,
    )
    if test_spec.pooling_type == PoolingType.BOUNDARY:
        return _compute_midpoint_scores(split_name, line_data.texts, unit_vector, tokenizer, pair_config, experiment_config, run_dirs)

    sweep_identifier = get_sweep_identifier(test_spec.layer, test_spec.pooling_type)
    activations = torch.load(
        get_sweep_tensor_filepath(run_dirs.activations, sweep_identifier, split_name),
        weights_only=True,
    )
    q2_scores = Q2Scores(
        scores=activations.float() @ unit_vector,
        midpoint_char_indices=None,
    )
    return q2_scores


def _has_probe_before_cut(
    text: str,
    probes: tuple[AaeFeature, ...],
    cut_char_idx: int | None,
) -> bool:
    if cut_char_idx is None:
        return False
    has_probe_before_cut = any(
        match.end() <= cut_char_idx
        for probe in probes
        for match in text_rules.AAE_FEATURE_PATTERNS[probe].finditer(text)
    )
    return has_probe_before_cut


def _get_q2_masks(
    line_data: Q2LineData,
    q2_scores: Q2Scores,
    test_spec: Q2TestSpec,
) -> Q2Masks:
    usable_mask = (
        ~q2_scores.scores.isnan()
        & ~_get_annotation_tensor(line_data.annotations_list, 'is_truncated')
        & torch.tensor([
            not (
                test_spec.excludes_filter_flagged_words
                and _is_filter_flagged(annotations)
            )
            for annotations in line_data.annotations_list
        ])
    )
    definition_probe_mask = torch.tensor([
        any(
            probe in test_spec.positive_probes
            for probe in annotations['probes']
        )
        for annotations in line_data.annotations_list
    ])
    no_probe_mask = torch.tensor([
        not annotations['probes']
        for annotations in line_data.annotations_list
    ])
    positive_mask = usable_mask & definition_probe_mask
    num_probes_after_cut_lines = 0
    if q2_scores.midpoint_char_indices is not None:
        probe_before_cut_mask = torch.tensor([
            _has_probe_before_cut(text, test_spec.positive_probes, midpoint_char_idx)
            for text, midpoint_char_idx in zip(line_data.texts, q2_scores.midpoint_char_indices)
        ])
        num_probes_after_cut_lines = int((positive_mask & ~probe_before_cut_mask).sum())
        positive_mask &= probe_before_cut_mask
    q2_masks = Q2Masks(
        positive_mask=positive_mask,
        negative_mask=usable_mask & no_probe_mask,
        num_probes_after_cut_lines=num_probes_after_cut_lines,
    )
    return q2_masks


def _get_q2_label(
    auroc_ci: CIStats,
    null_p_value_fdr: float | None,
    experiment_config: ExperimentConfig,
    chance_auroc: float = 0.5,
) -> Q2TransferLabel:
    is_significant = (
        null_p_value_fdr < experiment_config.p_alpha
        if null_p_value_fdr is not None
        else auroc_ci['ci_lower'] > chance_auroc
    )
    if (
        is_significant
        and auroc_ci['ci_lower'] > chance_auroc
        and auroc_ci['estimate'] >= experiment_config.q2.min_auroc
    ):
        return Q2TransferLabel.TRANSFER
    if is_significant and auroc_ci['estimate'] < experiment_config.q2.min_auroc:
        return Q2TransferLabel.SMALL
    if not is_significant and auroc_ci['ci_upper'] < experiment_config.q2.min_auroc:
        return Q2TransferLabel.BOUNDED_NULL
    return Q2TransferLabel.INCONCLUSIVE


def _compute_q2_results(
    corpus_name: str,
    test_spec: Q2TestSpec,
    q2_scores: Q2Scores,
    line_data: Q2LineData,
    matched_sets: matched_auroc.MatchedSets,
    q2_masks: Q2Masks,
    experiment_config: ExperimentConfig,
) -> Q2Results:
    scores = q2_scores.scores.double()
    usable_scores = scores[q2_masks.positive_mask | q2_masks.negative_mask]
    cluster_ids = line_data.cluster_ids[matched_sets.positive_indices]
    bootstrap_ci_fn = lambda per_positive_values: stats_utils.compute_percentile_ci(
        draws=stats_utils.compute_cluster_bootstrap_means(
            per_positive_values,
            cluster_ids,
            experiment_config.num_bootstrap_samples,
            experiment_config.random_seed + SeedOffset.Q2_BOOTSTRAP,
        ),
        p_alpha=experiment_config.p_alpha,
        estimate=float(per_positive_values.mean()),
    )
    auroc_ci = bootstrap_ci_fn(matched_auroc.compute_per_positive_aurocs(scores, matched_sets))
    standardized_mean_difference_ci = bootstrap_ci_fn(
        matched_auroc.compute_per_positive_differences(scores, matched_sets) / usable_scores.std()
    )

    null_p_value, null_band = None, None
    if test_spec.family != Q2Family.DESCRIPTIVE:
        null_aurocs = matched_auroc.compute_label_shuffle_null(
            scores,
            q2_masks.positive_mask,
            q2_masks.negative_mask,
            line_data.cell_ids,
            experiment_config.q2.num_negatives_per_positive,
            experiment_config.q2.num_null_shuffles,
            experiment_config.random_seed + SeedOffset.Q2_LABEL_SHUFFLE,
        )
        null_p_value = stats_utils.compute_one_sided_p_value(null_aurocs, auroc_ci['estimate'])
        null_ci = stats_utils.compute_percentile_ci(null_aurocs, experiment_config.p_alpha)
        null_band = [null_ci['ci_lower'], null_ci['ci_upper']]

    is_valid_negative = matched_sets.negative_indices != matched_auroc.NO_NEGATIVE_IDX
    num_negatives_by_positive = is_valid_negative.sum(dim=1)
    num_unique_negatives = len(matched_sets.negative_indices[is_valid_negative].unique())
    baseline_aurocs = {
        feature_name: float(matched_auroc.compute_per_positive_aurocs(feature_values, matched_sets).mean())
        for feature_name, feature_values in line_data.surface_features.items()
    }
    num_dropped_positives = int(q2_masks.positive_mask.sum()) - len(matched_sets.positive_indices)
    num_short_positives = int(
        (num_negatives_by_positive < experiment_config.q2.num_negatives_per_positive).sum()
    )
    results = Q2Results(
        corpus_name=corpus_name,
        layer=test_spec.layer,
        pooling_type=test_spec.pooling_type,
        vector=test_spec.vector,
        positive_probes=list(test_spec.positive_probes),
        excludes_filter_flagged_words=test_spec.excludes_filter_flagged_words,
        family=test_spec.family,
        num_positives=len(matched_sets.positive_indices),
        num_dropped_positives=num_dropped_positives,
        num_short_positives=num_short_positives,
        num_probes_after_cut_lines=q2_masks.num_probes_after_cut_lines,
        mean_num_negatives=float(num_negatives_by_positive.double().mean()),
        num_unique_negatives=num_unique_negatives,
        auroc_ci=auroc_ci,
        standardized_mean_difference_ci=standardized_mean_difference_ci,
        null_p_value=null_p_value,
        null_p_value_fdr=None,
        null_band=null_band,
        baseline_aurocs=baseline_aurocs,
        min_detectable_auroc=matched_auroc.compute_minimum_detectable_auroc(
            len(matched_sets.positive_indices),
            num_unique_negatives,
            experiment_config.p_alpha,
            experiment_config.q2.power,
        ),
        label=None,
    )
    return results


def _apply_q2_corrections_and_labels(
    q2_results: list[Q2Results],
    experiment_config: ExperimentConfig,
) -> None:
    for family in (Q2Family.PRIMARY, Q2Family.SECONDARY):
        family_results = [
            results
            for results in q2_results
            if results['family'] == family
        ]
        if not family_results:
            continue
        p_values_fdr = false_discovery_control([results['null_p_value'] for results in family_results])
        for results, p_value in zip(family_results, p_values_fdr):
            results['null_p_value_fdr'] = float(p_value)
    for results in q2_results:
        results['label'] = _get_q2_label(results['auroc_ci'], results['null_p_value_fdr'], experiment_config)


def compute_and_save_q2_results(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    q2_results_filename: str = Q2_RESULTS_FILENAME,
) -> None:
    paired_corpus_name = get_estimator_data(corpus_config).supervised_split_pairs[0].corpus_name
    tokenizer = (
        load_tokenizer(experiment_config)
        if experiment_config.q2.midpoint_corpus_names
        else None
    )
    matching_seed = experiment_config.random_seed + SeedOffset.Q2_MATCHING

    q2_results = []
    for corpus_name in tqdm(experiment_config.q2.primary_corpus_names + experiment_config.q2.reported_corpus_names, desc='Q2'):
        split_name = _get_q2_split_name(corpus_name, corpus_config)
        line_data = _load_q2_line_data(corpus_name, split_name, corpus_config, experiment_config, run_dirs, processed_data_dir)
        scores_by_read = {}
        for test_spec in _get_q2_test_specs(corpus_name, experiment_config):
            read_key = (test_spec.pooling_type, test_spec.layer, test_spec.vector)
            if read_key not in scores_by_read:
                scores_by_read[read_key] = _compute_q2_scores(
                    split_name, test_spec, line_data, tokenizer, paired_corpus_name, corpus_config[paired_corpus_name], experiment_config, run_dirs,
                )
            q2_masks = _get_q2_masks(line_data, scores_by_read[read_key], test_spec)
            matched_sets = matched_auroc.draw_matched_sets(
                q2_masks.positive_mask, q2_masks.negative_mask, line_data.cell_ids, experiment_config.q2.num_negatives_per_positive, matching_seed,
            )
            q2_results.append(_compute_q2_results(
                corpus_name, test_spec, scores_by_read[read_key], line_data, matched_sets, q2_masks, experiment_config,
            ))

    _apply_q2_corrections_and_labels(q2_results, experiment_config)
    save_json(q2_results, run_dirs.transfer_results, q2_results_filename)
