from pathlib import Path

import torch

from src.core.config_utils import (
    ExperimentConfig,
    SupervisedPair,
)
from src.core.paths import (
    PAIR_EDIT_DATA_PREFIX,
    RunDirs,
)
from src.core.utils import (
    PairAnnotations,
    PairEditData,
    PoolingType,
    Q4Stratum,
    SplitSuffixes,
    TextPairAnnotations,
    load_json,
    save_json,
)


def load_pair_annotations_list(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> list[PairAnnotations]:
    pair_annotations_list = load_json(
        run_dirs.text_annotations / f'{pair.corpus_name}{SplitSuffixes.PAIR_ANNOTATIONS}{split_suffix}.json',
    )[:experiment_config.debug_max_split_lines]
    return pair_annotations_list


def get_pooling_annotations_list(
    pair_annotations_list: list[PairAnnotations],
    pooling_type: PoolingType,
) -> list[TextPairAnnotations]:
    pooling_annotations_list = [
        pair_annotations['context']
        if pooling_type == PoolingType.BOUNDARY
        and pair_annotations['context'] is not None
        else pair_annotations['line']
        for pair_annotations in pair_annotations_list
    ]
    return pooling_annotations_list


def get_identical_pairs_mask(
    pair_annotations_list: list[PairAnnotations],
    pooling_type: PoolingType,
) -> torch.Tensor:
    identical_pairs_mask = torch.tensor([
        annotations['edit_data_by_threshold']['strict']['stratum'] == Q4Stratum.IDENTICAL
        for annotations in get_pooling_annotations_list(pair_annotations_list, pooling_type)
    ])
    return identical_pairs_mask


def save_pair_edit_data(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    pair_annotations_list: list[PairAnnotations],
    out_dir: Path,
) -> list[PairEditData]:
    pair_edit_data = [
        pair_annotations['line']['edit_data_by_threshold']['strict']
        for pair_annotations in pair_annotations_list
    ]
    save_json(pair_edit_data, out_dir, f'{PAIR_EDIT_DATA_PREFIX}{pair.corpus_name}{split_suffix}')
    return pair_edit_data
