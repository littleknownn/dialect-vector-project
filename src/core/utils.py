from enum import StrEnum
from pathlib import Path
import json
from typing import Iterable


type MetricsJson = dict[
    str, dict[
        str, dict[
            str, dict
        ]
    ]
]


class PoolingTypeEnum(StrEnum):
    LAST = 'last'
    MEAN = 'mean'


class AggregateEstimators(StrEnum):
    CHORDAL_MEAN_CORPORA = 'chordal_mean_corpora'
    CHORDAL_MEAN_ALL = 'chordal_mean_all'
    LEACE_CORPORA = 'leace_corpora'
    LEACE_ALL = 'leace_all'
    CONTROL = 'control'


CORE_DIR = Path(__file__).parent
SRC_DIR = CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = DATA_DIR / 'raw'

PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

ACTIVATIONS_DIR = DATA_DIR / 'activations'
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATORS_DIR = DATA_DIR / 'estimators'
ESTIMATORS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = DATA_DIR / 'metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def get_activation_filename(
        split_name: str,
        layer: int,
        pooling_type: PoolingTypeEnum,
) -> str:
    return f'{split_name}_layer{layer:02d}_{pooling_type}.pt'


def get_estimator_filename(
        estimator_name: str,
        layer: int,
        pooling_type: PoolingTypeEnum,
        rank: int,
) -> str:
    return f'{estimator_name}_layer{layer:02d}_{pooling_type}_{rank:02d}.pt'


def get_svd_estimator_name(split_name: str) -> str:
    return f'svd_{split_name}'


def get_supervised_estimator_name(corpus_name: str) -> str:
    return f'supervised_{corpus_name}'



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
            out_file.write(json.dumps(json_object) + '\n')


def save_json(
        data: dict,
        dir: Path,
        filename: str,
        json_indent: int = 4,
    ) -> None:
    out_filepath = dir / f'{filename}.json'
    with open(out_filepath, 'w') as file:
        json.dump(data, file, indent=json_indent)
