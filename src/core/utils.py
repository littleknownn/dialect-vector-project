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


CORE_DIR = Path(__file__).parent
SRC_DIR = CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

ACTIVATIONS_DIR = ROOT_DIR / 'activations'
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATORS_DIR = ROOT_DIR / 'estimators'
ESTIMATORS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = ROOT_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = OUTPUTS_DIR / 'metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PHASE_1_METRICS_FILENAME = 'phase_1_metrics'


def save_jsonl(
        text_iter: Iterable[str],
        out_dir: Path,
        out_filename: str,
        min_len_chars: int,
        min_len_words: int
    ) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename += '.jsonl'
    out_path = out_dir / out_filename
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for text in text_iter:
            if len(text) < min_len_chars or len(text.split()) < min_len_words:
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
