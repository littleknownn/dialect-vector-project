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


src_dir = Path(__file__).parent
root_dir = src_dir.parent

raw_data_dir = root_dir / 'data' / 'raw'
processed_data_dir = root_dir / 'data' / 'processed'
processed_data_dir.mkdir(parents=True, exist_ok=True)

models_dir = root_dir / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
weights_filename = 'lora_tensors'

checkpoints_subdir_name = 'checkpoints'

steering_vectors_dir = models_dir / 'steering_vectors'
steering_vectors_dir.mkdir(parents=True, exist_ok=True)
steering_vectors_filename = 'steering_vectors'

outputs_dir = root_dir / 'outputs'
outputs_dir.mkdir(parents=True, exist_ok=True)

metrics_dir = outputs_dir / 'metrics'
metrics_dir.mkdir(parents=True, exist_ok=True)
phase_1_metrics_filename = 'phase_1_metrics'


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
