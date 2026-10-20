import dataclasses
import json
from pathlib import Path
from typing import NamedTuple

from src.core.config_utils import (
    LOGGED_EXPERIMENT_CONFIG_FIELDS,
    ExperimentConfig,
)
from src.core.utils import (
    PoolingTypeEnum,
    compute_file_hashes,
    load_json,
)


CORE_DIR = Path(__file__).parent
SRC_DIR = CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / '00_raw'
PROCESSED_DATA_DIR = DATA_DIR / '01_processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = DATA_DIR / 'runs'

PROFANITY_LIST_FILENAME = 'profanity_list'
PROCESSING_LOG_FILENAME = 'processing_log'
EXTRACTION_LOG_FILENAME = 'extraction_log'

BEHAVIORAL_RESULTS_FILENAME = 'behavioral_results'
SWEEP_RESULTS_FILENAME = 'sweep_results'
H1_RESULTS_FILENAME = 'h1_results'

H1_SVD_PREFIX = 'svd_'
H1_NULL_PREFIX = 'null_'
H1_FLOORS_PREFIX = 'floors_'

Q1_FOLD_IDS_PREFIX = 'fold_ids_'
Q1_FOLD_MEAN_VECTORS_PREFIX = 'fold_mean_vectors_'
Q1_PROJECTIONS_PREFIX = 'q1_projections_'
Q1_RESULTS_PREFIX = 'q1_results_'

Q4_RESULTS_PREFIX = 'q4_results_'
PAIR_EDIT_DATA_PREFIX = 'q4_pair_edit_data_'
DIFFERENCE_VECTOR_PREFIX = 'difference_vector_'

class RunDirs(NamedTuple):
    text_annotations: Path
    activations: Path
    token_states: Path
    steering_estimators: Path
    h1_estimators: Path
    q1_estimators: Path
    sweeps: Path
    pair_results: Path
    transfer_results: Path
    steering_results: Path
    figures: Path


def get_run_dirs(
    experiment_config: ExperimentConfig,
) -> RunDirs:
    model_name = experiment_config.model_hf_path.split('/')[-1]
    if experiment_config.debug_max_split_lines is not None:
        model_name += f'_debug{experiment_config.debug_max_split_lines}'

    run_dir = RUNS_DIR / model_name
    activations_dir = run_dir / '03_activations'
    estimators_dir = run_dir / '04_estimators'
    run_dirs = RunDirs(
        text_annotations=run_dir / '02_text_annotations',
        activations=activations_dir,
        token_states=activations_dir / 'token_states',
        steering_estimators=estimators_dir / 'steering',
        h1_estimators=estimators_dir / 'h1',
        q1_estimators=estimators_dir / 'q1',
        sweeps=run_dir / '05_sweeps',
        pair_results=run_dir / '06_pair',
        transfer_results=run_dir / '07_transfer',
        steering_results=run_dir / '08_steering',
        figures=run_dir / 'figures',
    )
    
    for subdir in run_dirs:
        subdir.mkdir(parents=True, exist_ok=True)
    return run_dirs


def get_sweep_tensor_filepath(
    sweep_root_dir: Path,
    sweep_identifier: str,
    name: str,
) -> Path:
    sweep_dir = sweep_root_dir / sweep_identifier
    sweep_dir.mkdir(parents=True, exist_ok=True)
    filepath = sweep_dir / f'{name}.pt'
    return filepath


def get_sweep_identifier(
    layer: int,
    pooling_type: PoolingTypeEnum,
    rank: int | None = None,
) -> str:
    identifier = f'layer{layer:02d}_{pooling_type}'
    if rank is not None:
        identifier += f'_{rank:02d}'
    return identifier


def verify_extraction_log(
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    logged_experiment_config_fields: tuple[str, ...] = LOGGED_EXPERIMENT_CONFIG_FIELDS,
) -> None:
    extraction_log = load_json(run_dirs.activations / f'{EXTRACTION_LOG_FILENAME}.json')
    logged_config = extraction_log['experiment_config']
    logged_hashes = extraction_log['processed_data_hashes']
    current_config = json.loads(json.dumps(dataclasses.asdict(experiment_config))) # simulate pre-logging
    current_hashes = compute_file_hashes(processed_data_dir)

    mismatched_fields = [
        field for field in logged_experiment_config_fields
        if logged_config[field] != current_config[field]
    ]
    mismatched_files = sorted(
        filename for filename in logged_hashes.keys() | current_hashes.keys()
        if logged_hashes.get(filename) != current_hashes.get(filename)
    )

    if mismatched_fields or mismatched_files:
        raise ValueError(
            'Stored activations do not match the current setup.\n'
            f'Configuration fields not matching: {mismatched_fields}\n'
            f'Processed files not matching: {mismatched_files}'
        )
