from pathlib import Path
from typing import NamedTuple

from src.core.utils import PoolingTypeEnum


CORE_DIR = Path(__file__).parent
SRC_DIR = CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / '00_raw'
PROCESSED_DATA_DIR = DATA_DIR / '01_processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = DATA_DIR / 'runs'

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


class RunDirs(NamedTuple):
    activations: Path
    token_states: Path
    steering_estimators: Path
    h1_estimators: Path
    q1_estimators: Path
    sweeps: Path
    metrics: Path
    figures: Path


def get_run_dirs(
    model_hf_path: str,
    debug_max_split_lines: int | None,
) -> RunDirs:
    model_name = model_hf_path.split('/')[-1]
    if debug_max_split_lines is not None:
        model_name += f'_debug{debug_max_split_lines}'

    run_dir = RUNS_DIR / model_name
    activations_dir = run_dir / '02_activations'
    estimators_dir = run_dir / '03_estimators'
    run_dirs = RunDirs(
        activations=activations_dir,
        token_states=activations_dir / 'token_states',
        steering_estimators=estimators_dir / 'steering',
        h1_estimators=estimators_dir / 'h1',
        q1_estimators=estimators_dir / 'q1',
        sweeps=run_dir / '04_sweeps',
        metrics=run_dir / '05_metrics',
        figures=run_dir / '06_figures',
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
