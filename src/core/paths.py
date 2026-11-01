from pathlib import Path
from typing import NamedTuple

from src.core.config_utils import ExperimentConfig
from src.core.utils import PoolingType


CORE_DIR = Path(__file__).parent
SRC_DIR = CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent
SCRIPTS_DIR = SRC_DIR / 'scripts'
PLOTS_DIR = SRC_DIR / 'plots'

PAPER_STYLE_FILENAME = 'paper'

DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / '00_raw'
PROCESSED_DATA_DIR = DATA_DIR / '01_processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = DATA_DIR / 'runs'

PROFANITY_LIST_FILENAME = 'profanity_list'
PROCESSING_LOG_FILENAME = 'processing_log'
EXTRACTION_LOG_FILENAME = 'extraction_log'
FREEZE_RECORD_FILENAME = 'freeze_record'

BEHAVIORAL_RESULTS_FILENAME = 'behavioral_results'
SWEEP_RESULTS_FILENAME = 'sweep_results'
H1_RESULTS_FILENAME = 'h1_results'

H1_SVD_PREFIX = 'svd_'
H1_NULL_PREFIX = 'null_'
H1_FLOORS_PREFIX = 'floors_'

Q1_PROJECTIONS_PREFIX = 'q1_projections_'
Q1_SHUFFLED_PROJECTIONS_PREFIX = 'q1_shuffled_projections_'
Q1_RESULTS_PREFIX = 'q1_results_'
Q3_RESULTS_PREFIX = 'q3_results_'
Q4_RESULTS_PREFIX = 'q4_results_'
Q4_ROBUSTNESS_PREFIX = 'q4_robustness_'

FOLD_IDS_PREFIX = 'fold_ids_'
PAIR_EDIT_DATA_PREFIX = 'q4_pair_edit_data_'


class RunDirs(NamedTuple):
    text_annotations: Path
    activations: Path
    token_states: Path
    
    difference_vectors: Path
    h1_estimators: Path
    fold_vectors: Path
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
        
        difference_vectors=estimators_dir / 'difference_vectors',
        h1_estimators=estimators_dir / 'h1',
        fold_vectors=estimators_dir / 'fold_vectors',
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
    pooling_type: PoolingType,
    rank: int | None = None,
) -> str:
    identifier = f'layer{layer:02d}_{pooling_type}'
    if rank is not None:
        identifier += f'_{rank:02d}'
    return identifier

