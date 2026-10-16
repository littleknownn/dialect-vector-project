from pathlib import Path

from src.core.utils import PoolingTypeEnum


CORE_DIR = Path(__file__).parent
SRC_DIR = CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = DATA_DIR / '01_raw'

PROCESSED_DATA_DIR = DATA_DIR / '02_processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_LOG_FILENAME = 'processing_log'

ACTIVATIONS_DIR = DATA_DIR / '03_activations'
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTION_LOG_FILENAME = 'extraction_log'

ESTIMATORS_DIR = DATA_DIR / '04_estimators'
ESTIMATORS_DIR.mkdir(parents=True, exist_ok=True)

SWEEPS_DIR = DATA_DIR / '05_sweeps'
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = DATA_DIR / '06_metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
BEHAVIORAL_RESULTS_FILENAME = 'behavioral_results'
SWEEP_RESULTS_FILENAME = 'sweep_results'
H1_RESULTS_FILENAME = 'h1_results'

FIGURES_DIR = DATA_DIR / '07_figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


H1_SVD_PREFIX = 'h1_svd_'
H1_NULL_PREFIX = 'h1_null_'
H1_FLOORS_PREFIX = 'h1_floors_'

PAIR_EDIT_DATA_PREFIX = 'q4_pair_edit_data_'

Q1_FOLD_IDS_PREFIX = 'q1_fold_ids_'
Q1_FOLD_MEAN_VECTORS_PREFIX = 'q1_fold_mean_vectors_'
Q1_PROJECTIONS_PREFIX = 'q1_projections_'
Q1_RESULTS_PREFIX = 'q1_results_'

Q4_RESULTS_PREFIX = 'q4_results_'


def get_sweep_identifier(
        layer: int,
        pooling_type: PoolingTypeEnum,
        rank: int | None = None,
) -> str:
    identifier = f'layer{layer:02d}_{pooling_type}'
    if rank is not None:
        identifier += f'_{rank:02d}'
    return identifier


def get_model_activations_dir(
    model_hf_path: str,
    activations_dir: Path = ACTIVATIONS_DIR,
) -> Path:
    model_activations_dir = activations_dir / model_hf_path.split('/')[-1]
    model_activations_dir.mkdir(parents=True, exist_ok=True)
    return model_activations_dir
