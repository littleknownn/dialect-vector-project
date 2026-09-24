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

ACTIVATIONS_DIR = DATA_DIR / '03_activations'
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATORS_DIR = DATA_DIR / '04_estimators'
ESTIMATORS_DIR.mkdir(parents=True, exist_ok=True)

SWEEPS_DIR = DATA_DIR / '05_sweeps'
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = DATA_DIR / '06_metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRIC_RESULTS_FILENAME = 'geometric_results'
BEHAVIORAL_RESULTS_FILENAME = 'behavioral_results'


def get_sweep_identifier(
        layer: int,
        pooling_type: PoolingTypeEnum,
        rank: int | None = None,
) -> str:
    identifier = f'layer{layer:02d}_{pooling_type}'
    if rank is not None:
        identifier += f'_{rank:02d}'
    return identifier
