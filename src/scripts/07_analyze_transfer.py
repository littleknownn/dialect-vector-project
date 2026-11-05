import argparse

from tqdm import tqdm

from src.analysis.q2 import compute_and_save_q2_results
from src.core.config import (
    add_model_argument,
    get_model_configs,
)
from src.core.config_utils import (
    DatasetConfig,
    ExperimentConfig,
)
from src.core.data_provenance import verify_run_setup
from src.core.paths import (
    RunDirs,
    get_run_dirs,
)
from src.core.utils import RunMode


def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> None:
    compute_and_save_q2_results(corpus_config, experiment_config, run_dirs)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    add_model_argument(argument_parser)
    corpus_config, experiment_config = get_model_configs(argument_parser.parse_args().model)
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(experiment_config)
    
    verify_run_setup(experiment_config, run_dirs, RunMode.TESTING)
    compute_and_save_metrics(corpus_config, experiment_config, run_dirs)
    tqdm.write('07: Transfer Metrics Computation Complete\n')
