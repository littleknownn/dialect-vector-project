import argparse
from pathlib import Path

from tqdm import tqdm

from src.analysis.pair_data import (
    load_pair_annotations_list,
    save_pair_edit_data,
)
from src.analysis.q1 import compute_and_save_q1_results
from src.analysis.q3 import compute_and_save_q3_results
from src.analysis.q4 import compute_and_save_q4_results
from src.analysis.q4_robustness import compute_and_save_q4_robustness_results
from src.core.config import (
    add_model_argument,
    get_model_configs,
)
from src.core.config_utils import (
    DatasetConfig,
    ExperimentConfig,
    get_estimator_data,
)
from src.core.data_provenance import (
    save_freeze_record,
    verify_run_setup,
)
from src.core.paths import (
    PROCESSED_DATA_DIR,
    RunDirs,
    get_run_dirs,
)
from src.core.utils import (
    RunMode,
    SplitSuffixes,
)


def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    split_suffix: SplitSuffixes,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    num_steps_per_pair = 4
    supervised_pairs = get_estimator_data(corpus_config).supervised_split_pairs
    step_bar = tqdm(total=num_steps_per_pair * len(supervised_pairs), desc='06 Analyzing Pairs')
    for pair in supervised_pairs:
        pair_annotations_list = load_pair_annotations_list(pair, split_suffix, experiment_config, run_dirs)
        pair_edit_data = save_pair_edit_data(pair, split_suffix, pair_annotations_list, run_dirs.pair_results)
        
        step_bar.set_postfix_str(f'Q1 {pair.corpus_name}')
        q1_projections = compute_and_save_q1_results(pair, split_suffix, pair_edit_data, pair_annotations_list, experiment_config, run_dirs, processed_data_dir)
        step_bar.update()
        
        step_bar.set_postfix_str(f'Q3 {pair.corpus_name}')
        compute_and_save_q3_results(pair, split_suffix, q1_projections.dialect_projections_by_pooling, pair_annotations_list, experiment_config, run_dirs.pair_results)
        step_bar.update()

        step_bar.set_postfix_str(f'Q4 {pair.corpus_name}')
        compute_and_save_q4_results(pair, split_suffix, q1_projections.dialect_projections_by_pooling, pair_annotations_list, experiment_config, run_dirs.pair_results)
        step_bar.update()

        step_bar.set_postfix_str(f'Q4 Robustness {pair.corpus_name}')
        compute_and_save_q4_robustness_results(pair, split_suffix, q1_projections, pair_annotations_list, experiment_config, run_dirs.pair_results)
        step_bar.update()
    step_bar.close()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--mode', type=RunMode, choices=list(RunMode), default=RunMode.DEVELOPMENT,
    )
    add_model_argument(argument_parser)
    arguments = argument_parser.parse_args()
    corpus_config, experiment_config = get_model_configs(arguments.model)
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(experiment_config)
    
    verify_run_setup(experiment_config, run_dirs, arguments.mode)
    compute_and_save_metrics(corpus_config, arguments.mode.split_suffix, experiment_config, run_dirs)
    if arguments.mode == RunMode.FREEZE:
        save_freeze_record(experiment_config, run_dirs)
    tqdm.write('06: Pair Metrics Computation Complete\n')
