import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.config_utils import (
    LOGGED_EXPERIMENT_CONFIG_FIELDS,
    ExperimentConfig,
)
from src.core.paths import (
    CORE_DIR,
    EXTRACTION_LOG_FILENAME,
    FREEZE_RECORD_FILENAME,
    PROCESSED_DATA_DIR,
    SCRIPTS_DIR,
    RunDirs,
)
from src.core.utils import (
    FreezeRecord,
    RunMode,
    load_json,
    save_json,
)


def compute_file_hashes(
    in_dir: Path,
) -> dict[str, str]:
    file_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(in_dir.iterdir())
        if path.is_file()
    }
    return file_hashes


def serialize_config(
    experiment_config: ExperimentConfig,
) -> dict[str, Any]:
    serialized_config = json.loads(json.dumps(dataclasses.asdict(experiment_config)))
    return serialized_config


def _find_setup_mismatches(
    logged_config: dict[str, Any],
    logged_hashes: dict[str, str],
    experiment_config: ExperimentConfig,
    config_fields: tuple[str, ...],
    processed_data_dir: Path,
) -> list[str]:
    current_config = serialize_config(experiment_config)
    current_hashes = compute_file_hashes(processed_data_dir)
    mismatched_fields = [
        f'config: {field}'
        for field in config_fields
        if logged_config[field] != current_config[field]
    ]
    mismatched_files = sorted(
        f'file: {filename}'
        for filename in logged_hashes.keys() | current_hashes.keys()
        if logged_hashes.get(filename) != current_hashes.get(filename)
    )
    mismatches = mismatched_fields + mismatched_files
    return mismatches


def verify_extraction_log(
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    logged_experiment_config_fields: tuple[str, ...] = LOGGED_EXPERIMENT_CONFIG_FIELDS,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    extraction_log_filename: str = EXTRACTION_LOG_FILENAME,
) -> None:
    extraction_log = load_json(run_dirs.activations / f'{extraction_log_filename}.json')
    mismatches = _find_setup_mismatches(
        extraction_log['experiment_config'],
        extraction_log['processed_data_hashes'],
        experiment_config,
        logged_experiment_config_fields,
        processed_data_dir,
    )
    if mismatches:
        raise ValueError(f'Stored activations do not match the current setup: {mismatches}')


def save_freeze_record(
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    core_dir: Path = CORE_DIR,
    scripts_dir: Path = SCRIPTS_DIR,
    freeze_record_filename: str = FREEZE_RECORD_FILENAME,
) -> None:
    freeze_record_filepath = run_dirs.pair_results / f'{freeze_record_filename}.json'
    if freeze_record_filepath.exists():
        raise FileExistsError(f"Study already frozen: {freeze_record_filepath}")
    freeze_record = FreezeRecord(
        source_code_hashes=compute_file_hashes(core_dir) | compute_file_hashes(scripts_dir),
        experiment_config=serialize_config(experiment_config),
        processed_data_hashes=compute_file_hashes(processed_data_dir),
    )
    save_json(freeze_record, run_dirs.pair_results, freeze_record_filename)


def verify_freeze_record(
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    freeze_record_filename: str = FREEZE_RECORD_FILENAME,
) -> None:
    freeze_record_filepath = run_dirs.pair_results / f'{freeze_record_filename}.json'
    if not freeze_record_filepath.exists():
        raise FileNotFoundError(f"Testing mode requires freeze record: {freeze_record_filepath}")
    freeze_record = load_json(freeze_record_filepath)
    mismatches = _find_setup_mismatches(
        freeze_record['experiment_config'],
        freeze_record['processed_data_hashes'],
        experiment_config,
        tuple(freeze_record['experiment_config']),
        processed_data_dir,
    )
    if mismatches:
        raise ValueError(f'Setup cannot change after freeze: {mismatches}')


def verify_run_setup(
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    run_mode: RunMode,
) -> None:
    verify_extraction_log(experiment_config, run_dirs)
    if run_mode == RunMode.TESTING:
        verify_freeze_record(experiment_config, run_dirs)
