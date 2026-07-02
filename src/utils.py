from pathlib import Path

src_dir = Path(__file__).parent
root_dir = src_dir.parent

raw_data_dir = root_dir / 'data' / 'raw'
processed_data_dir = root_dir / 'data' / 'processed'
models_dir = root_dir / 'models'