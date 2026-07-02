from dataclasses import dataclass
from typing import Union, List

@dataclass
class ExperimentConfig:
    model_hf_path: str
    lora_rank: int
    lora_alpha: int
    target_weight_matrices: Union[List[str], str]

    max_training_samples: int
    batch_size: int
    max_seq_len: int
    learning_rate: float
    epochs: int
    random_seed: int

experiment_config = ExperimentConfig(
    model_hf_path='Qwen/Qwen2.5-1.5B',
    lora_rank = 8,
    lora_alpha = 16,
    target_weight_matrices = ['v_proj', 'o_proj'], # ['q_proj', 'v_proj'], 'all-linear'

    max_training_samples = 5499,
    batch_size = 2,
    max_seq_len = 512,
    learning_rate = 2e-4,
    epochs = 1,
    random_seed = 42,
)
