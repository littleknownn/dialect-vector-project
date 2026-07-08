from src.config import ExperimentConfig, experiment_config, corpus_config
from src.utils import models_dir, weights_filename
from pathlib import Path
from safetensors.torch import load_file
import torch
import re


def extract_weight_tensors(
        corpus_name: str,
        experiment_config: ExperimentConfig,
        out_dir: Path = models_dir,
        out_filename: str = weights_filename,
    ) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    output_dir = out_dir / corpus_name
    out_filepath = output_dir / f'{out_filename}.pt'
    tensors_filepath = output_dir / 'adapter_model.safetensors'
    tensors_dict = load_file(tensors_filepath)

    target_tensors_pattern = '|'.join([
        re.escape(tensor_type)
        for tensor_type in experiment_config.target_weight_matrices
    ])
    key_pattern = re.compile(rf'layers\.(\d+)\..+\.({target_tensors_pattern})\.lora_(A|B)\.weight')

    weights = {}
    for key, tensor in tensors_dict.items():
        match_groups = key_pattern.search(key)
        if match_groups:
            layer_idx = int(match_groups.group(1))
            tensor_type = match_groups.group(2)
            tensor_factor = match_groups.group(3)

            if layer_idx not in weights:
                weights[layer_idx] = {}
            if tensor_type not in weights[layer_idx]:
                weights[layer_idx][tensor_type] = {}
        
        cleaned_tensor = tensor.detach().cpu().float()
        weights[layer_idx][tensor_type][tensor_factor] = cleaned_tensor

    torch.save(weights, out_filepath)


if __name__ == '__main__':
    for corpus_name in corpus_config:
        extract_weight_tensors(corpus_name, experiment_config)
    print('Weight Tensor Extraction Complete')