from src.core.config import ExperimentConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

def get_quantization_config(experiment_config: ExperimentConfig) -> BitsAndBytesConfig:
    if not experiment_config.load_in_4bit:
        return None
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = experiment_config.load_in_4bit,
        bnb_4bit_quant_type = experiment_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = torch.bfloat16,
    )
    return quantization_config

def load_base_model(experiment_config: ExperimentConfig) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(
        experiment_config.model_hf_path,
        quantization_config = get_quantization_config(experiment_config),
        dtype = 'auto',
        device_map = 'auto',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(experiment_config.model_hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    return model, tokenizer
