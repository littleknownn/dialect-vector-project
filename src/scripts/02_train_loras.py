from src.utils import processed_data_dir, models_dir
from src.config import ExperimentConfig, experiment_config
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import  SFTTrainer, SFTConfig

def fit_lora(experiment_config: ExperimentConfig, corpus_name: str) -> None:
    output_dir = models_dir / corpus_name
    corpus_path = processed_data_dir / f'{corpus_name}.jsonl'

    corpus = load_dataset('json', data_files=str(corpus_path), split='train')
    num_training_samples = min(experiment_config.max_training_samples, len(corpus))
    train_dataset = corpus.shuffle(seed=experiment_config.random_seed).select(range(num_training_samples))

    tokenizer = AutoTokenizer.from_pretrained(experiment_config.model_hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_compute_dtype = torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        experiment_config.model_hf_path,
        quantization_config = quantization_config,
        device_map = 'auto',
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r = experiment_config.lora_rank,
        lora_alpha = experiment_config.lora_alpha,
        target_modules = experiment_config.target_weight_matrices,
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

    lora_training_config = SFTConfig(
        output_dir = str(models_dir / corpus_name / 'checkpoints'),
        per_device_train_batch_size = experiment_config.batch_size,
        learning_rate = experiment_config.learning_rate,
        num_train_epochs = experiment_config.epochs,
        dataset_text_field = 'text',
        packing = False,
        gradient_checkpointing = True,
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        peft_config = peft_config,
        processing_class = tokenizer,
        args = lora_training_config,
    )

    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    models_dir.mkdir(parents=True, exist_ok=True)
    fit_lora(experiment_config, 'coraal_merged')
    fit_lora(experiment_config, 'twitter_aae')
    fit_lora(experiment_config, 'aave_lyrics')
    print('LoRA Fitting Complete')
