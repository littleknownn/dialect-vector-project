import gc
import itertools
import math
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
)

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.base_model import load_base_model
from src.core.config import (
    BaselineDatasetConfig,
    CORPUS_CONFIG,
    DatasetConfig,
    ExperimentConfig,
    EXPERIMENT_CONFIG,
    SupervisedDatasetConfig,
)
import src.core.math_utils as math_utils
from src.core.paths import (
    ESTIMATORS_DIR,
    PROCESSED_DATA_DIR,
    SWEEPS_DIR,
    get_sweep_identifier,
)
from src.core.utils import (
    MetricTypeEnum,
    PoolingTypeEnum,
    SweepMetric,
    load_jsonl,
    save_json,
)


type HiddenStateData = tuple[torch.Tensor, *tuple[Any, ...]]


type TokenizedInputs = dict[str, torch.Tensor]


class BatchedPairedInputs(NamedTuple):
    aae: list[TokenizedInputs]
    sae: list[TokenizedInputs]


class PairedLosses(NamedTuple):
    aae: torch.Tensor
    sae: torch.Tensor


class EstimatorSteering:
    def __init__(
            self,
            estimator_matrix: torch.Tensor,
            layer_module: int,
            alpha: float,
        ):
        self.estimator_matrix = estimator_matrix
        self.layer_module = layer_module
        self.alpha = alpha
        self.forward_hook = None

    def __enter__(self):
        self.forward_hook = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.forward_hook is not None:
            self.forward_hook.remove()

    def _hook(
        self,
        target_layer: nn.Module,
        previous_layer_inputs: HiddenStateData,
        previous_layer_outputs: torch.Tensor | HiddenStateData,
    ) -> torch.Tensor | HiddenStateData:
        is_output_wrapped = isinstance(previous_layer_outputs, tuple)
        hidden_states = previous_layer_outputs[0] if is_output_wrapped else previous_layer_outputs
        
        subspace_projection = (hidden_states @ self.estimator_matrix) @ self.estimator_matrix.T
        steered_hidden_states = hidden_states + self.alpha * subspace_projection

        if is_output_wrapped:
            return (steered_hidden_states, *previous_layer_outputs[1:])
        return steered_hidden_states


def _batch_and_tokenize(
    lines: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
) -> list[TokenizedInputs]:
    batched_inputs = []
    for batch in itertools.batched(lines, batch_size):
        tokens = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        batched_inputs.append(tokens)
    return batched_inputs


def _load_and_tokenize_baseline_corpora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    processed_data_dir: Path,
) -> dict[str, list[TokenizedInputs]]:
    baseline_corpora = {}
    for corpus_name, config in corpus_config.items():
        if isinstance(config, BaselineDatasetConfig):
            filepath = processed_data_dir / f'{corpus_name}.jsonl'
            baseline_corpora[corpus_name] = _batch_and_tokenize(
                load_jsonl(filepath)[:experiment_config.max_eval_lines],
                tokenizer,
                experiment_config.batch_size,
                experiment_config.max_seq_len,
                model.device,
            )
    return baseline_corpora


def _load_and_tokenize_supervised_corpora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    processed_data_dir: Path,
) -> dict[str, BatchedPairedInputs]:
    supervised_corpora = {}
    for corpus_name, config in corpus_config.items():
        if isinstance(config, SupervisedDatasetConfig):
            aae_filepath = processed_data_dir / f'{corpus_name}{config.aae_split_suffix}{config.testing_split_suffix}.jsonl'
            sae_filepath = processed_data_dir / f'{corpus_name}{config.sae_split_suffix}{config.testing_split_suffix}.jsonl'

            supervised_corpora[corpus_name] = BatchedPairedInputs(
                aae=_batch_and_tokenize(
                    load_jsonl(aae_filepath)[:experiment_config.max_eval_lines],
                    tokenizer,
                    experiment_config.batch_size,
                    experiment_config.max_seq_len,
                    model.device
                ),
                sae=_batch_and_tokenize(
                    load_jsonl(sae_filepath)[:experiment_config.max_eval_lines],
                    tokenizer,
                    experiment_config.batch_size,
                    experiment_config.max_seq_len,
                    model.device
                ),
            )       
    return supervised_corpora


def _compute_batched_input_losses(
    model: PreTrainedModel,
    batched_inputs: list[TokenizedInputs],
) -> torch.Tensor:
    losses = []
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    for inputs in batched_inputs:
        with torch.inference_mode():
            outputs = model(**inputs)

        known_token_logits = outputs.logits[..., :-1, :].contiguous()
        logit_labels = inputs['input_ids'][..., 1:].contiguous()

        token_losses_unmasked = loss_fn(known_token_logits.transpose(1, 2), logit_labels)
        shifted_attention_mask = inputs['attention_mask'][..., 1:].contiguous()
        token_losses = token_losses_unmasked * shifted_attention_mask

        num_token_losses = shifted_attention_mask.sum(dim=1)
        sum_token_losses = token_losses.sum(dim=1)
        input_losses = sum_token_losses / num_token_losses

        losses.append(input_losses)
    losses_tensor = torch.cat(losses)
    return losses_tensor


def _compute_dialect_llr_shift_ci(
    model: PreTrainedModel,
    batched_aae_inputs: list[TokenizedInputs],
    batched_sae_inputs: list[TokenizedInputs],
    unsteered_losses: PairedLosses,
    layer_module: nn.Module,
    estimator_matrix: torch.Tensor,
    alpha: float,
    num_bootstrap_samples: int,
    bootstrap_alpha: float,
) -> math_utils.BootstrapResults:
    with EstimatorSteering(estimator_matrix, layer_module, alpha):
        steered_aae_losses = _compute_batched_input_losses(model, batched_aae_inputs)
        steered_sae_losses = _compute_batched_input_losses(model, batched_sae_inputs)
    
    llr_shift_tensor = ((steered_sae_losses - steered_aae_losses) - (unsteered_losses.sae - unsteered_losses.aae)).to(torch.float32)
    bootstrap_results = math_utils.compute_bootstrap(llr_shift_tensor, num_bootstrap_samples, bootstrap_alpha)
    return bootstrap_results


def _compute_steered_token_perplexity(
    model: PreTrainedModel,
    inputs: TokenizedInputs,
    layer_module: nn.Module,
    estimator_matrix: torch.Tensor,
    alpha: float,
) -> float:
    with EstimatorSteering(estimator_matrix, layer_module, alpha):
        losses_tensor = _compute_batched_input_losses(model, inputs)
    mean_loss = losses_tensor.mean().item()
    perplexity = math.exp(mean_loss)
    return perplexity


def _compute_baseline_sweep_metrics(
    model: PreTrainedModel,
    baseline_corpora: dict[str, list[TokenizedInputs]],
    layer: int,
    pooling_type: PoolingTypeEnum,
    rank: int,
    estimator_name: str,
    estimator_matrix: torch.Tensor,
    alpha: float,
) -> list[SweepMetric]:
    sweep_metrics = []
    layer_module = model.model.layers[layer]
    for corpus_name, baseline_inputs in baseline_corpora.items():
        perplexity = _compute_steered_token_perplexity(
            model, baseline_inputs, layer_module, estimator_matrix, alpha
        )
        sweep_metrics.append(SweepMetric(
            layer=layer,
            pooling_type=pooling_type,
            rank=rank,
            
            corpus_name=corpus_name,
            estimator_name=estimator_name,
            alpha=alpha,
        
            metric=MetricTypeEnum.PERPLEXITY,
            value=perplexity,
        ))
    return sweep_metrics


def _compute_supervised_sweep_metrics(
    model: PreTrainedModel,
    supervised_corpora: dict[str, BatchedPairedInputs],
    unsteered_supervised_losses: dict[str, PairedLosses],
    layer: int,
    pooling_type: PoolingTypeEnum,
    rank: int,
    estimator_name: str,
    estimator_matrix: torch.Tensor,
    alpha: float,
    num_bootstrap_samples: int,
    bootstrap_alpha: float,
) -> list[SweepMetric]:
    sweep_metrics = []
    layer_module = model.model.layers[layer]
    for corpus_name, paired_inputs in supervised_corpora.items():
        llr_shift_stats = _compute_dialect_llr_shift_ci(
            model,
            paired_inputs.aae,
            paired_inputs.sae,
            unsteered_supervised_losses[corpus_name],
            layer_module,
            estimator_matrix,
            alpha,
            num_bootstrap_samples,
            bootstrap_alpha,
        )
        sweep_metrics.append(SweepMetric(
            layer=layer,
            pooling_type=pooling_type,
            rank=rank,
            
            corpus_name=corpus_name,
            estimator_name=estimator_name,
            alpha=alpha,
        
            metric=MetricTypeEnum.LLR_SHIFT,
            value=llr_shift_stats.mean,
            ci_lower=llr_shift_stats.ci_lower,
            ci_upper=llr_shift_stats.ci_upper,
            p_value=llr_shift_stats.p_value,
        ))
    return sweep_metrics


def compute_and_save_subspace_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    processed_data_dir: Path,
    estimators_dir: Path,
    out_dir: Path,
) -> None:
    tqdm.write('Tokenizing Reference Corpora...')
    baseline_corpora = _load_and_tokenize_baseline_corpora(
        model, tokenizer, corpus_config, experiment_config, processed_data_dir
    )
    supervised_corpora = _load_and_tokenize_supervised_corpora(
        model, tokenizer, corpus_config, experiment_config, processed_data_dir
    )

    tqdm.write('Computing Unsteered Losses...')
    unsteered_supervised_losses = {
        corpus_name: PairedLosses(
            aae=_compute_batched_input_losses(model, paired_inputs.aae),
            sae=_compute_batched_input_losses(model, paired_inputs.sae),
        )
        for corpus_name, paired_inputs in supervised_corpora.items()
    }

    sweeps_tuples = itertools.product(
        experiment_config.target_layers,
        PoolingTypeEnum,
        experiment_config.target_ranks,
    )
    for layer, pooling_type, rank in sweeps_tuples:
        sweep_identifier = get_sweep_identifier(layer, pooling_type, rank)
        if (out_dir / f'{sweep_identifier}.json').exists():
            continue
        tqdm.write(f'Evaluating Sweep: Layer {layer:02d}, {pooling_type} Pooling, Rank {rank}...')

        sweep_metrics = []
        estimator_paths = estimators_dir.glob(f'*_{sweep_identifier}.pt')
        for estimator_path in estimator_paths:
            estimator_name = estimator_path.name.removesuffix(f'_{sweep_identifier}.pt')
            estimator_matrix = torch.load(estimator_path, weights_only=True).to(model.device, dtype=model.dtype)

            for alpha in tqdm(experiment_config.target_alphas, desc=f'Evaluating {estimator_name} Estimator'):
                sweep_metrics.extend(_compute_baseline_sweep_metrics(
                    model,
                    baseline_corpora,
                    layer,
                    pooling_type,
                    rank,
                    estimator_name,
                    estimator_matrix,
                    alpha,
                ))
                sweep_metrics.extend(_compute_supervised_sweep_metrics(
                    model,
                    supervised_corpora,
                    unsteered_supervised_losses,
                    layer,
                    pooling_type,
                    rank,
                    estimator_name,
                    estimator_matrix,
                    alpha,
                    experiment_config.num_bootstrap_samples,
                    experiment_config.bootstrap_alpha,
                ))
                
            del estimator_matrix
            gc.collect()
            torch.cuda.empty_cache()
        
        save_json(
            sweep_metrics,
            out_dir,
            sweep_identifier,
        )


if __name__ == '__main__':
    model, tokenizer = load_base_model(EXPERIMENT_CONFIG)
    compute_and_save_subspace_metrics(
        model, tokenizer, CORPUS_CONFIG, EXPERIMENT_CONFIG, PROCESSED_DATA_DIR, ESTIMATORS_DIR, SWEEPS_DIR
    )
    tqdm.write('04: Steering Evaluation Complete\n')
