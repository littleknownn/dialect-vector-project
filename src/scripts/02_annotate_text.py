import argparse
from pathlib import Path

import inflect
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.core.base_model import load_tokenizer
from src.core.config import (
    add_model_argument,
    get_model_configs,
)
from src.core.config_utils import ExperimentConfig
import src.core.edit_tagger as edit_tagger
from src.core.paths import (
    PROCESSED_DATA_DIR,
    PROFANITY_LIST_FILENAME,
    RAW_DATA_DIR,
    get_run_dirs,
)
import src.core.text_rules as text_rules
import src.core.text_utils as text_utils
from src.core.utils import (
    LineAnnotations,
    PairAnnotations,
    PairBoundary,
    PairEditData,
    Q4TaggerInputs,
    SplitSuffixes,
    TextPairAnnotations,
    load_json,
    load_jsonl,
    save_json,
)


def _load_profanities(
    filepath: Path,
) -> frozenset[str]:
    lines = filepath.read_text(encoding='utf-8').splitlines()
    candidate_profanities = (line.strip().lower() for line in lines)
    words = {
        word for word in candidate_profanities
        if word and ' ' not in word
    }
    inflect_engine = inflect.engine()
    standard_plurals = {inflect_engine.plural_noun(word) for word in words}
    nonstandard_plurals = {word + 's' for word in words}
    profanities = frozenset(words | standard_plurals | nonstandard_plurals)
    return profanities


def _annotate_line(
    line: str,
    num_tokens: int,
    num_special_tokens: int,
    profanities: frozenset[str],
    max_seq_len: int,
) -> LineAnnotations:
    num_words = len(line.split())
    fragmentation = num_tokens / num_words
    is_truncated = num_tokens + num_special_tokens > max_seq_len
    has_mention = text_rules.MENTION_PLACEHOLDER in line
    has_masked_word = text_utils.has_masked_word(line)
    has_quotative_be_like = bool(text_rules.QUOTATIVE_BE_LIKE_PATTERN.search(line))
    has_terminal_punctuation = bool(text_rules.TERMINAL_PUNCTUATION_PATTERN.search(line))
    probes = [
        feature for feature, pattern in text_rules.AAE_FEATURE_PATTERNS.items()
        if pattern.search(line)
    ]
    profanity_matches = text_utils.get_matched_words(line, profanities)
    line_annotations = LineAnnotations(
        num_words=num_words,
        fragmentation=fragmentation,
        is_truncated=is_truncated,
        has_mention=has_mention,
        has_masked_word=has_masked_word,
        has_quotative_be_like=has_quotative_be_like,
        has_terminal_punctuation=has_terminal_punctuation,
        probes=probes,
        profanity_matches=profanity_matches,
    )
    return line_annotations


def _compute_edit_data_by_threshold(
    aae_text: str,
    sae_text: str,
    experiment_config: ExperimentConfig,
) -> dict[str, PairEditData]:
    edit_data_by_threshold = {
        threshold_name: edit_tagger.compute_pair_edit_data(aae_text, sae_text, threshold, experiment_config.tagging.function_words)
        for threshold_name, threshold in experiment_config.tagging.orthographic_thresholds._asdict().items()
    }
    return edit_data_by_threshold


def _annotate_text_pair(
    aae_text: str,
    sae_text: str,
    tagger_inputs: Q4TaggerInputs,
    experiment_config: ExperimentConfig,
) -> TextPairAnnotations:
    text_pair_annotations = TextPairAnnotations(
        edit_data_by_threshold=_compute_edit_data_by_threshold(aae_text, sae_text, experiment_config),
        edits=edit_tagger.compute_pair_edits(aae_text, sae_text, tagger_inputs),
        controls=edit_tagger.compute_pair_controls(aae_text, sae_text, tagger_inputs.profanities),
        num_aae_features=sum(
            bool(pattern.search(aae_text))
            for pattern in text_rules.AAE_FEATURE_PATTERNS.values()
        ),
    )
    return text_pair_annotations


def _annotate_pair(
    aae_line: str,
    sae_line: str,
    boundary: PairBoundary | None,
    tagger_inputs: Q4TaggerInputs,
    experiment_config: ExperimentConfig,
) -> PairAnnotations:    
    context_annotations = None
    if boundary:
        aae_context = aae_line[:boundary['aae_idx']]
        sae_context = sae_line[:boundary['sae_idx']]
        context_annotations = _annotate_text_pair(aae_context, sae_context, tagger_inputs, experiment_config)
    
    pair_annotations = PairAnnotations(
        line=_annotate_text_pair(aae_line, sae_line, tagger_inputs, experiment_config),
        context=context_annotations,
    )
    return pair_annotations


def annotate_and_save_lines(
    tokenizer: PreTrainedTokenizerBase,
    experiment_config: ExperimentConfig,
    out_dir: Path,
    in_dir: Path = PROCESSED_DATA_DIR,
    profanity_list_filepath: Path = RAW_DATA_DIR / f'{PROFANITY_LIST_FILENAME}.txt',
) -> None:
    profanities = _load_profanities(profanity_list_filepath)
    num_special_tokens = tokenizer.num_special_tokens_to_add()
    for corpus_path in sorted(in_dir.glob('*.jsonl')):
        lines = load_jsonl(corpus_path)[:experiment_config.debug_max_split_lines]
        token_ids_by_line = tokenizer(lines, add_special_tokens=False)['input_ids']
        line_annotations = [
            _annotate_line(line, len(token_ids), num_special_tokens, profanities, experiment_config.max_seq_len)
            for line, token_ids in zip(lines, token_ids_by_line)
        ]
        save_json(line_annotations, out_dir, corpus_path.stem)


def annotate_and_save_pairs(
    experiment_config: ExperimentConfig,
    out_dir: Path,
    in_dir: Path = PROCESSED_DATA_DIR,
    profanity_list_filepath: Path = RAW_DATA_DIR / f'{PROFANITY_LIST_FILENAME}.txt',
) -> None:
    tagger_inputs = Q4TaggerInputs(
        function_words=frozenset(experiment_config.tagging.function_words),
        grammar_words=frozenset(experiment_config.tagging.grammar_words),
        copula_words=frozenset(experiment_config.tagging.copula_words),
        profanities=_load_profanities(profanity_list_filepath),
        respelling_threshold=experiment_config.tagging.respelling_threshold,
        single_letter_respelling_threshold=experiment_config.tagging.single_letter_respelling_threshold,
        aint_words=frozenset(experiment_config.tagging.aint_words)
    )
    for metadata_filepath in sorted(in_dir.glob(f'*{SplitSuffixes.PAIR_METADATA}*.json')):
        corpus_name, split_suffix = metadata_filepath.stem.split(SplitSuffixes.PAIR_METADATA)
        num_pairs = experiment_config.debug_max_split_lines
        pair_metadata = load_json(metadata_filepath)[:num_pairs]
        aae_lines = load_jsonl(in_dir / f'{corpus_name}{SplitSuffixes.AAE}{split_suffix}.jsonl')[:num_pairs]
        sae_lines = load_jsonl(in_dir / f'{corpus_name}{SplitSuffixes.SAE}{split_suffix}.jsonl')[:num_pairs]
        pair_annotations = [
            _annotate_pair(aae_line, sae_line, metadata['boundary'], tagger_inputs, experiment_config)
            for aae_line, sae_line, metadata in zip(aae_lines, sae_lines, pair_metadata, strict=True)
        ]
        save_json(pair_annotations, out_dir, f'{corpus_name}{SplitSuffixes.PAIR_ANNOTATIONS}{split_suffix}')


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    add_model_argument(argument_parser)
    corpus_config, experiment_config = get_model_configs(argument_parser.parse_args().model)
    run_dirs = get_run_dirs(experiment_config)
    
    tokenizer = load_tokenizer(experiment_config)
    annotate_and_save_lines(tokenizer, experiment_config, run_dirs.text_annotations)
    annotate_and_save_pairs(experiment_config, run_dirs.text_annotations)
    tqdm.write('02: Text Annotation Complete')
