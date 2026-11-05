from collections.abc import Callable
import difflib
import functools
import re
import string
from typing import NamedTuple

import spacy
import torch

import src.core.text_rules as text_rules
from src.core.utils import PairBoundary
from src.core.config_utils import (
    DatasetConfig,
    SupervisedDatasetConfig,
)


_EDGE_PUNCTUATION: str = string.punctuation.replace("'", '')
_MASK_EDGE_PUNCTUATION: str = _EDGE_PUNCTUATION.replace('*', '')
_WORD_PATTERN: re.Pattern = re.compile(r'\S+')


class CandidateBoundary(NamedTuple):
    word_a: re.Match
    word_b: re.Match
    distance_from_midpoint: float


@functools.cache
def _load_sentencizer() -> spacy.Language:
    sentencizer = spacy.blank('en')
    sentencizer.add_pipe('sentencizer')
    return sentencizer


def _normalize_word(
    word: str,
) -> str:
    return word.lower().strip(_EDGE_PUNCTUATION)


def normalize_words(
    text: str,
    replacement_patterns: tuple[tuple[str, str], ...] = text_rules.Q4_NORMALIZATION_REPLACEMENT_PATTERNS,
) -> list[str]:
    for pattern, replacement in replacement_patterns:
        text = re.sub(pattern, replacement, text)
    words = [word for word in map(_normalize_word, text.split()) if word]
    return words


def _compute_candidate_boundaries(
    text_a: str,
    text_b: str,
    min_previous_words: int,
    min_following_words: int,
) -> list[CandidateBoundary]:
    words_a = list(_WORD_PATTERN.finditer(text_a))
    words_b = list(_WORD_PATTERN.finditer(text_b))
    matcher = difflib.SequenceMatcher(
        None,
        [_normalize_word(word.group()) for word in words_a],
        [_normalize_word(word.group()) for word in words_b],
        autojunk=False,
    )
    candidate_boundaries = []
    for start_idx_a, start_idx_b, block_size in matcher.get_matching_blocks():
        for idx_a, idx_b in zip(
            range(start_idx_a, start_idx_a + block_size),
            range(start_idx_b, start_idx_b + block_size),
        ):
            is_boundary_valid = (
                min(idx_a, idx_b) + 1 >= min_previous_words
                and len(words_a) - idx_a > min_following_words
                and len(words_b) - idx_b > min_following_words
            )
            if is_boundary_valid and _normalize_word(words_a[idx_a].group()):
                distance_from_midpoint = abs(
                    (idx_b + 1) - (len(words_b) / 2)
                )
                candidate_boundaries.append(CandidateBoundary(
                    word_a=words_a[idx_a],
                    word_b=words_b[idx_b],
                    distance_from_midpoint=distance_from_midpoint,
                ))
    return candidate_boundaries


def _compute_boundary(
    text_a: str,
    text_b: str,
    sort_fn: Callable[
        [CandidateBoundary],
        tuple[bool, float, int] | tuple[int, int],
    ],
    min_previous_words: int,
    min_following_words: int,
) -> CandidateBoundary | None:
    candidate_boundaries = _compute_candidate_boundaries(text_a, text_b, min_previous_words, min_following_words)
    boundary = min(candidate_boundaries, key=sort_fn, default=None)
    return boundary


def check_text_len_bounds(
    text: str, 
    min_len_chars: int,
    min_len_words: int,
    max_len_words: int | None,
) -> bool:
    num_chars = len(text)
    num_words = len(text.split())
    above_lower_bounds = min_len_chars <= num_chars and min_len_words <= num_words
    below_upper_bounds = num_words <= max_len_words if max_len_words is not None else True
    return above_lower_bounds and below_upper_bounds


def clean_corpus_text(
    text: str,
    config: DatasetConfig,
    cleanup_replace_map: dict[int, int | str | None] = text_rules.CLEANUP_REPLACEMENT_MAP,
) -> str | None:
    text = text.translate(cleanup_replace_map)
    text_lowercase = text.lower()
    for pattern in config.skipped_data_patterns:
        if pattern.search(text_lowercase):
            return None    
    
    for pattern, replacement in config.replacement_patterns:
        text = pattern.sub(replacement, text)

    if not (cleaned_text := text.strip()):
        return None
    
    for pattern in config.skipped_case_patterns:
        if pattern.search(cleaned_text):
            return None
    
    return cleaned_text


def get_matched_words(
    text: str,
    match_words: frozenset[str],
) -> list[str]:
    matches = [
        word for word in map(_normalize_word, text.split())
        if word in match_words
    ]
    return matches


def has_masked_word(
    text: str,
) -> bool:
    for word in text.lower().split():
        word = word.strip(_MASK_EDGE_PUNCTUATION).lstrip('*')
        word = text_rules.ACTION_MARKER_PATTERN.sub('', word)
        if text_rules.MASKED_WORD_PATTERN.fullmatch(word):
            return True
    return False


def compute_pair_boundary(
    aae_text: str,
    sae_text: str,
    aae_back_text: str,
    config: SupervisedDatasetConfig,
) -> PairBoundary | None:
    distance_to_midpoint_sort_key_fn = lambda candidate_boundary: (
        candidate_boundary.word_a.group() != candidate_boundary.word_b.group(),
        candidate_boundary.distance_from_midpoint,
        candidate_boundary.word_b.end(),
    )
    aae_to_sae_boundary = _compute_boundary(
        aae_text,
        sae_text,
        sort_fn=distance_to_midpoint_sort_key_fn,
        min_previous_words=config.boundary_min_previous_words,
        min_following_words=config.boundary_min_following_words,
    )
    
    if aae_to_sae_boundary is None:
        return None
    sae_idx = aae_to_sae_boundary.word_b.end()
    
    distance_to_sae_sort_key_fn = lambda candidate_boundary: (
        abs(candidate_boundary.word_a.end() - sae_idx),
        candidate_boundary.word_a.end(),
    )
    sae_to_aae_back_boundary = _compute_boundary(
        sae_text,
        aae_back_text,
        sort_fn=distance_to_sae_sort_key_fn,
        min_previous_words=config.boundary_min_previous_words,
        min_following_words=config.boundary_min_following_words,
    )
    
    aae_idx = aae_to_sae_boundary.word_a.end()
    aae_back_idx = (
        sae_to_aae_back_boundary.word_b.end()
        if sae_to_aae_back_boundary else None
    )

    is_exact = (aae_to_sae_boundary.word_a.group() == aae_to_sae_boundary.word_b.group())
    is_aae_back_shared = (
        sae_to_aae_back_boundary is not None
        and sae_to_aae_back_boundary.word_a.end() == sae_idx
    )

    pair_boundary = PairBoundary(
        aae_idx=aae_idx,
        sae_idx=sae_idx,
        aae_back_idx=aae_back_idx,
        is_exact=is_exact,
        is_aae_back_shared=is_aae_back_shared,
    )
    return pair_boundary


def compute_line_midpoint_char_idx(
    text: str,
    config: SupervisedDatasetConfig,
) -> int | None:
    midpoint_sort_key_fn = lambda candidate_boundary: (
        candidate_boundary.distance_from_midpoint,
        candidate_boundary.word_a.end(),
    )
    midpoint = _compute_boundary(
        text,
        text,
        sort_fn=midpoint_sort_key_fn,
        min_previous_words=config.boundary_min_previous_words,
        min_following_words=config.boundary_min_following_words,
    )
    midpoint_char_idx = (
        midpoint.word_a.end()
        if midpoint
        else None
    )
    return midpoint_char_idx


def get_boundary_token_idx(
    token_char_indices: torch.Tensor,
    boundary_char_idx: int,
) -> int:
    token_end_char_indices = token_char_indices[:, 1]
    is_before_boundary = (
        (token_end_char_indices > 0)
        & (token_end_char_indices <= boundary_char_idx)
    )
    boundary_token_idx = int(is_before_boundary.nonzero().max())
    return boundary_token_idx


def sentencize(text: str) -> list[str]:
    sentencizer = _load_sentencizer()
    sentences = [sentence.text for sentence in sentencizer(text).sents]
    return sentences
