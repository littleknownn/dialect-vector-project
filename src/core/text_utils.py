from collections import Counter
from collections.abc import Callable
import difflib
import functools
import re
import string
from typing import NamedTuple

import spacy

import src.core.text_rules as text_rules
from src.core.utils import (
    PairBoundary,
    PairEditData,
    Q4Stratum,
)
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


def _normalize_word(
    word: str,
) -> str:
    return word.lower().strip(_EDGE_PUNCTUATION)


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


def _normalize_words(
    text: str,
    replacement_patterns: tuple[tuple[str, str], ...] = text_rules.Q4_NORMALIZATION_REPLACEMENT_PATTERNS,
) -> list[str]:
    for pattern, replacement in replacement_patterns:
        text = re.sub(pattern, replacement, text)
    words = [word for word in map(_normalize_word, text.split()) if word]
    return words


def _classify_edit(
    aae_words: list[str],
    sae_words: list[str],
    orthographic_threshold: float,
    function_words: tuple[str, ...],
) -> Q4Stratum:
    def _get_letters(
        words: list[str],
    ) -> str:
        return re.sub(r'[^\da-z]', '', ''.join(words))
    
    aae_letters = _get_letters(aae_words)
    sae_letters = _get_letters(sae_words)
    if aae_letters and sae_letters:
        similarity = difflib.SequenceMatcher(None, aae_letters, sae_letters).ratio()
        if similarity >= orthographic_threshold:
            return Q4Stratum.ORTHOGRAPHIC
    if all(word in function_words for word in aae_words + sae_words):
        return Q4Stratum.FUNCTION_WORD
    return Q4Stratum.LEXICAL
    

def _get_ending_class(
    text: str,
    terminal_punctuation_pattern: re.Pattern = text_rules.TERMINAL_PUNCTUATION_PATTERN,
) -> str:
    ending_class = text.rstrip()[-1:]
    if terminal_punctuation_pattern.search(ending_class):
        return ending_class 
    if ending_class == '':
        return 'none' # why hardcode this
    return 'other'


def _get_case_style(
    text: str
) -> str:
    alpha_chars = [char for char in text if char.isalpha()]
    if not alpha_chars:
        return 'none'
    if all(char.islower() for char in alpha_chars):
        return 'lower'
    if all(char.isupper() for char in alpha_chars):
        return 'upper'
    return 'mixed'


def _get_num_handle_tags(
    text: str,
) -> int:
    num_handle_tags = sum(token.startswith(('#', '@')) for token in text.split())
    return num_handle_tags


def _get_stratum(
    aae_text: str,
    sae_text: str,
    edit_counts: Counter[Q4Stratum],
) -> Q4Stratum:
    if aae_text == sae_text:
        return Q4Stratum.IDENTICAL
    if not edit_counts:
        return Q4Stratum.FORMAT_ONLY
    if edit_counts[Q4Stratum.LEXICAL]:
        return Q4Stratum.LEXICAL
    if edit_counts[Q4Stratum.FUNCTION_WORD]:
        return Q4Stratum.FUNCTION_WORD
    return Q4Stratum.ORTHOGRAPHIC


def compute_pair_edit_data(
    aae_text: str,
    sae_text: str,
    orthographic_threshold: float,
    function_words: tuple[str, ...],
) -> PairEditData:
    aae_words = _normalize_words(aae_text)
    sae_words = _normalize_words(sae_text)

    matcher = difflib.SequenceMatcher(None, aae_words, sae_words, autojunk=False)
    edit_counts = Counter(
        _classify_edit(aae_words[aae_start:aae_end], sae_words[sae_start:sae_end], orthographic_threshold, function_words)
        for tag, aae_start, aae_end, sae_start, sae_end in matcher.get_opcodes()
        if tag != 'equal'
    )
    
    has_asterisk = '*' in aae_text or '*' in sae_text
    pair_edit_data = PairEditData(
        stratum=_get_stratum(aae_text, sae_text, edit_counts),
        num_orthographic_edits=edit_counts[Q4Stratum.ORTHOGRAPHIC],
        num_function_word_edits=edit_counts[Q4Stratum.FUNCTION_WORD],
        num_lexical_edits=edit_counts[Q4Stratum.LEXICAL],
        ending_changed=_get_ending_class(aae_text) != _get_ending_class(sae_text),
        case_changed=_get_case_style(aae_text) != _get_case_style(sae_text),
        tag_count_changed=_get_num_handle_tags(aae_text) != _get_num_handle_tags(sae_text),
        has_asterisk=has_asterisk,
        num_words_aae=len(aae_text.split()), # replace with more intuitive arg
        num_words_sae=len(sae_text.split()), # replace with more intuitive arg
    )
    return pair_edit_data


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


@functools.cache
def _load_sentencizer() -> spacy.Language:
    sentencizer = spacy.blank('en')
    sentencizer.add_pipe('sentencizer')
    return sentencizer


def sentencize(text: str) -> list[str]:
    sentencizer = _load_sentencizer()
    sentences = [sentence.text for sentence in sentencizer(text).sents]
    return sentences
