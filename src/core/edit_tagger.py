import difflib
import functools
import re

import src.core.text_rules as text_rules
import src.core.text_utils as text_utils
from src.core.utils import (
    Q4Edit,
    Q4EditKind,
    Q4TaggerInputs,
)


_MIN_ABBREVIATION_LEN: int = 2
_MAX_ABBREVIATION_VOWELS: int = 1
_MIN_CLIPPED_WORD_LEN: int = 3
_MAX_CLIPPED_PREFIX_LEN: int = 2
_MIN_CONSONANTS_LEN: int = 2
_MAX_SINGLE_LETTER_LEN: int = 1
_MAX_SHORTHAND_WORD_LEN: int = 2
_MAX_EXPANDED_WORD_LEN: int = 5


def _get_letters(
    words: tuple[str, ...],
    non_alphanumeric_pattern: re.Pattern = text_rules.NON_ALPHANUMERIC_PATTERN,
    repeated_letter_pattern: re.Pattern = text_rules.REPEATED_LETTER_PATTERN,
) -> str:
    letters = non_alphanumeric_pattern.sub('', ''.join(words))
    letters = repeated_letter_pattern.sub(r'\1', letters)
    return letters


def _get_similarity_score(
    text_a: str,
    text_b: str,
) -> float:
    if not (text_a and text_b):
        return 0.0
    similarity_score = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    return similarity_score


def _is_abbreviation(
    short: str,
    long: str,
    min_len: int = _MIN_ABBREVIATION_LEN,
    max_vowels: int = _MAX_ABBREVIATION_VOWELS,
    vowels: str = text_rules.VOWELS,
) -> bool:
    if (
        len(short) < min_len
        or len(short) >= len(long)
        or short[0] != long[0]
        or sum(char in vowels for char in short) > max_vowels
    ):
        return False
    long_chars = iter(long)
    is_abbreviation = all(char in long_chars for char in short)
    return is_abbreviation


def _count_negations(
    words: tuple[str, ...]
) -> int:
    num_negations = sum(
        bool(text_rules.Q4_NEGATION_PATTERN.search(word))
        for word in words
    )
    return num_negations


@functools.cache
def _compute_respelling_score(
    aae_words: tuple[str, ...],
    sae_words: tuple[str, ...],
    min_clipped_word_len: int = _MIN_CLIPPED_WORD_LEN,
    max_clipped_prefix_len: int = _MAX_CLIPPED_PREFIX_LEN,
    min_consonants_len: int = _MIN_CONSONANTS_LEN,
    vowel_pattern: re.Pattern = text_rules.VOWEL_PATTERN,
) -> float:
    aae_letters, sae_letters = _get_letters(aae_words), _get_letters(sae_words)
    if not (aae_letters and sae_letters):
        return 0.0
    if len(sae_words) == 1 and _is_abbreviation(aae_letters, sae_letters):
        return 1.0
    if len(aae_words) == len(sae_words) == 1:
        aae_word, sae_word = aae_words[0], sae_words[0]
        aae_consonants, sae_consonants = vowel_pattern.sub('', aae_word), vowel_pattern.sub('', sae_word) 
        is_clipped = (
            len(aae_word) >= min_clipped_word_len
            and any(
                aae_word == sae_word[prefix_len:]
                for prefix_len in range(1, max_clipped_prefix_len + 1)
            )
        )
        are_consonants_identical = (
            len(aae_consonants) >= min_consonants_len
            and aae_consonants == sae_consonants
            and aae_word[0] == sae_word[0]
        )
        if is_clipped or are_consonants_identical:
            return 1.0
    respelling_score = _get_similarity_score(aae_letters, sae_letters)
    return respelling_score


def _is_dropped_flag(
    word: str,
    sae_words: tuple[str, ...],
    tagger_inputs: Q4TaggerInputs,
) -> bool:
    is_kept_respelled = any(
        sae_word in tagger_inputs.profanities
        and _get_similarity_score(word, sae_word) >= tagger_inputs.respelling_threshold
        for sae_word in sae_words
    )
    is_dropped_flag = (
        word in tagger_inputs.profanities
        and word not in sae_words
        and not is_kept_respelled
    )
    return is_dropped_flag


def _is_respelling(
    aae_words: tuple[str, ...],
    sae_words: tuple[str, ...],
    tagger_inputs: Q4TaggerInputs,
    max_single_letter_len: int = _MAX_SINGLE_LETTER_LEN,
) -> bool:
    if (
        not (aae_words and sae_words)
        or aae_words == sae_words
        or _count_negations(aae_words) != _count_negations(sae_words)
        or any(
            word in tagger_inputs.grammar_words
            for word in aae_words
        )
        or any(
            _is_dropped_flag(word, sae_words, tagger_inputs)
            for word in aae_words
        )
        or (
            len(aae_words) == len(sae_words)
            and all(
                aae_word in tagger_inputs.function_words
                and sae_word in tagger_inputs.function_words
                and not sae_word.startswith(aae_word)
                for aae_word, sae_word in zip(aae_words, sae_words)
            )
        )
    ):
        return False
    
    is_single_letter = (
        min(len(_get_letters(aae_words)), len(_get_letters(sae_words)))
        <= max_single_letter_len
    )
    threshold = (
        tagger_inputs.single_letter_respelling_threshold
        if is_single_letter
        else tagger_inputs.respelling_threshold
    )
    is_respelling = _compute_respelling_score(aae_words, sae_words) >= threshold
    return is_respelling


def _is_shorthand_function_words(
    aae_words: tuple[str, ...],
    sae_words: tuple[str, ...],
    tagger_inputs: Q4TaggerInputs,
    max_word_len: int = _MAX_SHORTHAND_WORD_LEN,
) -> bool:
    if (
        not aae_words
        or len(aae_words) != len(sae_words)
        or len(aae_words) > max_word_len
    ):
        return False
    is_shorthand_function_words = (
        all(
            len(word) <= max_word_len
            and word not in tagger_inputs.function_words
            for word in aae_words
        )
        and all(
            word in tagger_inputs.function_words
            for word in sae_words
        )
    )
    return is_shorthand_function_words


def _get_edit_kind(
    aae_words: tuple[str, ...],
    sae_words: tuple[str, ...],
    tagger_inputs: Q4TaggerInputs,
    max_expanded_word_len: int = _MAX_EXPANDED_WORD_LEN,
) -> Q4EditKind:
    is_one_to_many = (
        len(aae_words) == 1
        and len(sae_words) >= 2
    )
    aae_block = ' '.join(aae_words)
    if _is_shorthand_function_words(aae_words, sae_words, tagger_inputs):
        return Q4EditKind.SHORTHAND_FUNCTION_WORDS
    if (
        is_one_to_many
        and aae_words[0] == ''.join(word[0] for word in sae_words)
    ):
        return Q4EditKind.SHORTHAND_EXPANDED
    if _is_respelling(aae_words, sae_words, tagger_inputs):
        return Q4EditKind.ORTHOGRAPHIC
    if (
        not aae_words
        and all(word in tagger_inputs.copula_words for word in sae_words)
    ):
        return Q4EditKind.COPULA_ADDED
    if (
        text_utils.get_matched_words(aae_block, tagger_inputs.profanities)
        or text_utils.has_masked_word(aae_block)
    ):
        return Q4EditKind.FILTER_FLAGGED_WORD
    if any(
        word in tagger_inputs.grammar_words
        for word in aae_words
    ):
        return Q4EditKind.GRAMMAR_WORD
    if all(
        word in tagger_inputs.function_words
        for word in aae_words + sae_words
    ):
        return Q4EditKind.FUNCTION_WORD
    if not (aae_words and sae_words):
        return Q4EditKind.ADDED_OR_DROPPED
    if (
        is_one_to_many
        and len(aae_words[0]) <= max_expanded_word_len
        and aae_words[0][0] == sae_words[0][0]
    ):
        return Q4EditKind.SHORTHAND_EXPANDED
    return Q4EditKind.CONTENT_CHANGE


def _compute_word_edits(
    aae_words: list[str],
    sae_words: list[str],
    tagger_inputs: Q4TaggerInputs,
) -> list[Q4Edit]:
    matcher = difflib.SequenceMatcher(None, aae_words, sae_words, autojunk=False)
    word_edits = [
        Q4Edit(
            kind=_get_edit_kind(
                tuple(aae_words[aae_start:aae_end]),
                tuple(sae_words[sae_start:sae_end]),
                tagger_inputs,
            ),
            distance=max(len(aae_words) - aae_end, 1),
            size=max(aae_end - aae_start, sae_end - sae_start),
        )
        for tag, aae_start, aae_end, sae_start, sae_end in matcher.get_opcodes()
        if tag != 'equal'
    ]
    return word_edits


def _compute_case_and_punctuation_edits(
    aae_text: str,
    sae_text: str,
    text_unit_pattern: re.Pattern,
) -> list[Q4Edit]:
    is_word = lambda text_unit: (
        text_unit[0].isalnum() or text_unit[0] == "'"
    )
    aae_text_units = text_unit_pattern.findall(aae_text)
    sae_text_units = text_unit_pattern.findall(sae_text) 
    num_words_after = [
        sum(map(is_word, aae_text_units[unit_idx:]))
        for unit_idx in range(len(aae_text_units) + 1)
    ]
    matcher = difflib.SequenceMatcher(
        None,
        [text_unit.lower() for text_unit in aae_text_units],
        [text_unit.lower() for text_unit in sae_text_units],
        autojunk=False,
    )

    case_and_punctuation_edits = []
    for tag, aae_start, aae_end, sae_start, sae_end in matcher.get_opcodes():
        if tag == 'equal':
            case_and_punctuation_edits.extend([
                Q4Edit(
                    kind=Q4EditKind.CAPITALIZATION,
                    distance=max(num_words_after[aae_idx + 1], 1),
                    size=1,
                )
                for aae_idx, sae_idx
                in zip(
                    range(aae_start, aae_end),
                    range(sae_start, sae_end)
                )
                if aae_text_units[aae_idx] != sae_text_units[sae_idx]
            ])
        elif not any(map(
            is_word,
            aae_text_units[aae_start:aae_end] + sae_text_units[sae_start:sae_end],
        )):
            case_and_punctuation_edits.append(Q4Edit(
                kind=Q4EditKind.PUNCTUATION,
                distance=max(num_words_after[aae_end], 1),
                size=1,
            ))

    return case_and_punctuation_edits


def compute_pair_edits(
    aae_text: str,
    sae_text: str,
    tagger_inputs: Q4TaggerInputs,
    text_unit_pattern: re.Pattern = text_rules.TEXT_UNIT_PATTERN,
) -> list[Q4Edit]:
    aae_words = text_utils.normalize_words(aae_text)
    sae_words = text_utils.normalize_words(sae_text)
    pair_edits = (
        _compute_word_edits(aae_words, sae_words, tagger_inputs)
        + _compute_case_and_punctuation_edits(aae_text, sae_text, text_unit_pattern)
    )
    return pair_edits
