from collections import Counter
import difflib
import re
import string

import src.core.text_rules as text_rules
from src.core.utils import (
    PairEditData,
    Q4Stratum,
)
from src.core.config import (
    DatasetConfig,
)


_Q4_NORMALIZATION_REPLACEMENT_PATTERNS = (
    (r'&', ' and '),
    (re.escape(text_rules.MENTION_PLACEHOLDER), ''),
)


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


def _normalize_words(
    text: str,
    replacement_patterns: tuple[tuple[str, str], ...] = _Q4_NORMALIZATION_REPLACEMENT_PATTERNS,
) -> list[str]:
    for pattern, replacement in replacement_patterns:
        text = re.sub(pattern, replacement, text)
    stripped_words = [
        word.strip(string.punctuation.replace("'", ''))
        for word in text.lower().split()
    ]
    words = [word for word in stripped_words if word]
    return words


def _classify_edit(
    aae_words: list[str],
    sae_words: list[str],
    orthographic_threshold: float,
    function_words: frozenset[str],
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
) -> str:
    ending_class = text.rstrip()[-1:]
    if ending_class in {'.', '!', '?'}:
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
    function_words: frozenset[str],
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
        num_words_aae=len(aae_text.split()), #
        num_words_sae=len(sae_text.split()), #
    )
    return pair_edit_data