import re

from src.core.utils import AaeFeature


### Regex patterns drafted with LLM assistance and and manually checked against corpus samples
# model input cleaning
MENTION_PLACEHOLDER: str = '@user'
CLEANUP_REPLACEMENT_MAP: dict[int, int | str | None] = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'", '\u2032': "'",  # curly single quotes, prime
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"', '\u2033': '"',  # curly double quotes, double prime
    '\u2013': '-', '\u2014': '-', '\u2212': '-',  # en dash, em dash, minus sign
    '\u2026': '...',  # ellipsis
    '\u00a0': ' ',  # non-breaking space
    '\u200b': None, '\u200c': None, '\u200d': None, '\u2060': None, '\ufeff': None,  # zero-width characters, byte-order mark
})


# pair filtering (applied after splits)
SAE_ARTIFACT_PATTERN: re.Pattern = re.compile(
    r'standard american english translation', re.IGNORECASE
)
TERMINAL_PUNCTUATION_PATTERN: re.Pattern = re.compile(r'\s*[.!?]+$')


# feature probe patterns
_HABITUAL_SUBJECTS: str = (
    r"(?:"
    r"i|you|he|she|it|we|they|yall|y'all|people|everybody|somebody"
    r")"
)
AAE_FEATURE_PATTERNS: dict[AaeFeature, re.Pattern] = {
    AaeFeature.HABITUAL_BE: re.compile(
        rf"(?i)\b{_HABITUAL_SUBJECTS}\s+be\s+(?:\w+ing|mad|chillin'?|trippin'?|wildin'?|at\s+the|in\s+the|on\s+that)\b",
    ),
    AaeFeature.COMPLETIVE_DONE: re.compile(
        r"(?i)\b(?:done)\s+(?:told|said|seen|went|ate|took|gave|left|did|been|made|got|had|heard|broke|forgot|\w+ed)\b",
    ),
    AaeFeature.NEGATIVE_CONCORD: re.compile(
        r"(?i)\b(?:ain'?t|don'?t|didn'?t|won'?t|can'?t|couldn'?t|never|not)\b(?:\s+\w+){0,4}\s+\b(?:nothing|nobody|nowhere|none|no)\b",
    ),
    AaeFeature.UNCONJUGATED_AINT: re.compile(
        r"(?i)\bain'?t\b(?:\s+\w+){1,3}",
    ),
}
QUOTATIVE_BE_LIKE_PATTERN: re.Pattern = re.compile(
    rf"(?i)\b{_HABITUAL_SUBJECTS}\s+be\s+like\b",
)


# 02: text annotations
ACTION_MARKER_PATTERN: re.Pattern = re.compile(
    r"\*[a-z']{3,}\*",
)
MASKED_WORD_PATTERN: re.Pattern = re.compile(
    r"[a-z']+(?:\*+[a-z']+)+\**|[a-z]\*+|[a-z']{2}\*{2,}",
)


# Q4
TEXT_UNIT_PATTERN: re.Pattern = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']") # LLM-generated
NON_ALPHANUMERIC_PATTERN: re.Pattern = re.compile(r'[^\da-z]')
REPEATED_LETTER_PATTERN: re.Pattern = re.compile(r'(.)\1{2,}')
VOWEL_PATTERN: re.Pattern = re.compile(r'[aeiouy]') # includes y
VOWELS: str = 'aeiou'

Q4_NORMALIZATION_REPLACEMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'&+', ' and '),
    (re.escape(MENTION_PLACEHOLDER), ''),
    (r'(?<=\w\w)-+(?=\w\w)', ' '),
    (r'(?<=\w)\.{2,}(?=[#@\w])', ' '),
    (r'(?<=[A-Za-z]{2})\.(?=[A-Za-z]{2})|(?<=\b[iI])\.(?=[A-Za-z]{2})', ' '),
)
Q4_NEGATION_PATTERN: re.Pattern = re.compile(
    r"(?:n't$|^(?:do|does|did|have|has|had|is|was|were|are|ca|wo|could|would|should|must|need|ai|might)nt$|^cannot$|^not$|^no$|^never$|^ain$|^nt$)"
)