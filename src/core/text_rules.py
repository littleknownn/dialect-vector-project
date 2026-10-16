import re


### most regexes built with LLM calls and manually tested against datasets, but not extensively yet
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


# feature probes
HABITUAL_SUBJECTS: str = (
    r"(?:"
    r"i|you|he|she|it|we|they|yall|y'all|people|everybody|somebody"
    r")"
)
AAE_FEATURE_PATTERNS: dict[str, re.Pattern] = { # refine when able
    'negative_concord': re.compile(
        r"(?i)\b(?:ain'?t|don'?t|didn'?t|won'?t|can'?t|couldn'?t|never|not)\b(?:\s+\w+){0,4}\s+\b(?:nothing|nobody|nowhere|none|no)\b",
    ),
    'habitual_be': re.compile(
        rf"(?i)\b{HABITUAL_SUBJECTS}\s+be\s+(?:\w+ing|mad|chillin'?|trippin'?|wildin'?|at\s+the|in\s+the|on\s+that)\b",
    ),
    'completive_done': re.compile(
        r"(?i)\b(?:done)\s+(?:told|said|seen|went|ate|took|gave|left|did|been|made|got|had|heard|broke|forgot|\w+ed)\b",
    ),
    'unconjugated_aint': re.compile(
        r"(?i)\bain'?t\b(?:\s+\w+){1,3}",
    ),
}
UNMATCHED_FEATURE_KEY: str = 'other'


# tags
TAG_PATTERNS: dict[str, re.Pattern] = {
    'quotative_be_like': re.compile(
        rf"(?i)\b{HABITUAL_SUBJECTS}\s+be\s+like\b"
    ),
}

# Q4
Q4_NORMALIZATION_REPLACEMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'&', ' and '),
    (re.escape(MENTION_PLACEHOLDER), ''),
)
