from enum import (
    IntEnum,
    StrEnum,
)
from pathlib import Path
import json
from typing import (
    Any,
    Iterable,
    NamedTuple,
    TypedDict,
)


class ModelName(StrEnum):
    LLAMA_3B = 'llama_3b'
    LLAMA_8B = 'llama_8b'
    OLMO_7B = 'olmo_7b'
    GEMMA_2B = 'gemma_2b'


class PoolingType(StrEnum):
    LAST = 'last'
    MEAN = 'mean'
    BOUNDARY = 'boundary'

    @property
    def is_paired_only(self) -> bool:
        return self is PoolingType.BOUNDARY


class SplitSuffixes(StrEnum):
    AAE = '_aae'
    SAE = '_sae'
    AAE_BACK = '_aae_back'
    PAIR_METADATA = '_pair_metadata'
    SUPERVISED = '_supervised'
    TESTING = '_testing'
    RESERVE = '_reserve'
    STRIPPED = '_stripped'
    POOL = '_pool'
    PAIR_ANNOTATIONS = '_pair_annotations'
    SPEAKERS = '_speakers'


class RunMode(StrEnum):
    DEVELOPMENT = 'development'
    FREEZE = 'freeze'
    TESTING = 'testing'

    @property
    def split_suffix(self) -> SplitSuffixes:
        return (
            SplitSuffixes.TESTING
            if self is RunMode.TESTING
            else SplitSuffixes.SUPERVISED
        )


class AaeFeature(StrEnum):
    NEGATIVE_CONCORD = 'negative_concord'
    HABITUAL_BE = 'habitual_be'
    COMPLETIVE_DONE = 'completive_done'
    UNCONJUGATED_AINT = 'unconjugated_aint'


class DifferenceVector(StrEnum):
    DIALECT = ''
    AUTHORSHIP = '_aae_minus_aae_back'
    DIALECT_GEMINI = '_aae_back_minus_sae'
    NO_TYPOGRAPHY = '_no_typography'
    SAME_TERMINAL_PUNCTUATION = '_same_terminal_punctuation'
    NO_PROFANITY = '_no_profanity'

    @property
    def includes_aae_back(self) -> bool:
        return self in (
            DifferenceVector.AUTHORSHIP,
            DifferenceVector.DIALECT_GEMINI,
        )


class Q1ConsistencyLabel(StrEnum):
    CONSISTENT = 'consistent'
    PARTIAL = 'partial'
    WEAK = 'weak'
    POOLING_DEPENDENT = 'pooling_dependent'


class Q1Statistic(StrEnum):
    POSITIVE_PROJECTION_RATIO = 'positive_projection_ratio_ci'
    STANDARDIZED_PROJECTION_MEAN = 'standardized_projection_mean_ci'


class Q2TransferLabel(StrEnum):
    TRANSFER = 'transfer'
    SMALL = 'small'
    BOUNDED_NULL = 'bounded_null'
    INCONCLUSIVE = 'inconclusive'


class Q2Family(StrEnum):
    PRIMARY = 'primary'
    SECONDARY = 'secondary'
    DESCRIPTIVE = 'descriptive'


class Q3EditCount(StrEnum):
    TOTAL = 'total'
    ORTHOGRAPHIC = 'orthographic'
    CASE_AND_PUNCTUATION = 'case_and_punctuation'
    GRAMMATICAL = 'grammatical'
    LEXICAL = 'lexical'
    PROBE_COUNT = 'probe_count'


class Q4EditVerdict(StrEnum):
    STRONGER = 'stronger'
    WEAKER = 'weaker'
    NOT_DIFFERENT = 'not_different'
    TOO_RARE = 'too_rare'


class Q4RobustnessVariant(StrEnum):
    SINGLE_WORD_MARKERS = 'single_word_markers'
    AINT_EXCLUDED = 'aint_excluded'
    DISTANCE_FROM_TWO = 'distance_from_two'
    NO_TYPOGRAPHY_VECTOR = 'no_typography_vector'
    SAME_TERMINAL_PUNCTUATION_VECTOR = 'same_terminal_punctuation_vector'
    NO_PROFANITY_VECTOR = 'no_profanity_vector'
    LEAVE_ONE_OUT = 'leave_one_out'


class Q4Stratum(StrEnum):
    IDENTICAL = 's0_identical'
    FORMAT_ONLY = 's1_format_only'
    ORTHOGRAPHIC = 's2_orthographic'
    FUNCTION_WORD = 's3_function_word'
    LEXICAL = 's4_lexical'


class Q4EditKind(StrEnum):
    ORTHOGRAPHIC = 'orthographic'
    COPULA_ADDED = 'copula_added'
    FUNCTION_WORD = 'function_word'
    FILTER_FLAGGED_WORD = 'filter_flagged_word'
    GRAMMAR_WORD = 'grammar_word'
    ADDED_OR_DROPPED = 'added_or_dropped'
    SHORTHAND_FUNCTION_WORDS = 'shorthand_function_words'
    SHORTHAND_EXPANDED = 'shorthand_expanded'
    CONTENT_CHANGE = 'content_change'
    CAPITALIZATION = 'capitalization'
    PUNCTUATION = 'punctuation'


class Q4CurveFunction(StrEnum):
    LOG = 'log'
    CEILING = 'ceiling'
    STRAIGHT = 'straight'
    FREE_SHAPE = 'free_shape'


class SeedOffset(IntEnum):
    BOOTSTRAP = 1
    SIGN_FLIP = 2
    SHUFFLED_PAIRING = 3
    HALF_SPLITS = 4
    Q2_MATCHING = 5
    Q2_BOOTSTRAP = 6
    Q2_LABEL_SHUFFLE = 7
    Q3_Q4_BOOTSTRAP = 8
    Q4_HELD_OUT_FOLDS = 9
    Q4_SHUFFLED_PROJECTIONS = 15


class PairBoundary(TypedDict):
    aae_idx: int
    sae_idx: int
    aae_back_idx: int | None
    is_exact: bool
    is_aae_back_shared: bool


class PairMetadata(TypedDict):
    row_idx: int
    is_sae_typography_normalized: bool
    is_aae_back_typography_normalized: bool
    is_aae_back_valid: bool
    boundary: PairBoundary | None


class LinesProcessingCounts(TypedDict):
    num_cleaned: int
    num_out_of_bounds: int
    num_duplicates: int
    num_kept: int


class PairedCorpusProcessingCounts(TypedDict):
    num_rows: int
    num_pairs_kept: int
    num_aae_back_invalid: int
    num_dropped_by_split: dict[str, int]


class ProcessingLog(TypedDict):
    paired_corpus_counts: dict[str, PairedCorpusProcessingCounts]
    single_corpus_counts: dict[str, LinesProcessingCounts]
    split_sizes: dict[str, int]


class CIStats(TypedDict):
    estimate: float
    ci_lower: float
    ci_upper: float


class Q1Results(TypedDict):
    corpus_name: str
    label: Q1ConsistencyLabel
    vector: DifferenceVector
    layer: int
    pooling_type: PoolingType
    num_pairs: int
    num_excluded_pairs: int
    
    projection_mean_ci: CIStats
    standardized_projection_mean_ci: CIStats
    positive_projection_ratio_ci: CIStats
    ending_same_projection_mean_ci: CIStats
    ending_changed_projection_mean_ci: CIStats

    activation_difference_energy: float
    projection_energy: float
    content_matching_gain: float

    sign_flip_p_value: float
    sign_flip_p_value_fdr: float | None

    identical_pairs_median_abs_projection: float
    other_pairs_median_abs_projection: float

    shuffled_positive_projection_ratio: float
    shuffled_standardized_projection_mean: float

    num_identical_pairs: int
    nonidentical_pairs_positive_projection_ratio_ci: CIStats
    nonidentical_pairs_standardized_projection_mean_ci: CIStats
    difference_vector_cosine: float


class Q2Results(TypedDict):
    corpus_name: str
    layer: int
    pooling_type: PoolingType
    vector: DifferenceVector
    positive_probes: list[AaeFeature]
    excludes_filter_flagged_words: bool
    family: Q2Family
    num_positives: int
    num_dropped_positives: int
    num_short_positives: int
    num_probes_after_cut_lines: int
    mean_num_negatives: float
    num_unique_negatives: int
    auroc_ci: CIStats
    standardized_mean_difference_ci: CIStats
    null_p_value: float | None
    null_p_value_fdr: float | None
    null_band: list[float] | None
    baseline_aurocs: dict[str, float]
    min_detectable_auroc: float
    label: Q2TransferLabel | None


class Q3CountResults(TypedDict):
    spearman_ci: CIStats
    spearman_p_value: float
    spearman_p_value_fdr: float | None
    partial_spearman_ci: CIStats
    partial_spearman_p_value: float
    partial_spearman_p_value_fdr: float | None
    projection_mean_ci_by_bin: dict[int, CIStats]
    bin_edges: list[float]
    has_zero_bin: bool


class Q3Results(TypedDict):
    corpus_name: str
    pooling_type: PoolingType
    count_results: dict[Q3EditCount, Q3CountResults]


class Q4Edit(TypedDict):
    kind: Q4EditKind
    distance: int
    size: int
    kind_without_aint: Q4EditKind
    kind_by_removed_marker_word: dict[str, Q4EditKind]


class Q4CurveResults(TypedDict):
    curve: Q4CurveFunction
    r_squared: float
    held_out_r_squared: float
    distance_decay_exponent: CIStats | None
    size_exponent: CIStats
    weights: dict[Q4EditKind, CIStats]
    mean_marginal_effects_by_distance: dict[int, dict[Q4EditKind, float]]
    shares: dict[Q4EditKind, CIStats]
    format_flags_share: float
    pair_controls_share: float
    unexplained_share: float
    calibration: list[list[float]]
    mean_residual_by_kind: dict[Q4EditKind, float]
    shuffled_r_squared: float
    weight_p_values: dict[Q4EditKind, float] | None
    weight_p_values_fdr: dict[Q4EditKind, float] | None
    num_failed_fits: int


class Q4Results(TypedDict):
    corpus_name: str
    pooling_type: PoolingType
    num_pairs_by_stratum: dict[Q4Stratum, int]
    mean_projection_by_stratum: dict[Q4Stratum, CIStats]

    num_edits_by_kind: dict[Q4EditKind, int]
    curve_results: dict[Q4CurveFunction, Q4CurveResults]
    ceiling_minus_log_held_out_r_squared: CIStats
    verdicts: dict[Q4EditKind, Q4EditVerdict] | None
    single_edit_weights: dict[Q4EditKind, CIStats] | None


class Q4RobustnessResults(TypedDict):
    corpus_name: str
    pooling_type: PoolingType
    variant: Q4RobustnessVariant
    removed_word: str | None
    curve: Q4CurveFunction
    num_marker_edits: int
    weights: dict[Q4EditKind, CIStats]


class PairControls(TypedDict):
    aae_filter_flagged: bool
    sae_filter_flagged: bool
    log_length_ratio: float


class PairEditData(TypedDict):
    stratum: Q4Stratum
    num_orthographic_edits: int
    num_function_word_edits: int
    num_lexical_edits: int
    ending_changed: bool
    case_changed: bool
    tag_count_changed: bool
    has_asterisk: bool
    num_words_aae: int
    num_words_sae: int


class LineAnnotations(TypedDict):
    num_words: int
    fragmentation: float
    is_truncated: bool
    has_mention: bool
    has_masked_word: bool
    has_quotative_be_like: bool
    has_terminal_punctuation: bool
    probes: list[AaeFeature]
    profanity_matches: list[str]


class TextPairAnnotations(TypedDict):
    edit_data_by_threshold: dict[str, PairEditData]
    edits: list[Q4Edit]
    controls: PairControls
    num_aae_features: int


class PairAnnotations(TypedDict):
    line: TextPairAnnotations
    context: TextPairAnnotations | None


class ExtractionLog(TypedDict):
    model_hf_path: str
    torch_version: str
    transformers_version: str
    experiment_config: dict[str, Any]
    processed_data_hashes: dict[str, str]
    num_lines_by_split: dict[str, int]
    num_truncated_lines_by_split: dict[str, int]


class FrozenInputHashes(NamedTuple):
    text_annotations: dict[str, str]
    estimators: dict[str, str]
    profanity_list: str


class FreezeRecord(TypedDict):
    source_code_hashes: dict[str, str]
    experiment_config: dict[str, Any]
    processed_data_hashes: dict[str, str]
    frozen_input_hashes: dict[str, dict[str, str] | str]
    declared_model_specs: dict[str, dict[str, Any]]


class Q4TaggerInputs(NamedTuple):
    function_words: frozenset[str]
    grammar_words: frozenset[str]
    copula_words: frozenset[str]
    profanities: frozenset[str]
    respelling_threshold: float
    single_letter_respelling_threshold: float
    aint_words: frozenset[str]


def save_jsonl(
    text_iter: Iterable[str],
    out_dir: Path,
    out_filename: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename += '.jsonl'
    out_path = out_dir / out_filename
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for text in text_iter:
            json_object = {'text': text}
            out_file.write(f'{json.dumps(json_object, ensure_ascii=False)}\n')


def load_jsonl(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as corpus_file:
        lines = [json.loads(line)['text'] for line in corpus_file]
    return lines


def save_json(
    data: dict | list,
    out_dir: Path,
    out_filename: str,
    json_indent: int = 4,
) -> None:
    out_filepath = out_dir / f'{out_filename}.json'
    with open(out_filepath, 'w') as file:
        json.dump(data, file, indent=json_indent)


def load_json(
    filepath: Path,
) -> dict | list:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)
