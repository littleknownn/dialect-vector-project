import hashlib
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

import torch
import torch.nn as nn


type HiddenStateData = tuple[torch.Tensor, *tuple[Any, ...]]


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


class AaeFeature(StrEnum):
    NEGATIVE_CONCORD = 'negative_concord'
    HABITUAL_BE = 'habitual_be'
    COMPLETIVE_DONE = 'completive_done'
    UNCONJUGATED_AINT = 'unconjugated_aint'


class MetricType(StrEnum):
    LLR_SHIFT = 'llr_shift'
    PERPLEXITY = 'perplexity'


class DifferenceVector(StrEnum):
    OFFSET = ''
    AUTHORSHIP = '_aae_minus_aae_back'
    DIALECT = '_aae_back_minus_sae'
    NO_TYPOGRAPHY = '_no_typography'
    SAME_TERMINAL_PUNCTUATION = '_same_terminal_punctuation'
    NO_PROFANITY = '_no_profanity'

    @property
    def includes_aae_back(self) -> bool:
        return self in (
            DifferenceVector.AUTHORSHIP,
            DifferenceVector.DIALECT,
        )


class Q1ConsistencyLabel(StrEnum):
    CONSISTENT = 'consistent'
    PARTIAL = 'partial'
    WEAK = 'weak'
    POOLING_DEPENDENT = 'pooling_dependent'


class Q3EditCount(StrEnum):
    TOTAL = 'total'
    ORTHOGRAPHIC = 'orthographic'
    FUNCTION_WORD = 'function_word'
    LEXICAL = 'lexical'
    MECHANICS = 'mechanics'
    PROBE_COUNT = 'probe_count'


class Q4EditVerdict(StrEnum):
    STRONGER = 'stronger'
    WEAKER = 'weaker'
    NOT_DIFFERENT = 'not_different'


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


class AucSummary(TypedDict):
    layer: int
    pooling_type: PoolingType
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricType
    auc_pos: float
    auc_neg: float


class FdrSummary(TypedDict):
    layer: int
    pooling_type: PoolingType
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricType
    alpha: float
    p_value_fdr: float


class BehavioralSummary(TypedDict):
    auc: list[AucSummary]
    fdr: list[FdrSummary]


class SweepSummary(TypedDict):
    layer: int
    pooling_type: PoolingType
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricType
    alpha: float
    value: float
    p_value: float


class H1Results(TypedDict):
    supervised_vector_norm: float
    floors: dict[str, list[float]]
    pair_distances: dict[str, list[float]]
    null_distances: list[float]
    projection_energies: dict[str, list[float]]


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


class Q3CountResults(TypedDict):
    spearman_ci: CIStats
    spearman_p_value: float
    spearman_p_value_fdr: float | None
    partial_spearman_ci: CIStats
    projection_mean_ci_by_bin: dict[int, CIStats]


class Q3Results(TypedDict):
    corpus_name: str
    pooling_type: PoolingType
    count_results: dict[Q3EditCount, Q3CountResults]


class Q4Edit(TypedDict):
    kind: Q4EditKind
    distance: int
    size: int


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


class FreezeRecord(TypedDict):
    git_commit: setattr
    experiment_config: dict[str, Any]
    processed_data_hashes: dict[str, str]


class GeometricMetricsPair(NamedTuple):
    principal_angles: float
    chordal_distance: float


class SweepKey(NamedTuple):
    layer: int
    pooling_type: PoolingType
    rank: int
    estimator_name: str
    corpus_name: str
    metric: MetricType


class Q4TaggerInputs(NamedTuple):
    function_words: frozenset[str]
    grammar_words: frozenset[str]
    copula_words: frozenset[str]
    profanities: frozenset[str]
    respelling_threshold: float
    single_letter_respelling_threshold: float


class EstimatorSteering:
    def __init__(
        self,
        estimator_matrix: torch.Tensor,
        layer_module: int,
        alpha: float,
        act_add: bool = False,
    ):
        self.estimator_matrix = estimator_matrix
        self.layer_module = layer_module
        self.alpha = alpha
        self.act_add = act_add
        self.forward_hook = None

        if act_add:
            assert estimator_matrix.shape[1] == 1, 'ActAdd Steering requires a Rank-1 Estimator'

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

        if self.act_add:
            steered_hidden_states = hidden_states + self.alpha * self.estimator_matrix.T
        else:
            subspace_projection = (hidden_states @ self.estimator_matrix) @ self.estimator_matrix.T
            steered_hidden_states = hidden_states + (self.alpha * subspace_projection)

        if is_output_wrapped:
            return (steered_hidden_states, *previous_layer_outputs[1:])
        return steered_hidden_states


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


def compute_file_hashes(
    in_dir: Path,
) -> dict[str, str]:
    file_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(in_dir.iterdir())
        if path.is_file()
    }
    return file_hashes


def load_boundary_char_indices_by_split(
    in_dir: Path,
) -> dict[str, list[int | None]]:
    boundary_keys = {
        SplitSuffixes.AAE: 'aae_idx', #
        SplitSuffixes.SAE: 'sae_idx', #
        SplitSuffixes.AAE_BACK: 'aae_back_idx', #
    }
    boundary_char_indices_by_split = {}
    for metadata_filepath in  in_dir.glob(f'*{SplitSuffixes.PAIR_METADATA}*.json'):
        corpus_name, split_suffix = metadata_filepath.stem.split(SplitSuffixes.PAIR_METADATA)
        pair_metadata = load_json(metadata_filepath)
        for dialect_suffix, boundary_key in boundary_keys.items():
            split_name = f'{corpus_name}{dialect_suffix}{split_suffix}'
            boundary_char_indices_by_split[split_name] = [
                metadata['boundary'][boundary_key]
                if metadata['boundary'] else None
                for metadata in pair_metadata
            ]
    return boundary_char_indices_by_split


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
