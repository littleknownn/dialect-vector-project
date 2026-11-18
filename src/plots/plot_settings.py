from src.core.utils import (
    AaeFeature,
    DifferenceVector,
    PoolingType,
    Q3EditCount,
    Q4CurveFunction,
    Q4EditKind,
    Q4RobustnessVariant,
    Q4Stratum,
)


DEFAULT_FIGSIZE = (5, 3)
DEFAULT_LEGEND_POS = (1.05, 1)
FIGURE_FORMATS = ('pdf', 'png')

### LLM-Generated LaTeX Paper Figure Style Setup ###
# Q1
FOREST_WIDTH = 0.25
HISTOGRAM_ALPHA = 0.6
HEADROOM_SCALAR = 1.3
SUMMARY_TEXT_POS = (0.02, 0.98)

# Q2
CHANCE_AUROC = 0.5
NULL_BAND_KWARGS = {'color': 'gray', 'alpha': 0.15, 'linewidth': 0}
MDA_MARKER = '|'
BELOW_AXES_LEGEND_KWARGS = {'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.32), 'ncol': 3}
REMOVED_CORPUS_NAMES = ('aave_lyrics', 'aave_social_media', 'twitter_aae')

# Q3
ZERO_REFERENCE = 0.0
Q3_MIN_COEF = 0.10

# Q4
UNIT_WEIGHT = 1.0
DISTANCE_MARKERS = ('o', 's')
ROBUSTNESS_TICK_STEP = 0.5

# spacing
CM_PER_INCH = 2.54
COLUMN_WIDTH = 7.7 / CM_PER_INCH
TEXT_WIDTH = 16.0 / CM_PER_INCH

# markers
POOLING_COLORS = {
    PoolingType.BOUNDARY: '#0072B2',
    PoolingType.MEAN: '#E69F00',
    PoolingType.LAST: '#009E73',
}
CURVE_MARKERS = {
    Q4CurveFunction.LOG: 'o',
    Q4CurveFunction.CEILING: 's',
    Q4CurveFunction.STRAIGHT: '^',
    Q4CurveFunction.FREE_SHAPE: 'D',
}

SECONDARY_ALPHA = 0.4
REFERENCE_LINE_KWARGS = {'color': 'black', 'linewidth': 0.6, 'alpha': 0.5}
THRESHOLD_LINE_KWARGS = {'color': 'gray', 'linewidth': 0.6, 'linestyle': '--'}

# labels
PROJECTION_LABEL = 'Projection $p_i$ on $A$'
MEAN_PROJECTION_LABEL = 'Mean projection $p_i$ on $\\hat{A}$'

DIFFERENCE_VECTOR_LABELS = {
    DifferenceVector.DIALECT: '$A$',
    DifferenceVector.AUTHORSHIP: '$A_1$ (AAE $-$ AAE$_\\mathrm{back}$)',
    DifferenceVector.DIALECT_GEMINI: '$A_2$ (AAE$_\\mathrm{back}$ $-$ SAE)',
    DifferenceVector.NO_TYPOGRAPHY: '$A$, no typography',
    DifferenceVector.SAME_TERMINAL_PUNCTUATION: '$A$, same ending',
    DifferenceVector.NO_PROFANITY: '$A$, no filtered words',
}
EDIT_KIND_LABELS = {
    Q4EditKind.ORTHOGRAPHIC: 'Respelling',
    Q4EditKind.COPULA_ADDED: 'Copula added',
    Q4EditKind.FUNCTION_WORD: 'Function word',
    Q4EditKind.FILTER_FLAGGED_WORD: 'Filter-flagged word',
    Q4EditKind.GRAMMAR_WORD: 'Marker word',
    Q4EditKind.ADDED_OR_DROPPED: 'Added or dropped',
    Q4EditKind.SHORTHAND_FUNCTION_WORDS: 'Shorthand function word',
    Q4EditKind.SHORTHAND_EXPANDED: 'Shorthand expanded',
    Q4EditKind.CONTENT_CHANGE: 'Content change',
    Q4EditKind.CAPITALIZATION: 'Capitalization',
    Q4EditKind.PUNCTUATION: 'Punctuation',
}
STRATUM_LABELS = {
    Q4Stratum.IDENTICAL: 'Identical',
    Q4Stratum.FORMAT_ONLY: 'Format only',
    Q4Stratum.ORTHOGRAPHIC: 'Respelling',
    Q4Stratum.FUNCTION_WORD: 'Function word',
    Q4Stratum.LEXICAL: 'Lexical',
}
EDIT_COUNT_LABELS = {
    Q3EditCount.TOTAL: 'All edits',
    Q3EditCount.ORTHOGRAPHIC: 'Respellings',
    Q3EditCount.GRAMMATICAL: 'Grammatical edits',
    Q3EditCount.LEXICAL: 'Lexical edits',
    Q3EditCount.CASE_AND_PUNCTUATION: 'Capitalization and punctuation',
    Q3EditCount.PROBE_COUNT: 'Feature detector count',
}

CORPUS_LABELS = {
    'coraal_merged': 'CORAAL',
    'real_aae': 'REAL-AAE testing',
    'twitter_aae': 'TwitterAAE',
    'aave_lyrics': 'AAVE lyrics',
    'aave_social_media': 'AAVE social media',
}
AUROC_LABEL = 'Matched AUROC'
PROBE_SET_LABELS = {
    tuple(AaeFeature): 'Any feature',
    tuple(probe for probe in AaeFeature if probe != AaeFeature.UNCONJUGATED_AINT): "Any feature except ain't",
    (AaeFeature.NEGATIVE_CONCORD,): 'Negative concord',
    (AaeFeature.UNCONJUGATED_AINT,): "Ain't",
}
BASELINE_LABELS = {
    'uppercase_share': 'Uppercase share',
    'profanity': 'Filter-flagged word',
}
BASELINE_COLORS = {  # <<< ADD
    'uppercase_share': '#D55E00',
    'profanity': '#CC79A7',
}
CURVE_LABELS = {
    Q4CurveFunction.LOG: 'Log curve',
    Q4CurveFunction.CEILING: 'Ceiling curve',
    Q4CurveFunction.STRAIGHT: 'Straight line',
    Q4CurveFunction.FREE_SHAPE: 'Free shape',
}
ROBUSTNESS_VARIANT_LABELS = {
    Q4RobustnessVariant.SINGLE_WORD_MARKERS: 'Single-word markers only',
    Q4RobustnessVariant.AINT_EXCLUDED: "Ain't excluded",
    Q4RobustnessVariant.DISTANCE_FROM_TWO: 'Distance counted from 2',
    Q4RobustnessVariant.NO_TYPOGRAPHY_VECTOR: '$A$, no typography',
    Q4RobustnessVariant.SAME_TERMINAL_PUNCTUATION_VECTOR: '$A$, same ending',
    Q4RobustnessVariant.NO_PROFANITY_VECTOR: '$A$, no filtered words',
    Q4RobustnessVariant.LEAVE_ONE_OUT: 'Without',
}
WEIGHT_LABEL = 'Weight (content change = 1)'
PARTIAL_SPEARMAN_LABEL = 'Partial Spearman $\\rho$'
