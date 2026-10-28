from src.core.paths import RAW_DATA_DIR
import src.core.text_rules as text_rules
from src.core.config_utils import (
    BaselineDatasetConfig,
    CoraalSourceConfig,
    DatasetConfig,
    ExperimentConfig,
    GithubSourceConfig,
    HFSourceConfig,
    Q1LabeledThresholds,
    Q1Thresholds,
    Q4OrthographicThresholds,
    Q4RegressionSettings,
    RealAaeSourceConfig,
    SupervisedDatasetConfig,
)
from src.core.utils import (
    DifferenceVector,
    PoolingType,
    Q3EditCount,
    Q4CurveFunction,
    Q4EditKind,
)


EXPERIMENT_CONFIG = ExperimentConfig(
    model_hf_path='meta-llama/Llama-3.2-3B',
    load_in_4bit=False,
    bnb_4bit_quant_type='nf4',

    target_layers=[4, 8, 14, 20, 24],
    target_ranks=[4],
    target_alphas=[
        -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0,
    ],

    primary_layer=14,
    primary_pooling_type=PoolingType.BOUNDARY,
    
    debug_max_split_lines=None,

    h1_corpora=[
        'twitter_aae',
        'aave_social_media',
        'aave_lyrics',
        'coraal_merged',
        'wikitext',
        'real_aae_aae_supervised',
        'real_aae_sae_supervised',
    ],

    num_equal_n_repeats=100,
    num_null_draws=200,
    num_cv_folds=5,

    q1_labeled_bounds=Q1LabeledThresholds(
        consistent=Q1Thresholds(
            positive_projection_ratio=0.90,
            standardized_mean_ci_lower=1.0,
        ),
        weak=Q1Thresholds(
            positive_projection_ratio=0.70,
            standardized_mean_ci_lower=0.5,
        ),
    ),
    q1_fdr_vectors=(DifferenceVector.OFFSET, DifferenceVector.AUTHORSHIP, DifferenceVector.DIALECT),
    q1_layer_sweep_vectors=(DifferenceVector.OFFSET,),
    q1_num_sign_flips=10_000,
    q1_num_shuffled_pairings=1_000,

    q3_edits_by_count={
        Q3EditCount.TOTAL: tuple(Q4EditKind),
        Q3EditCount.ORTHOGRAPHIC: (
            Q4EditKind.ORTHOGRAPHIC,
        ),
        Q3EditCount.FUNCTION_WORD: (
            Q4EditKind.FUNCTION_WORD,
            Q4EditKind.COPULA_ADDED,
        ),
        Q3EditCount.LEXICAL: (
            Q4EditKind.FILTER_FLAGGED_WORD,
            Q4EditKind.GRAMMAR_WORD,
            Q4EditKind.ADDED_OR_DROPPED,
            Q4EditKind.SHORTHAND_FUNCTION_WORDS,
            Q4EditKind.SHORTHAND_EXPANDED,
            Q4EditKind.CONTENT_CHANGE,
        ),
        Q3EditCount.MECHANICS: (
            Q4EditKind.CAPITALIZATION,
            Q4EditKind.PUNCTUATION,
        ),
    },
    q3_num_dose_bins=5,
    q3_q4_pooling_types=(PoolingType.BOUNDARY, PoolingType.MEAN),

    q4_primary_curves=(Q4CurveFunction.LOG, Q4CurveFunction.CEILING),
    q4_orthographic_thresholds=Q4OrthographicThresholds(
        strict=0.75,
        relaxed=0.6,
        loose=0.5,
    ),
    q4_respelling_threshold=0.6,
    q4_single_letter_respelling_threshold=0.7,
    q4_function_words=tuple('''
        the a an is are was were be been am do does did has have had to of in on at
        for and but that it they he she we you i not no with it's that's what who this
        these those my your his her their our
    '''.split()),
    q4_grammar_words=tuple('''
        ain't aint be done finna gon tryna nothing nobody nowhere none
    '''.split()),
    q4_copula_words=('is', 'are'),
    q4_format_flags=('ending_changed', 'case_changed', 'tag_count_changed', 'has_asterisk'),
    q4_regression_settings=Q4RegressionSettings(
        free_shape_breakpoints=(
            0.5, 1.0, 1.5, 2.0, 3.0,
        ),
        initial_parameters_by_curve={
            Q4CurveFunction.LOG: (1.0,), # scale
            Q4CurveFunction.CEILING: (3.0, 0.0), # top, log half-effect dose
            Q4CurveFunction.STRAIGHT: (0.5, 0.5), # slope, intercept
            Q4CurveFunction.FREE_SHAPE: (1.0,), # slope (bends at 0)
        },
        initial_distance_decay_exponent=0.5,
        initial_size_exponent=0.2,
    ),
    q4_num_held_out_folds=5,
    q4_num_calibration_bins=10,
    q4_num_bootstrap_samples=2_000,
    q4_marginal_effect_distances=(1, 3),
    
    corpus_split_size=3_782,
    batch_size=32,
    max_seq_len=64,
    random_seed=42,

    num_bootstrap_samples=10_000,
    p_alpha=0.05,
)


CORPUS_CONFIG = {
    'coraal_merged': DatasetConfig(
        source_config=CoraalSourceConfig(
            in_dir=RAW_DATA_DIR / 'coraal',
            header=True,
            delimiter='\t',
            text_col_idx=3,
            speaker_col_idx=1,
            interviewer_marker='_int_',
            pause_pattern=r'\(pause \d+\.\d+\)'
        ),

        min_len_chars=15,
        min_len_words=5,
        max_len_words=35,

        sentencize=True,
        pool_size=40_000,

        skipped_data_patterns=[
            r'rd-',
            r'\(pause \d+\.\d+\)',
            r'inaudible',
            r'unintelligible',
            r'\?\?+',
            r'(^|[\[\]\(\)<>])\s*\?',
        ],
        skipped_case_patterns=[
            r'[()<>]',
            r'[^.!?]\s*$',
            r'(|\s)\?(\s|$)',
            r'-\s*[.,!?]*\s*$',
            r'\b\w+-\s',
        ],
        replacement_patterns=[
            (r'\([^\)]+\)', ''),
            (r'<[^>]+>',    ''),
            (r'[\[\]/]',    ''),
            (r'\s+',        ' '),
        ],
    ),

    'real_aae': SupervisedDatasetConfig(
        source_config=RealAaeSourceConfig(
            in_dir=RAW_DATA_DIR / 'real_aae',
            header=True,
            delimiter=',',
            aae_col_idx=0,
            sae_col_idx=1,
            aae_back_col_idx=2,
        ),

        min_len_chars=15,
        min_len_words=5,
        max_len_words=35,

        sentencize=False,

        boundary_min_previous_words=2,
        boundary_min_following_words=1,

        skipped_data_patterns=[],
        skipped_case_patterns=[],
        replacement_patterns=[
            (r'(?:http|www\.)\S+', ''),
            (r'^RT\s+', ''),
            (r'&amp;|&lt;|&gt;', ''),
            (r'@\w+', text_rules.MENTION_PLACEHOLDER),
            (r'\s+', ' '),
        ],
    ),

    'aave_lyrics': DatasetConfig(
        source_config=GithubSourceConfig(
            git_url='https://raw.githubusercontent.com/jazmiahenry/aave_corpora/refs/heads/main/corpora/lyrics_corpora.txt',
            delimiter='\n',
        ),

        min_len_chars=15,
        min_len_words=5,
        max_len_words=35,

        sentencize=False,

        skipped_data_patterns=[
            r'(?:^|\blyrics)(?:verse|chorus|intro|outro|hook|bridge|interlude|skit|narrator|scratches|produced|vocals sampled|beat changes)\b',
            r'\bsample[ds]?$',
        ],
        skipped_case_patterns=[
            r'^[a-z]+$',
        ],
        replacement_patterns=[
            (r'embed$', ''),
            (r'\s+', ' '),
        ],
    ),

    'aave_social_media': DatasetConfig(
        source_config=GithubSourceConfig(
            git_url='https://raw.githubusercontent.com/jazmiahenry/aave_corpora/refs/heads/main/corpora/social_media_corpora.txt',
            delimiter='\n',
        ),

        min_len_chars=15,
        min_len_words=5,
        max_len_words=35,

        sentencize=False,

        skipped_data_patterns=[],
        skipped_case_patterns=[],
        replacement_patterns=[
            (r'(?:http|www\.)\S+', ''),
            (r'\bRT\b\s*:?\s*', ''),
            (r'&amp;|&lt;|&gt;', ''),
            (r'(?:^|\s)_[\w]+', f' {text_rules.MENTION_PLACEHOLDER} '),
            (
                r'[\U0001F1E6-\U0001F1FF'
                r'\U0001F300-\U0001F9FF'
                r'\U0001FA70-\U0001FAFF'
                r'\u2600-\u27BF'
                r'\uFE0F\u200D]+',
                '',
            ),
            (r'\s+', ' '),
        ],
    ),

    'twitter_aae': DatasetConfig(
        source_config=HFSourceConfig(
            hf_path='lighteval/TwitterAAE',
            split='test',
            text_col='tweet',
        ),

        min_len_chars=15,
        min_len_words=5,
        max_len_words=35,

        sentencize=False,

        skipped_data_patterns=[],
        skipped_case_patterns=[],
        replacement_patterns=[
            (r'(?:http|www\.)\S+', ''),
            (r'^RT\s+', ''),
            (r'&amp;|&lt;|&gt;', ''),
            (r'@\w+', text_rules.MENTION_PLACEHOLDER),
            (r'\\u[0-9a-fA-F]{4}', ''),
            (r'\s+', ' '),
        ],
    ),

    'wikitext': BaselineDatasetConfig(
        source_config=HFSourceConfig(
            hf_path='wikitext',
            hf_name='wikitext-2-raw-v1',
            split='test',
            text_col='text',
        ),
        
        min_len_chars=15,
        min_len_words=5,
        max_len_words=35,

        sentencize=True,

        skipped_data_patterns=[
            r'^(?:\s*=\s*)+[^=]+(?:\s*=\s*)+\s*$',
        ],
        skipped_case_patterns=[],
        replacement_patterns=[
            (r'\s+@(.)@\s+', r'\1'),
            (r'\s+([.,;:?!])', r'\1'),
            (r"\s+('\w+)", r'\1'),
            (r"(\w+)\s+(')(?=\s|$)", r'\1\2'),
            (r'([(\[{])\s+', r'\1'),
            (r'\s+([)\]}])', r'\1'),
            (r'"\s+([^"]*?)\s+"', r'"\1"'),
            (r'(\d+):\s+(\d+)', r'\1:\2'),
            (r'\s+', ' '),
        ],
    ),
}
