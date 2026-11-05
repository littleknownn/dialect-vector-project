import argparse
import dataclasses

from src.core.paths import RAW_DATA_DIR
import src.core.text_rules as text_rules
from src.core.config_utils import (
    BaselineDatasetConfig,
    CoraalSourceConfig,
    DatasetConfig,
    ExperimentConfig,
    GithubSourceConfig,
    HFSourceConfig,
    ModelConfigs,
    ModelSpec,
    Q1Config,
    Q2Config,
    Q3Config,
    Q4Config,
    TaggingConfig,
    Q1LabeledThresholds,
    Q1Thresholds,
    Q4OrthographicThresholds,
    Q4RegressionSettings,
    RealAaeSourceConfig,
    SupervisedDatasetConfig,
)
from src.core.utils import (
    DifferenceVector,
    ModelName,
    PoolingType,
    Q3EditCount,
    Q4CurveFunction,
    Q4EditKind,
)


PRIMARY_MODEL_NAME: ModelName = ModelName.LLAMA_3B
MODEL_SPECS: dict[ModelName, ModelSpec] = {
    ModelName.LLAMA_3B: ModelSpec(
        'meta-llama/Llama-3.2-3B', primary_layer=14,
    ),
    ModelName.LLAMA_8B: ModelSpec(
        'meta-llama/Llama-3.1-8B', primary_layer=16,
    ),
    ModelName.OLMO_7B: ModelSpec(
        'allenai/Olmo-3-1025-7B', primary_layer=16,
    ),
    ModelName.GEMMA_2B: ModelSpec(
        'google/gemma-2-2b', primary_layer=13, attn_implementation='eager',
    ),
}
REPLICATION_CORPUS_NAMES: tuple[str, ...] = ('real_aae', 'coraal_merged')


EXPERIMENT_CONFIG = ExperimentConfig(
    model_hf_path=MODEL_SPECS[PRIMARY_MODEL_NAME].model_hf_path,
    attn_implementation=MODEL_SPECS[PRIMARY_MODEL_NAME].attn_implementation,
    load_in_4bit=False,
    bnb_4bit_quant_type='nf4',
    
    corpus_split_size=3_782,
    batch_size=32,
    max_seq_len=64,
    random_seed=42,

    target_layers=[4, 8, 14, 20, 24],

    primary_layer=MODEL_SPECS[PRIMARY_MODEL_NAME].primary_layer,
    primary_pooling_type=PoolingType.BOUNDARY,

    num_cv_folds=5,
    num_bootstrap_samples=10_000,
    p_alpha=0.05,

    q1=Q1Config(
        labeled_bounds=Q1LabeledThresholds(
            consistent=Q1Thresholds(
                positive_projection_ratio=0.90,
                standardized_mean_ci_lower=1.0,
            ),
            weak=Q1Thresholds(
                positive_projection_ratio=0.70,
                standardized_mean_ci_lower=0.5,
            ),
        ),
        fdr_vectors=(DifferenceVector.DIALECT, DifferenceVector.AUTHORSHIP, DifferenceVector.DIALECT_GEMINI),
        layer_sweep_vectors=(DifferenceVector.DIALECT,),
        num_sign_flips=10_000,
        num_shuffled_pairings=1_000,
    ),

    q2=Q2Config(
        primary_corpus_names=(
            'twitter_aae', 'aave_lyrics', 'coraal_merged', 'real_aae',
        ),
        reported_corpus_names=('aave_social_media',),
        midpoint_corpus_names=('coraal_merged',),
        primary_pooling_types=(PoolingType.MEAN, PoolingType.LAST),
        num_negatives_per_positive=5,
        num_null_shuffles=10_000,
        min_auroc=0.55,
        power=0.80,
        sensitivity_vectors=(
            DifferenceVector.NO_TYPOGRAPHY,
            DifferenceVector.SAME_TERMINAL_PUNCTUATION,
            DifferenceVector.NO_PROFANITY,
        ),
    ),

    q3=Q3Config(
        edits_by_count={
            Q3EditCount.TOTAL: tuple(Q4EditKind),
            Q3EditCount.ORTHOGRAPHIC: (
                Q4EditKind.ORTHOGRAPHIC,
                Q4EditKind.SHORTHAND_FUNCTION_WORDS,
                Q4EditKind.SHORTHAND_EXPANDED,
            ),
            Q3EditCount.CASE_AND_PUNCTUATION: (
                Q4EditKind.CAPITALIZATION,
                Q4EditKind.PUNCTUATION,
            ),
            Q3EditCount.LEXICAL: (
                Q4EditKind.CONTENT_CHANGE,
                Q4EditKind.FILTER_FLAGGED_WORD,
                Q4EditKind.ADDED_OR_DROPPED,
            ),
            Q3EditCount.GRAMMATICAL: (
                Q4EditKind.GRAMMAR_WORD,
                Q4EditKind.FUNCTION_WORD,
                Q4EditKind.COPULA_ADDED,
            ),
        },
        num_dose_bins=5,
    ),

    q3_q4_pooling_types=(PoolingType.BOUNDARY, PoolingType.MEAN),

    tagging=TaggingConfig(
        orthographic_thresholds=Q4OrthographicThresholds(
            strict=0.75,
            relaxed=0.6,
            loose=0.5,
        ),
        respelling_threshold=0.6,
        single_letter_respelling_threshold=0.7,
        aint_words=tuple("""
            aint ain't
        """.split()),
        function_words=tuple('''
            the a an is are was were be been am do does did has have had to of in on at
            for and but that it they he she we you i not no with it's that's what who this
            these those my your his her their our
        '''.split()),
        grammar_words=tuple('''
            ain't aint be done finna gon nothing nobody nowhere none
        '''.split()),
        copula_words=('is', 'are'),
    ),

    q4=Q4Config(
        primary_curves=(Q4CurveFunction.LOG, Q4CurveFunction.CEILING),
        format_flags=('ending_changed', 'case_changed', 'tag_count_changed', 'has_asterisk'),
        regression_settings=Q4RegressionSettings(
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
        min_edits_for_verdict=30,
        num_held_out_folds=5,
        num_calibration_bins=10,
        num_bootstrap_samples=2_000,
        num_robustness_bootstrap_samples=300,
        robustness_distance_shift=1,
        marginal_effect_distances=(1, 3),
    ),
    
    debug_max_split_lines=None,
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


def get_model_configs(
    model_name: ModelName,
    corpus_config: dict[str, DatasetConfig] = CORPUS_CONFIG, #
    experiment_config: ExperimentConfig = EXPERIMENT_CONFIG, #
    model_specs: dict[ModelName, ModelSpec] = MODEL_SPECS,
    primary_model_name: ModelName = PRIMARY_MODEL_NAME,
    replication_corpus_names: tuple[str, ...] = REPLICATION_CORPUS_NAMES,
) -> ModelConfigs:
    if model_name == primary_model_name:
        return ModelConfigs(corpus_config, experiment_config)
    model_spec = model_specs[model_name]
    replication_corpus_config = {
        corpus_name: corpus_config[corpus_name]
        for corpus_name in replication_corpus_names
    }
    keep_replication_corpora = lambda corpus_names: tuple(
        corpus_name
        for corpus_name in corpus_names
        if corpus_name in replication_corpus_names
    )
    replication_q2_config = dataclasses.replace(
        experiment_config.q2,
        primary_corpus_names=keep_replication_corpora(experiment_config.q2.primary_corpus_names),
        reported_corpus_names=keep_replication_corpora(experiment_config.q2.reported_corpus_names),
        midpoint_corpus_names=keep_replication_corpora(experiment_config.q2.midpoint_corpus_names),
    )
    replication_experiment_config = dataclasses.replace(
        experiment_config,
        model_hf_path=model_spec.model_hf_path,
        attn_implementation=model_spec.attn_implementation,
        target_layers=[model_spec.primary_layer],
        primary_layer=model_spec.primary_layer,
        q2=replication_q2_config,
    )
    model_configs = ModelConfigs(replication_corpus_config, replication_experiment_config)
    return model_configs


def add_model_argument(
    argument_parser: argparse.ArgumentParser,
    primary_model_name: ModelName = PRIMARY_MODEL_NAME,
) -> None:
    argument_parser.add_argument(
        '--model', type=ModelName, choices=list(ModelName), default=primary_model_name,
    )
