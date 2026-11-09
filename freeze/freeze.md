This file describes the methodology at the time of the code freeze on November 6th, 2026, and records the analysis as fixed before the testing split is run. `06_analyze_pairs.py --mode freeze` records a model's freeze before a testing run with a timestamp for that run. This file is committed with all four freeze records after the final testing run. See the paper for details about definitions and methodology. All confidence intervals are set to 95%.

# What is frozen
*See `freeze/<model_name>/freeze_record.json`, `freeze/<model_name>/extraction_log.json` for items*
- Hashes of all files under `src/core/`, `src/analysis/`, `src/scripts/`
- The complete experiment configuration setup
- Hashes of processed corpora, text annotations, difference vectors, and profanity list
- The primary model and three replication models

# Setup
- Model: Llama-3.2-3B base, primary layer 14 of 28 with secondary sweep on 4, 8, 14, 20, 24
- Vectors extracted from text under either mean pooling, last-token, or boundary pooling (the activation of the final token of the aligned word nearest to the midpoint of each line, preferring identical words )
- A: the mean difference of AAE activations from SAE activations over all REAL-AAE development pairs

# Replication and Robustness
Replication Models:
- Llama-3.1-8B base, layer 16 of 32
- Olmo 3 7B base, layer 16 of 32
- Gemma 2 2B, layer 13 of 26
Replications run on CORAAL and REAL-AAE at the primary layer only.

Variant Vectors:
- $A_\text{no typography}$: $A$ computed only over pairs that get no typography normalization
- $A_\text{same ending}$: $A$ computed only over pairs whose lines end with the same punctuation
- $A_\text{no profanity}$: $A$ computed only over pairs with no filter-flagged word on either side
- $A_1$: Mean difference of REAL-AAE AAE backtranslations of AAE tweets subtracted from the original AAE tweets 
- $A_2$: Mean difference of REAL-AAE Gemini SAE translations subtracted from Gemini AAE backtranslations of AAE tweets


# Analysis
*Development results are context only; testing split results decide reported outcomes.*
## Q1
Do held-out AAE-SAE activation differences align consistently with A?

Metrics:
- $s+$: the share of pairs with a positive projection on $A$
- $d_z$: the standardized mean of pair projections on $A$

Tests and Corrections:
- Sign-flip null at 10,000 flips
- Benjamini-Hochberg over 9 tests

Label Rules:
- Consistent: $s+ \ge 0.90$ and $d_z > 1$ at the lower bound
- Weak: $s+ < 0.70$ or $d_z \le 0.5$ at the lower bound
- Partial: otherwise

## Q2
Does $A$ separate AAE grammar-probe lines from matched lines in other AAE corpora?

Metrics:
- Matched AUROC (5 negatives per positive with matched word count, ending, and inclusion of a mention tag), with mean and last pooling

Tests and Corrections:
- Label-shuffle null at 10,000 shuffles
- Benjamini-Hochberg over primary (4 corpora by 2 poolings) and secondary probe families

Label Rules:
- Transfer: FDR $p < 0.05$, with a lower bound $> 0.5$, and estimate $\ge 0.55$
- Small: FDR $p < 0.05$, and estimate $< 0.55$
- Bounded null: FDR $p \ge 0.05$, with an upper bound $< 0.55$
- Inconclusive: otherwise

## Q3
Do pairs' projections on $A$ grow with the amount and type of edit? 

Metrics:
- Spearman and partial Spearman with edits per word (controlled for length)

Tests and Corrections:
- Bootstrap CI 95%
- Benjamini-Hochberg over 12 tests (6 counts by 2 poolings, boundary and mean)

Label Rules:
- Rises: lower bound $> 0$ and $|\rho| \ge 0.10$

## Q4
Which kinds of edit move a pair, and by how much for each?

Metrics:
- Non-linear regression (log, ceiling), weight per edit kind, units in content change edit weight (the umbrella edit kind, including unclassified edits)

Tests and Corrections:
- Bootstrap CI 95% over 2,000 resamples
- Benjamini-Hochberg across kinds, except content change, with num edits $\ge 30$

Label Rules:
- Stronger/weaker: log and ceiling curves both reject $w=1$ after correction with estimates on the same side
- Under 30 edits: kind is too rare
- Not different: otherwise

Robustness:
- Fit again using multi-word marker blocks counted as content change
- Fit again using markers with *ain't* excluded
- Fit again using distance beginning at 2 instead of 1
- Fit again using each variant vector for $A$
- Fit again using grammatical markers with each word left out

Equations for regression on projections $p$ for a pair $i$, summing over edits $e$, with one weight $w$ per edit kind, exponents $\alpha$ and $\beta$, and fitted constants $c$, $\text{top}$, and $D_{50}$:
$$
p_i \approx \text{curve}(\sum_e w_{\text{kind}(e)}\, \text{dist}_e^{-\alpha}\, \text{size}_e^{\beta}) + \text{controls}
$$
$$
\text{curve}(x) = c\log({x + 1}) \quad\text{or}\quad \text{curve}(x) = \frac{x(top)}{x + D_{50}}
$$
- There is no distance term at mean pooling

Controls:
- Format flags
- Filter flags on each side,
- Log length ratio

## Analysis Notes
- Grammatical markers: *ain't, aint, be, done, finna, gon, nothing, nobody, nowhere, none* (Green 2002)
- Q2 corpora: CORAAL, TwitterAAE, AAVE lyrics, and the held-out REAL-AAE split
- Q2 AAVE Social Media corpus results reported without a test due to too few lines
- Q2 secondary family: any probe except *ain't*, negative concord, *ain't*, no filter-flagged lines, CORAAL midpoint pooling (activations read at midpoint word, similar to boundary pooling)
- Q3 levels: orthographic, case and punctuation, lexical, grammatical

# Deviations
*All deviations finalized before examining affected results*
- Nov. 7, 2026: remove AAVE lyrics and AAVE social media corpora from Q2 due to provenance concerns: the AAVE corpus has a single curator, little documentation, and has not passed peer review. Recompute primary family FDR results as 3 corpora by 2 poolings (6 tests), and apply the same adjustment to secondary results as well.
- Nov. 7, 2026: interrupt and restart OLMo 7B freeze step due to compute pod restart.
- Nov. 8, 2026: Add label-error sensitivity check to regression weight verdicts. Use the hand label check to estimate a weight for content change separate from its skew from being the umbrella category; then verdicts whose CIs include this estimate are not robust to label error (See Appendix B). Verdicts and label rules which were frozen remain unchanged.


# Run Order
- `01_process_corpora.py`
- `02_annotate_text.py`
- `03_extract_activations.py`
- `04_compute_estimators.py`
- `06_analyze_pairs.py` with `--mode freeze`
- `06_analyze_pairs.py` with `--mode testing`
- `07_analyze_transfer.py`
Repeat all except for the first with all replication models using `--model`

# Commit Tag
- `freeze-v1`
