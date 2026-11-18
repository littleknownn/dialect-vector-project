# What Is in a "Dialect Direction"?

Code for "AAE Signals in Language Model Activation Space: What Is in a 'Dialect Direction'?" by Matthew Philbin.

The project rebuilds an AAE–SAE "dialect direction" from AAE tweets and their SAE translations in the REAL-AAE corpus, then tests whether it is consistent across pairs, which kinds of edits it carries, and whether it separates AAE grammatical features in CORAAL.

## Layout

- `src/scripts/`: the pipeline, run in numbered order
- `src/analysis/`: the analyses
- `src/core/`: shared settings, the edit tagger, and statistics
- `src/plots/`: figures and tables
- `freeze/`: the frozen analysis rules and freeze record

## Running

Requires Python 3.12. Model weights come from Hugging Face; some (e.g., Llama) require accepting a license there first. Run settings are in `src/core/config.py`.

## Data

REAL-AAE is from Wu et al. (2026) and CORAAL is from Kendall and Farrington (2023); see each source for access and terms.
