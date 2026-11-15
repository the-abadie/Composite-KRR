# Search controls and trial evidence

Every normal configured run saves its numerical search evidence automatically.
Existing search budgets, bounds, proposal distributions and stopping defaults are
preserved. New runs start independently; the evidence files are not a resumable
optimizer checkpoint and are never imported from another model.

## Configuration

All controls appear in `krr_cli.py template`, `krr_cli.py schema`, the bundled
`config.json` and each run's frozen `resolved_config.json`. This example spells
out the search defaults:

```json
"search": {
  "seed": null,
  "history_batch_size": 32,
  "alpha_bounds": [1e-9, 100.0],
  "gamma_bounds": [1e-9, 100.0],
  "gamma_prior_max_samples": 5000,
  "gamma_prior_decades": 2.5,
  "refine_decades": 1.0,
  "top_k_fraction": 0.25,
  "top_k_min_candidates": 5,
  "top_k_padding_decades": 0.5,
  "stage3_weight_concentration": 25.0,
  "random": {"stage1": 75, "stage2": 75, "stage3": 0},
  "bayesian": {
    "trials": 50,
    "rounds": 1,
    "timeout": null,
    "patience": 25,
    "batch_size": 1,
    "refine_decades": 0.3,
    "weight_logit_radius": 1.5,
    "min_delta_absolute": 0.0,
    "min_delta_relative": 0.0,
    "sampler": {
      "multivariate": false,
      "group": false,
      "n_startup_trials": 10,
      "n_ei_candidates": 24,
      "constant_liar": false,
      "consider_endpoints": false,
      "consider_magic_clip": true,
      "prior_weight": 1.0
    }
  },
  "scoring": "neg_mean_absolute_error",
  "target_space": "original"
}
```

`seed` at the top level continues to control data splitting, folds and model
randomness. `search.seed` controls optimizer proposals and prior subsampling;
null inherits the top-level seed. The effective optimizer seed is saved directly
in `run_metrics.json` and `search_summary.json`, even when the declared value is
null. Changing only `search.seed` leaves ordered outer/CV indices unchanged.

| Control | Meaning |
|---|---|
| `gamma_prior_decades` | Initial gamma range is the median-based center divided/multiplied by `10**value`, clipped to legal bounds. |
| `refine_decades` | Stage 2 range around the Stage 1 winner, using the same decade convention. |
| `top_k_padding_decades` | Padding around Stage 3 top-candidate log ranges. |
| `stage3_weight_concentration` | Dirichlet concentration around the top candidates' mean weights. |
| `bayesian.refine_decades` | Bayesian alpha/gamma range around this run's best random candidate. |
| `bayesian.rounds` | Divide the total Bayesian trial cap into successive local searches, recentering on the best candidate from this same run. Default 1 preserves the standard search. |
| `bayesian.weight_logit_radius` | Natural-logit radius around the Bayesian weight center. |
| `bayesian.min_delta_absolute` | Absolute score improvement required to reset meaningful-progress patience. |
| `bayesian.min_delta_relative` | Relative improvement threshold, multiplied by the absolute anchor score. |
| `bayesian.sampler` | Explicit Optuna TPE options; grouping requires `multivariate=true`. |
| `history_batch_size` | Maximum cached candidate count between history checkpoints, rounded up to a whole GPU candidate batch. |

The meaningful improvement threshold is
`max(min_delta_absolute, min_delta_relative * abs(anchor_score))`. Improvement is
strictly greater than this threshold. Small gains accumulate against the last
meaningful anchor. The exact best-scoring model is always retained, even when its
gain is too small to reset patience. Zero tolerances preserve the previous
stopping rule. A batched Bayesian search finishes the already evaluated batch
before stopping.

When `bayesian.rounds > 1`, the total `trials` budget is divided as evenly as
possible (earlier rounds receive any remainder). Each round has a fresh TPE
study, with a deterministic seed derived from `search.seed` after round 1, and
reevaluates its current center as its first counted trial. No observations or
fitted parameters are imported from other runs. Alpha and gamma boxes are
recentered within the original legal bounds; weight-logit boxes are recentered
on the same incumbent. All earlier candidates remain eligible for final model
selection. Patience resets per round; unused patience-stopped trial allocations
are not transferred. `timeout` covers the entire Bayesian phase. Bounds, counts,
stop reasons and scores are recorded per round, and aggregate history keeps
sequential evaluation indices. A single round retains the original proposal
sequence, seed, bounds, history naming and stopping behavior.

For example, CLI overrides can vary optimizer randomness and refinement width:

```bash
.venv/bin/python krr_cli.py resolve run.json \
  --set search.seed=123 \
  --set search.gamma_prior_decades=3.0 \
  --set search.bayesian.refine_decades=0.6 \
  --set search.bayesian.min_delta_relative=0.001 \
  --output configs/search-seed123.json
```

Structured TOML uses `[search]`, `[search.bayesian]` and
`[search.bayesian.sampler]`. Legacy flat conversion supports explicit aliases
including `KRR_SEARCH_SEED`, `KRR_GAMMA_PRIOR_DECADES`, `KRR_REFINE_DECADES`,
`KRR_TOP_K_PADDING_DECADES`, `KRR_STAGE3_WEIGHT_CONCENTRATION`,
`KRR_BAYESIAN_REFINE_DECADES`, `KRR_BAYESIAN_WEIGHT_LOGIT_RADIUS`,
`KRR_BAYESIAN_MIN_DELTA_ABSOLUTE`, `KRR_BAYESIAN_MIN_DELTA_RELATIVE`,
`KRR_BAYESIAN_SAMPLER` and `KRR_HISTORY_BATCH_SIZE`.

## Saved artifacts

| File | Contents |
|---|---|
| `resolved_config.json` | Declared and defaulted configuration, including every exposed search setting. |
| `search_history.jsonl` | Append-only events: settings, numerical gamma centers, stage bounds/distributions, every recorded candidate and fold scores, failures, stage completion and final status. |
| `search_summary.json` | Effective settings, seed context, per-stage counts/bounds/times/stop reasons, selected parameters, best-so-far evidence and the history SHA-256. |
| `run_metrics.json` | Explicit search configuration and effective optimizer seed, plus hashes of completed search artifacts. |

A trial event stores canonical `alpha`, `gammas` and normalized `kernel_weights`,
including fixed Stage 1 weights. It also retains sampled parameter names or TPE
logits, ordered `split_scores`, mean/std score, failed-fold indices, validity,
best-so-far, elapsed search wall time and timing. Scores retain sklearn's maximize convention; negative
MAE remains negative. Non-finite values are JSON null, accompanied by failure
status rather than nonstandard NaN/Infinity tokens. Actual adaptive bounds and
Dirichlet parameters are saved numerically, not reconstructed from rounded logs.

Cached random candidates are sampled before scoring and flushed after each
completed checkpoint batch. GPU checkpoint boundaries align with the configured
candidate batch size. The history records measured batch wall time and an
explicitly labeled amortized time per candidate. It does not claim an individual
wall time for parallel solves. Uncached sklearn random search flushes at stage
completion; individual sequential Bayesian evaluations and completed Bayesian
batches flush immediately. An interrupted in-flight batch has no completed fold
scores. Caught batch exceptions retain its proposed parameters with failed status.
Already flushed records survive later preparation/scoring/refit failures, with a
failed summary for caught exceptions. A hard process kill can leave only the
complete JSONL lines written before the kill and no final summary. File close
flushes writes to the filesystem; there is no per-trial fsync or guarantee against
host/filesystem power loss. Existing evidence files are never replaced or resumed.

Optional kernel-contribution searches use distinct
`kernel_contribution_searches/single_<index>` and `drop_<index>` directories with
the same artifacts. Their hashes are included in the main native metrics.

Direct Python callers can pass `search_output_dir` and optional `search_context`
to `staged_random_search_cv`. Without an output directory the API performs no
evidence-file writes. Candidate proposal settings and selection behavior are the
same with or without recording.
