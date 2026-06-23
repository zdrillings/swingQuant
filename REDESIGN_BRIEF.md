# SwingQuant Redesign Brief

## Goal

Produce a short list of stocks that delivers **meaningful forward `20d` alpha vs sector**.

This is the only product objective that matters for the redesign.

If the system cannot produce a shortlist that reliably outperforms the eligible names it leaves out, it is not succeeding.

## Why Redesign

The current system has useful infrastructure and useful diagnostics, but it is not achieving the product goal.

Observed failures from current evidence:

- selected shortlist `20d` alpha is weaker than excluded eligible names
- selected names beat excluded names on too few dates
- recent `20d` results are especially weak
- selector complexity has not produced stable uplift
- cross-slot allocation is suppressing stronger sleeves and overexposing weaker ones

The conclusion is not that all research is useless.

The conclusion is:

- the **data and research substrate** is valuable
- the **current shortlist construction approach** is not good enough

## Keep vs Replace

### Keep

- DuckDB historical market store
- SQLite operational ledger
- daily universe snapshot storage
- scan candidate snapshot storage
- command surface (`sq sync`, `sq scan`, `sq monitor`, `sq universe-backfill`, reporting commands)
- feature engineering pipeline
- reporting and attribution tooling

### Replace

- heuristic-first selector layering as the main decision engine
- sleeve-first shortlist construction as the final product logic
- assumption that more ranking heuristics will fix shortlist quality

## Product Definition

### Primary Product

A **daily ranked shortlist** of names expected to generate strong `20d` sector-relative alpha.

### Product Output

Each daily recommendation set should include:

- ticker
- predicted `20d` alpha vs sector
- confidence / rank percentile
- key evidence summary
- slot / sector context
- risk context

### Secondary Products

These are useful, but subordinate:

- candidate discovery / watchlist
- monitor / holdings context
- research diagnostics

The redesign should optimize for the shortlist first, not for generic candidate discovery.

## Core Modeling Shift

### Current Framing

The system currently does:

1. sector sleeve gate
2. score setup quality
3. rank within and across sleeves
4. allocate limited slots

### New Framing

The redesign should do:

1. define the **prediction target** explicitly
2. train a model to estimate cross-sectional forward `20d` alpha
3. rank eligible names by predicted alpha
4. apply only the minimum necessary risk and diversification constraints

The core question becomes:

> Among stocks we are willing to trade, which names have the highest expected forward `20d` alpha?

Not:

> Which names best satisfy a growing stack of heuristic ranking rules?

## Prediction Target

### Primary Target

`alpha_vs_sector_20d`

This should remain the primary optimization target.

### Supporting Targets

Useful secondary targets for analysis:

- `fwd_return_20d`
- `alpha_vs_spy_20d`
- `mfe_20d`
- `mae_20d`

But the main shortlist objective should stay tied to `alpha_vs_sector_20d`.

## Candidate Universe

The redesign should use a broad, daily research universe drawn from the existing `universe_daily_snapshots` substrate.

### Universe Inclusion

Keep practical constraints:

- liquidity floor
- valid price history
- tradable common-stock style universe

### Universe Gating Philosophy

Use gates to remove clearly non-tradable or structurally invalid names.

Do **not** use gates to do most of the prediction work.

Gates should be:

- liquidity / data sanity
- broad regime exclusions when truly necessary
- maybe minimal structural validity

Everything else should be modeled and ranked, not hard-filtered whenever possible.

## Feature Set

The redesign should reuse the current feature base heavily.

### Reusable Feature Families

- relative strength
- momentum
- moving-average distance / slope
- volume participation
- gap behavior
- breakout structure
- earnings timing / earnings reaction
- sector breadth
- contextual benchmarks (`SPY`, `QQQ`, `XLK`, subindustry ETF)

### Feature Policy

- Keep features broad.
- Let the model learn weights/interactions.
- Avoid encoding too much final decision logic as handcrafted score rules.

### Sleeve Information as Features

The existing sleeve signals should not disappear.

Instead, encode them as:

- sector / slot indicators
- pass/fail flags
- signal decomposition features
- structural archetype tags

That turns current sleeve logic into model input rather than model replacement.

## Model Architecture

### First Redesign Version

Start simple.

Recommended first-pass approach:

1. Build a daily cross-sectional dataset from `universe_daily_snapshots`.
2. Restrict to tradable/liquid names.
3. Predict `alpha_vs_sector_20d`.
4. Use a simple, well-regularized ranking model first.

Good initial candidates:

- linear/ridge/lasso rank baseline
- gradient boosting regressor / ranker
- monotonic tree-based model if constraints are useful

Do not start with a highly complex ensemble.

The first requirement is a trustworthy baseline that can beat current heuristics.

### Output

Per ticker per date:

- predicted `20d` alpha
- rank
- optional confidence / uncertainty bucket

## Evaluation Framework

This is the most important part of the redesign.

### Primary Evaluation

Evaluate the actual **daily shortlist outcome**, not just model correlation.

For each scan date:

- take top `N` predicted names
- compare to the rest of the eligible universe
- compute average `20d` alpha
- compute hit rate
- compute date-level win/loss vs excluded universe

### Required Metrics

- shortlist mean `alpha_vs_sector_20d`
- excluded-universe mean `alpha_vs_sector_20d`
- shortlist hit rate
- beat-excluded rate by date
- `% of dates shortlist mean > +2%`
- `% of dates shortlist mean > +5%`
- MFE / MAE and drawdown context

### Validation Discipline

Use time-series-safe validation only.

Required:

- chronological walk-forward
- purged / embargoed splits where needed
- no random shuffle

### Baselines

Every redesign experiment must beat:

- current runtime selector
- simple opportunity-score rank
- simple signal-score rank
- random eligible baseline
- sector-neutral equal-weight eligible baseline

If the new design cannot beat these consistently, reject it.

## Acceptance Criteria

The redesign is not successful unless it clears explicit bars.

### Minimum Bar

Over a meaningful out-of-sample window:

- shortlist mean `alpha_vs_sector_20d` > excluded mean
- shortlist beat-excluded rate > `55%`
- shortlist positive-alpha rate > `55%`
- shortlist mean `20d` alpha > `+2%`

### Strong Bar

For true production confidence:

- beat-excluded rate > `60%`
- shortlist mean `20d` alpha > `+3%`
- strong recent-window stability
- no major collapse in the latest `40` matured dates

These bars should be revisited only if they prove unrealistic after honest testing.

But the redesign should start with a real bar, not a vague hope.

## Command Structure Plan

We should preserve the command surface where practical, but repurpose the internals.

### Keep

- `sq sync`
- `sq universe-backfill`
- `sq scan`
- `sq scan-analysis`
- `sq monitor`

### Add / Repurpose

- a new research/model-training command for direct `20d` alpha ranking
- a new shortlist bakeoff report focused on `20d` objective
- optional model registry / promotion path for the shortlist model

### Likely Internal Shift

`sq scan` should eventually become:

1. build latest feature frame
2. score all eligible tradable names with the promoted alpha model
3. apply light diversification/risk constraints
4. emit top shortlist

Not:

1. run many heuristic slot gates
2. stack multiple selector overlays
3. hope the final allocation solves the ranking problem

## Role of Slots / Sleeves

Slots should become **risk and context tools**, not necessarily the first-class product unit.

Possible roles:

- sector diversification constraint
- archetype labeling
- fallback safety rails
- explanatory context in reports

But the redesign should not assume the final shortlist must be built by giving each slot its own fixed quota unless the evidence later proves that helps.

## Exit Logic in the Redesign

Exits are important, but not the first bottleneck to solve.

Current status:

- exit evidence is too sparse after cleaning non-SwingQuant holdings
- the bigger failure is still shortlist quality

So redesign order should be:

1. shortlist prediction engine
2. shortlist evaluation
3. only then deeper exit redesign

## Implementation Phases

### Phase 1: Dataset and Target

- formalize the redesign dataset from `universe_daily_snapshots`
- define tradable universe inclusion rules
- define exact `20d` target and eligible set
- build baseline evaluation tables

### Phase 2: Simple Ranking Baselines

- signal-score baseline
- opportunity-score baseline
- simple linear / regularized model
- simple tree-based regressor / ranker

### Phase 3: Shortlist Evaluation Harness

- top-`N` daily shortlist evaluation
- date-level beat-excluded comparisons
- sector / slot breakdowns
- recent-window stability

### Phase 4: Production Candidate

- choose one winning shortlist model
- integrate into `sq scan`
- keep the old path available for comparison during transition

### Phase 5: Exits and Monitoring

- once shortlist quality is real, revisit exit optimization
- align monitor around the new shortlist logic

## Questions the Redesign Must Answer

1. Can a direct `20d` alpha ranker beat the heuristic shortlist?
2. Does sector-relative alpha prediction work better broad-universe or sleeve-scoped?
3. Do we need slot quotas at all?
4. Are the best names concentrated in a few sectors/archetypes, or broadly distributed?
5. How much of current sleeve logic survives as features versus hard gates?

## Immediate Next Build

The next practical build should be:

1. a direct `20d` shortlist evaluation dataset / service
2. a baseline bakeoff over:
   - current runtime
   - opportunity-score top `N`
   - signal-score top `N`
   - simple model top `N`
3. a report whose only question is:
   - which daily shortlist best delivers forward `20d` alpha?

That should become the new center of the project.
