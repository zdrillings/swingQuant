# Research Ideas — SwingQuant Signal Layer

- generated: 2026-10-05
- purpose: candidate research directions for the shortlist signal layer, grounded in the current gate standoff.
- status: idea register, not a plan. Nothing here is implemented.

## Why this exists

As of 2026-10-02 the gate has rejected the entire 11-candidate roster five consecutive nights
(ISSUE-073). The standoff shape: models passing full-OOS Spearman (signal_proxy +0.013, ic_sign
+0.019, xgboost +0.008) fail the recency floors, while the recency passers fail `min_full_oos_spearman`.
Two demands — skill sometime, skill recently — are satisfied by disjoint candidate sets.

Three root findings frame everything below:

1. **Forensic (2026-09-29, `reports/baseline_forensic.md`)**: the old +0.46 pooled Spearman edge
   (run 65) was real but lived in a *different configuration* — 20d label, top-10 basket,
   sector-specific xgboost, 19-20 OOS dates. Rescoring today's roster on that setup recovers only
   ~0.05. The ~0.39 remainder is unexplained by measurement: either the signal layer lost skill in
   the migration or the config changes destroyed it. This is unresolved and is the cheapest thing
   to resolve.
2. **ISSUE-077**: the 09-25 champion's pooled deciles are monotonically *inverted* (dec0 -2.15% →
   dec9 +5.80% mean actual). The raw ranking points the wrong way in reversal stretches. The
   unconditional_reversal flip fixes exactly 2 of 11 models, and the fixed result is +0.021 — noise.
3. **ISSUE-074**: the "11 candidates" are ~4 independent bets (signal_proxy ≡ ensemble ≡
   reversal_rules in several windows to 6 decimals). Clone clusters, not diversity.

## How to evaluate an idea

The funnel an idea must survive: **feature (or label) → fold-local rank-IC screen
(min_feature_ic 0.030, 20% observation floor) → walk-forward folds → promotion gate windows
(1fold/3fold floors + full-OOS Spearman ≥ 0.0)**. A bakeoff (`shortlist_bakeoff`) compares
candidates head-to-head; nothing ships without passing the same gate the champion faces.

Every idea below carries: data source, implementation sketch, and verification. Effort scale:
**S** = a day of work, **M** = several days, **L** = a week+ or new data pipeline.

---

## A. Direct attacks on the standoff (run first)

### A1. Forensic replication candidate
- **What**: re-run the exact run-65 config — `alpha_vs_sector_20d` label, top-10 basket,
  `sector_specific` model scope, xgboost balanced_depth4, no fold-local IC screen — as a research
  candidate in the bakeoff.
- **Why**: splits "signal layer died" from "config migration killed it". If the edge is still there
  under 20d/sector-specific, the recovery path is reverting toward that structure, not inventing new
  signals. Highest information-per-hour available.
- **Implementation**: clone the bakeoff runner with run-65 params; the SQLite run row (id=65)
  records the full config.
- **Verify**: bakeoff report shows pooled + per-date Spearman on the run-65 date grid; compare
  against the forensic's stored numbers.
- **Effort**: S.

### A2. Dual-horizon champion
- **What**: train 20d-label models alongside 60d; let the gate promote whichever passes. A 20d
  champion still holds through the 60d evaluation window, so nothing about the operating model
  changes.
- **Why**: the 60d conversion is where most measurable skill disappeared. Shortening the label
  horizon re-surfaces the regime the old edge lived in.
- **Implementation**: add `horizon_days: 20` variants of the existing candidate models; the
  pipeline already reads horizon from config.
- **Verify**: 20d variants' 1fold/3fold/full-OOS windows vs the current 60d roster in one bakeoff.
- **Effort**: S–M.

### A3. Decile-inversion "avoid list"
- **What**: harvest both ends of an inverted ranker instead of fixing its sign. When a model's
  pooled deciles are monotone-inverted (ISSUE-077), forbid its raw-top decile from the buyable
  universe and promote from the rest.
- **Why**: negative skill is signal. The inversion is stable across nights; the flip-fix is noise
  (+0.021). An avoid-filter uses the information without pretending the ranking is right.
- **Implementation**: post-ranking filter in the champion selection or scan layer; decile inversion
  detected from the OOS artifact (reuse the forensic's decile table).
- **Verify**: backtest the "promote-after-drop-top-decile" rule on the 09-25 champion's OOS
  predictions; compare hit/beat vs the as-is and flipped variants.
- **Effort**: S.

### A4. Regime-interacted features (kill the flip debate)
- **What**: add `feature × regime_onehot` interactions to the feature matrix so the model learns
  per-regime behavior directly (momentum works in trending, mean-reversion in reversal).
- **Why**: the entire flip saga (reverted 09-29, diagnostic-only since) exists because a global
  model cannot express regime-conditional sign. Learning the interaction removes score negation,
  variant adoption, and ISSUE-077's structural problem in one move.
- **Implementation**: expand the feature builder with regime-interaction columns; the regime
  classification per date is already computed for regime matching.
- **Verify**: walk-forward folds show regime-interaction terms surviving the IC screen with
  non-zero coefficients; full-OOS Spearman improves without any flip applied.
- **Effort**: M.

---

## B. After-hours signals

AH data currently feeds **zero** model features — it only powers the scan-side
`extended-hours-snapshot` (255 tickers, nightly, price + spotty volume). The shortlist feature
matrix sees nothing after 16:00.

### B1. Overnight vs RTH return decomposition
- **What**: split each daily return into overnight (prev close → open) and intraday (open → close),
  then roll 5/20d. Features: `overnight_ret_5d`, `rth_ret_5d`, `overnight_minus_rth_5d`.
- **Why**: names accumulated overnight and distributed intraday are being bought by informed hands
  — the classic swing setup. Pure computation from existing OHLC.
- **Implementation**: compute in the snapshot feature builder from market_data.duckdb OHLC.
- **Verify**: rank-IC screen on the new features fold-locally; they must clear 0.030 on their own.
- **Effort**: S.

### B2. AH-RTH divergence confirmation/distribution
- **What**: cross-sectional rank of tonight's AH move vs today's RTH move.
  `ah_minus_rth_return_1d` and its 5d persistence. Strong RTH close + AH fade = distribution
  (avoid next day); flat RTH + AH bid = accumulation (the candidate).
- **Why**: the extended-hours snapshot already exists nightly; this turns it from a report into a
  signal.
- **Implementation**: join `extended-hours-snapshot` output into the feature pipeline; needs a
  rolling history table of AH prices (start accumulating nightly, then backfill).
- **Verify**: same IC screen; check hit/beat on the accumulation side specifically.
- **Effort**: M (history accumulation is the long pole).

### B3. AH volume share
- **What**: `ah_volume / total_day_volume`, rolled 5/20d. Persistent elevated AH share around a
  base = institutional interest.
- **Why**: the snapshot already captures AH volume (spotty — some tickers report 0); volume
  concentration after hours is a distinct signal from price.
- **Implementation**: extend the AH snapshot to persist volume reliably; compute share in the
  feature builder.
- **Verify**: coverage check first (how many tickers have non-zero AH volume ≥ 80% of nights),
  then IC screen.
- **Effort**: M.

### B4. Cross-sectional AH breadth (market-state feature)
- **What**: % of universe with positive AH returns tonight, plus its 5d z-score. Feeds every model
  as a market-level feature.
- **Why**: a next-morning regime input available *before* the regular session — 14 hours old
  instead of the regime meter's ~4-week lag (ISSUE-067).
- **Implementation**: computed from the same AH history as B2.
- **Verify**: correlation with next-day regime outcomes; IC as a market feature across folds.
- **Effort**: S once B2's history exists.

### B5. AH earnings reaction decay
- **What**: AH-session earnings move, and the decay of the post-earnings gap over 5/10d.
  `last_earnings_gap_pct` exists for RTH; add the AH leg and `gap_decay_5d`.
- **Why**: persistent post-earnings drift = the swing candidate; fast fade = the trap. Splits the
  existing earnings-gap feature into durable and transient components.
- **Implementation**: needs AH prices on earnings dates (mostly in the AH history from B2).
- **Verify**: IC screen on the decay feature; check the drift names' 20/60d forward beat rate.
- **Effort**: M.

---

## C. Label / target engineering

### C1. Rank-percentile label
- **What**: predict the cross-sectional percentile of forward 60d sector-relative alpha instead of
  raw alpha.
- **Why**: the gate metric is Spearman — a rank label optimizes exactly what is gated. Also likely
  fixes the degenerate calibration (live `calibrated_p_beat_sector` was a single constant 0.4437
  for all 676 rows on 09-28 — the score distribution is collapsed).
- **Implementation**: label transformation in the training pipeline; target column
  `alpha_vs_sector_60d_pct_rank`.
- **Verify**: full walk-forward; compare Spearman and calibration spread vs current.
- **Effort**: S–M.

### C2. Path-quality labels
- **What**: label on `mfe_20d / mae_20d` (best excursion before worst drawdown) or time-to-peak,
  using the path columns already persisted (`mfe_20d`, `mae_20d`, `path_return_20d`).
- **Why**: a model that picks names which rally early and smoothly is more valuable for swing
  entries than terminal-alpha maximization, and it diversifies against the terminal-alpha models
  that all collapsed together.
- **Implementation**: new target column from existing path data.
- **Verify**: walk-forward + gate windows on the path label; bakeoff vs the alpha label.
- **Effort**: S.

### C3. Binary beat label
- **What**: predict P(beat sector by ≥ 2%) — a classification label matching the gate's hit/beat
  floors.
- **Why**: three of the four gate families (hit rate, beat rate, mean excess) are threshold
  statistics; the objective should be too. Continuous regression on a heavy-tailed target is
  pulling the models toward the mean, which is exactly the collapse the forensic describes.
- **Implementation**: thresholded label + log-loss models for the linear candidates.
- **Verify**: compare calibrated hit/beat on fold windows vs the regression versions.
- **Effort**: M.

---

## D. New signal families

### D1. Options market (IV percentile, skew, IV-crush avoidance)
- **What**: IV rank (current IV vs 252d), 25Δ put-call skew, and a post-earnings IV-crush flag.
- **Why**: options markets price upcoming event risk and positioning; IV rank is one of the
  strongest standalone cross-sectional signals in the literature. Zero correlation with the current
  technical block.
- **Data**: **not present** — needs a source decision (Polygon/ORATS/yfinance snapshotting). This
  is data work first, feature work second.
- **Verify**: IC screen once history exists.
- **Effort**: L (data pipeline dominates).

### D2. Estimate revision acceleration
- **What**: the 2-week *change* in analyst revision breadth and in EPS estimate dispersion.
- **Why**: you already have revision breadth (level). The moment dispersion collapses and revisions
  accelerate is the rerating trigger — the level feature alone misses the timing.
- **Implementation**: deltas of existing analyst columns in the snapshot history.
- **Verify**: IC screen; check the signal's hit rate in the 20d after an acceleration event.
- **Effort**: S.

### D3. Failed-breakout flag
- **What**: broke above the 20d/52w high then closed back inside within N days.
- **Why**: the "second breakout works" pattern — first breakout clears weak holders, second is the
  trade. A structure feature the current builder doesn't express.
- **Implementation**: pattern flag from OHLC in the feature builder.
- **Verify**: forward 20/60d beat rate conditioned on the flag vs unconditioned.
- **Effort**: S.

### D4. Buyback / flow data
- **What**: net payout yield (buybacks + dividends), announced buyback programs, insider buying.
- **Why**: a slow, fundamental accumulation signal, additive to the momentum block and negatively
  correlated with its failures.
- **Data**: needs a source (yfinance has some; OpenInsider scrapeable).
- **Effort**: M–L.

### D5. Cross-asset betas for macro-sensitive names
- **What**: energy names vs WTI futures 20d beta-residual; utilities/REITs vs 10y yield changes;
  gold miners vs spot gold.
- **Why**: the label is sector-relative, but these sectors price off macro factors first. A stock
  beating its sector while its macro driver moves is a purer signal than raw RS.
- **Implementation**: fetch the futures/rates series nightly; compute rolling beta residuals per
  ticker.
- **Effort**: M.

### D6. Index-inclusion event calendar
- **What**: S&P 500 add/delete announcements (and Russell recon season).
- **Why**: near-certain 20-60d drift on inclusion; pure event signal, zero correlation with the
  technical block. Event candidates (`event_signal`) already have a home for this.
- **Data**: announcements are public; needs a scraper or feed.
- **Effort**: M.

---

## E. Model architecture

### E1. Deliberate niche split (breaks the clones)
- **What**: rebuild the roster as orthogonal specialists — earnings-only, volume-only,
  analyst-only, technical-only, AH-only — instead of 11 models on one feature pool.
- **Why**: ISSUE-074: the roster is ~4 independent bets. Orthogonal feature sets per model are the
  direct fix, and the gate's choice set becomes genuinely diverse.
- **Implementation**: split `MODEL_FEATURE_COLUMNS` into disjoint profiles; one candidate model per
  profile.
- **Verify**: pairwise pick-overlap on trailing folds drops below ~50% between candidates; gate
  inputs stop aliasing.
- **Effort**: M.

### E2. Pairwise / learning-to-rank loss
- **What**: train with a pairwise (A > B) loss on same-date pairs instead of regression on alpha.
- **Why**: optimizes Spearman directly — the metric every downstream gate enforces. The regression
  objective is a proxy that has measurably stopped delivering.
- **Implementation**: lightGBM/XGBoost pairwise objective on the existing feature matrix.
- **Verify**: full-OOS Spearman vs the regression twins in a bakeoff.
- **Effort**: M.

### E3. Per-sector models (forensic-supported)
- **What**: sector-specific models (or sector embeddings in one model) instead of a global ranker.
- **Why**: the old +0.46 edge was sector-specific xgboost. Sectors differ in drift, vol, and event
  structure; a global ranker averages those differences away. The single biggest structural change
  the forensic points to.
- **Implementation**: `sector_specific` scope existed in the baseline config — revive it as a
  research variant.
- **Verify**: per-sector fold Spearman vs global on the same dates.
- **Effort**: M–L.

### E4. Sequence models over trailing windows
- **What**: a lightweight transformer/attention model over each stock's trailing 20-60d normalized
  price/volume window.
- **Why**: current features are point-in-time snapshots; bases, dry-ups, and breakouts are shapes.
  A sequence model can express "tight base" directly instead of via proxies
  (`base_range_pct_20` et al.).
- **Implementation**: new candidate model with a small sequence encoder; reuse the walk-forward
  scaffold.
- **Verify**: bakeoff vs xgboost on identical dates; watch train-time cost.
- **Effort**: L.

### E5. Meta-labeling confidence filter
- **What**: a second model predicting whether the primary rank will be *correct* this week
  (features: regime, sector, feature-state, trailing model accuracy).
- **Why**: an adaptive answer to the "skill recently" gate demand — suppresses the ranker when it
  is out of regime instead of failing the whole roster.
- **Implementation**: train on historical rank-vs-outcome pairs; filter or weight live picks.
- **Verify**: does the filter's suppression coincide with the fold windows that fail the gate?
  A good filter should have flagged the current standoff in advance.
- **Effort**: M.

---

## F. Universe / deployment (makes any signal tradeable)

### F1. Fix ISSUE-017 for real
- **What**: strategy slots or an `ALL`-sector fallback so champion ranks 1/2/6/8/9/10 stop being
  silently undeployable (only 5 sectors have strategies today).
- **Why**: any signal work is worthless while the top of the list can't be bought. The diagnostic
  half shipped 09-25; the elimination half didn't.
- **Implementation**: `_build_model_strategy_map` fallback in `src/scan/service.py`.
- **Effort**: S.

### F2. Sector-neutral top-N
- **What**: pick the top rank per sector instead of global top-2.
- **Why**: matches the sector-relative label, kills single-sector concentration, and uses more of
  the rank signal.
- **Implementation**: selection-layer change; bakeoff at the promotion basket level first.
- **Effort**: S.

---

## Run-first shortlist

1. **A1** (forensic replication) + **A2** (dual-horizon) — resolves the standoff's root question
   in one bakeoff.
2. **B1** (overnight decomposition) + **B2** (AH-RTH divergence) — pure computation, no new data
   source, directly the AH ask.
3. **C1** (rank label) — aligns the objective with the gate for free.
4. **A4** (regime-interacted features) — ends the flip/inversion saga structurally.

## Data prerequisites (do before feature work)

- **AH history table**: start persisting nightly AH prices + volumes per ticker (B2/B3/B4/B5 all
  need it; backfill only from whatever the snapshot files retain).
- **Options source decision**: needed for D1; not needed for anything else.
- **Macro series**: WTI, 10y, spot gold, for D5.

## Open questions

- Does the run-65 config still produce skill today (A1)? This gates everything else's priority.
- Is the calibration collapse (constant `calibrated_p_beat_sector`) a label-shape problem (C1/C3)
  or a feature-collapse problem? Diagnose before choosing.
- AH volume coverage: what fraction of the 255-ticker universe has reliable non-zero AH volume on
  ≥ 80% of nights? Decides B3's viability.
