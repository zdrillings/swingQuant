# Shortlist Sector Reactivation Analysis

- generated_at: 2026-06-02T20:08:39+00:00
- run_generated_at: 2026-06-02T20:08:10+00:00
- selected_model_name: xgboost_model
- eligible_universe_mode: passed_or_trend
- model_scope: sector_specific
- xgboost_config: balanced_depth4
- active_sectors: Energy, Industrials, Materials
- candidate_sectors: Information Technology
- recent_oos_dates: 60

## Step 1: Sector Admission Comparison

Compare today's active-sector shortlist set against the same model with the candidate sector admitted.

### raw_top_n
- baseline_active: full_mean_target=0.054331, full_beat_universe=0.684375, recent_mean_target=0.069768, recent_beat_universe=0.750000, live_max_sector_share=0.666667
  sectors: Energy, Industrials, Materials
  live_sector_mix: Materials 67%, Energy 33%
  live_picks: CE (Materials), CC (Materials), AMR (Materials), CF (Materials), AESI (Energy), TALO (Energy)
- baseline_plus_candidate: full_mean_target=0.083856, full_beat_universe=0.737500, recent_mean_target=0.145004, recent_beat_universe=0.816667, live_max_sector_share=0.666667
  sectors: Energy, Industrials, Materials, Information Technology
  live_sector_mix: Information Technology 67%, Materials 33%
  live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), INTC (Information Technology), CC (Materials)
- candidate_only: full_mean_target=0.096314, full_beat_universe=0.798479, recent_mean_target=0.165034, recent_beat_universe=0.916667, live_max_sector_share=1.000000
  sectors: Information Technology
  live_sector_mix: Information Technology 100%
  live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), INTC (Information Technology), SLAB (Information Technology), TDC (Information Technology)

### sector_cap_3
- baseline_active: full_mean_target=0.049847, full_beat_universe=0.640625, recent_mean_target=0.063612, recent_beat_universe=0.733333, live_max_sector_share=0.500000
  sectors: Energy, Industrials, Materials
  live_sector_mix: Materials 50%, Energy 33%, Industrials 17%
  live_picks: CE (Materials), CC (Materials), AMR (Materials), AESI (Energy), TALO (Energy), ST (Industrials)
- baseline_plus_candidate: full_mean_target=0.075066, full_beat_universe=0.709375, recent_mean_target=0.112056, recent_beat_universe=0.800000, live_max_sector_share=0.500000
  sectors: Energy, Industrials, Materials, Information Technology
  live_sector_mix: Information Technology 50%, Materials 50%
  live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), CC (Materials), AMR (Materials)
- candidate_only: full_mean_target=0.096037, full_beat_universe=0.722433, recent_mean_target=0.165357, recent_beat_universe=0.770833, live_max_sector_share=1.000000
  sectors: Information Technology
  live_sector_mix: Information Technology 100%
  live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology)

## Step 2: Expanded Set Allocation Balance

Within the expanded sector set, compare diversification policies at the current shortlist size.

- top_n: 6

### raw_top_n
- full_mean_target: 0.083856
- full_beat_universe: 0.737500
- recent_mean_target: 0.145004
- recent_beat_universe: 0.816667
- live_max_sector_share: 0.666667
- live_sector_mix: Information Technology 67%, Materials 33%
- live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), INTC (Information Technology), CC (Materials)

### sector_cap_3
- full_mean_target: 0.075066
- full_beat_universe: 0.709375
- recent_mean_target: 0.112056
- recent_beat_universe: 0.800000
- live_max_sector_share: 0.500000
- live_sector_mix: Information Technology 50%, Materials 50%
- live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), CC (Materials), AMR (Materials)

### sector_cap_2
- full_mean_target: 0.064098
- full_beat_universe: 0.696875
- recent_mean_target: 0.095068
- recent_beat_universe: 0.750000
- live_max_sector_share: 0.333333
- live_sector_mix: Information Technology 33%, Materials 33%, Energy 33%
- live_picks: MXL (Information Technology), SMCI (Information Technology), CE (Materials), CC (Materials), AESI (Energy), TALO (Energy)

### sector_round_robin
- full_mean_target: 0.064529
- full_beat_universe: 0.696875
- recent_mean_target: 0.101058
- recent_beat_universe: 0.800000
- live_max_sector_share: 0.333333
- live_sector_mix: Information Technology 33%, Materials 33%, Energy 17%, Industrials 17%
- live_picks: MXL (Information Technology), SMCI (Information Technology), CE (Materials), CC (Materials), AESI (Energy), ST (Industrials)

## Step 3: Expanded Set Top-N Sensitivity

Hold the standard sector cap policy constant and test whether widening beyond 6 names improves balance.

### top_n=6
- full_mean_target: 0.075066
- full_beat_universe: 0.709375
- recent_mean_target: 0.112056
- recent_beat_universe: 0.800000
- live_max_sector_share: 0.500000
- live_sector_mix: Information Technology 50%, Materials 50%
- live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), CC (Materials), AMR (Materials)

### top_n=8
- full_mean_target: 0.065432
- full_beat_universe: 0.721875
- recent_mean_target: 0.101328
- recent_beat_universe: 0.816667
- live_max_sector_share: 0.375000
- live_sector_mix: Information Technology 38%, Materials 38%, Energy 25%
- live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), CC (Materials), AMR (Materials), AESI (Energy), TALO (Energy)

### top_n=10
- full_mean_target: 0.056539
- full_beat_universe: 0.715625
- recent_mean_target: 0.090351
- recent_beat_universe: 0.833333
- live_max_sector_share: 0.300000
- live_sector_mix: Information Technology 30%, Materials 30%, Energy 30%, Industrials 10%
- live_picks: MXL (Information Technology), SMCI (Information Technology), GEN (Information Technology), CE (Materials), CC (Materials), AMR (Materials), AESI (Energy), TALO (Energy), ST (Industrials), CHRD (Energy)
