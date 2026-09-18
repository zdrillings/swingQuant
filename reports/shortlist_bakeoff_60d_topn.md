# 60d Shortlist Bakeoff: Position Count Sweep

- target_column: alpha_vs_sector_60d
- eligible_universe_mode: passed_or_trend
- test_date_range: 2025-06-16 -> 2026-06-18
- test_dates: 242
- recent_window: 120 test dates
- note: scan candidate history does not yet include alpha_vs_sector_60d, so this compares universe-model policy results only.

## Signal Proxy Results

| top_n | full_mean_target | full_hit_rate | full_beat_universe | recent_mean_target | recent_hit_rate | recent_beat_universe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.146160 | 0.599174 | 0.595041 | 0.106258 | 0.583333 | 0.591667 |
| 2 | 0.156087 | 0.617769 | 0.623967 | 0.127553 | 0.591667 | 0.616667 |
| 3 | 0.147452 | 0.597796 | 0.623967 | 0.115538 | 0.563889 | 0.625000 |
| 4 | 0.149966 | 0.603306 | 0.677686 | 0.117126 | 0.562500 | 0.633333 |
| 6 | 0.136521 | 0.588154 | 0.706612 | 0.101246 | 0.543056 | 0.633333 |

## Readout

- top_2 has the strongest alpha profile: 15.61% full-window mean sector alpha and 12.76% recent-window mean sector alpha.
- top_4 is the diversification compromise: still 11.71% recent mean sector alpha, with stronger full-window universe-beat breadth than top_2.
- top_6 improves full-window breadth but dilutes recent mean alpha and hit rate.
- lasso_model was weaker than signal_proxy across this sweep; the best lasso run was top_6 with 3.55% full mean target and 1.09% recent mean target.

## Caveat

This is not a promoted production champion. It is a 60d fixed-horizon policy bakeoff after adding/backfilling alpha_vs_sector_60d labels. A full shortlist-model gate run is still needed before using a 60d model in production scan selection.
