# Scan Analysis

- scan_date: 2026-05-11
- refreshed: no
- candidate_count: 32
- selected_count: 6
- min_opportunity_score: 0.55

## Selection Summary

### Eligible universe
- count: 32
- avg_opportunity_score: 0.6031
- avg_signal_score: 34.5893
- avg_setup_quality_score: 0.7289
- avg_expected_alpha_score: 0.7368
- avg_freshness_score: 0.7799
- avg_breadth_score: 0.7234
- avg_overlap_penalty: 0.1375

### Selected ideas
- count: 6
- avg_opportunity_score: 0.6745
- avg_signal_score: 34.5119
- avg_setup_quality_score: 0.7411
- avg_expected_alpha_score: 0.7928
- avg_freshness_score: 0.8220
- avg_breadth_score: 0.6190
- avg_overlap_penalty: 0.0800

### Top raw opportunity ideas
- count: 6
- avg_opportunity_score: 0.7043
- avg_signal_score: 37.1927
- avg_setup_quality_score: 0.7748
- avg_expected_alpha_score: 0.7557
- avg_freshness_score: 0.8275
- avg_breadth_score: 0.8061
- avg_overlap_penalty: 0.0800


## Threshold Diagnostics

- threshold: 0.55
- count_above_threshold_before_overlap: 32
- count_above_threshold_after_overlap: 30
- overlap_blocked_count: 2
- already_owned_count: 2

### Top Overlap-Blocked Candidates
### APA
- slot: energy
- sector: Energy
- pre_penalty_opportunity_score: 0.8163
- overlap_penalty: 1.0000
- post_penalty_opportunity_score: -0.1837

### CC
- slot: materials
- sector: Materials
- pre_penalty_opportunity_score: 0.6928
- overlap_penalty: 1.0000
- post_penalty_opportunity_score: -0.3072

## Slot Summary

### energy
- strategy_sector: Energy
- eligible_count: 25
- selected_count: 3
- avg_opportunity: 0.6375
- avg_signal_score: 35.3555

### materials
- strategy_sector: Materials
- eligible_count: 6
- selected_count: 3
- avg_opportunity: 0.4630
- avg_signal_score: 31.7566

### industrials
- strategy_sector: Industrials
- eligible_count: 1
- selected_count: 0
- avg_opportunity: 0.5843
- avg_signal_score: 32.4303

## Owned Strength Watchlist

### APA
- status: already owned, setup still valid
- slot: energy
- sector: Energy
- pre_penalty_opportunity_score: 0.8163
- post_penalty_opportunity_score: -0.1837
- overlap_penalty: 1.0000
- overlap_components: same_slot=0.08, same_ticker=1.00

### CC
- status: already owned, setup still valid
- slot: materials
- sector: Materials
- pre_penalty_opportunity_score: 0.6928
- post_penalty_opportunity_score: -0.3072
- overlap_penalty: 1.0000
- overlap_components: same_slot=0.08, same_ticker=1.00

## Learned Buy Review

- target_column: alpha_vs_sector_10d
- train_rows: 13076
- train_dates: 526
- learned_selected_count: 6
- runtime_selected_count: 6
- overlap_with_runtime_selected: 4/6
- runtime_selected_mean_predicted_alpha: 0.045983
- runtime_rejected_mean_predicted_alpha: 0.029741

### Top Learned Buys
#### CF
- learned_rank: 2
- runtime_selected_rank: 4
- runtime_selected: yes
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.069786
- opportunity_score: 0.6530
- signal_score: 32.2863
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0311), avg_abs_gap_pct_20 (+0.0107), base_atr_contraction_20 (+0.0083)
- ranker_negative_reasons: signal_score (-0.0263), candidate_count_slot_day (-0.0019), candidate_count_sector_day (-0.0019)
- overlap_components: same_slot=0.08

#### PARR
- learned_rank: 3
- runtime_selected_rank: 1
- runtime_selected: yes
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.054688
- opportunity_score: 0.7050
- signal_score: 37.4646
- ranker_positive_reasons: signal_score (+0.0366), max_gap_down_pct_60 (+0.0352), avg_abs_gap_pct_20 (+0.0201)
- ranker_negative_reasons: setup_quality_score (-0.0360), sector_pct_above_200 (-0.0116), sma_200_dist (-0.0093)
- overlap_components: same_slot=0.08

#### SM
- learned_rank: 5
- runtime_selected: no
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.050394
- opportunity_score: 0.6832
- signal_score: 35.5474
- runtime_exclusion_reason: portfolio cap / overlap selection loss
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0221), signal_score (+0.0133), avg_abs_gap_pct_20 (+0.0122)
- ranker_negative_reasons: sector_pct_above_200 (-0.0116), setup_quality_score (-0.0093), rsi_14 (-0.0031)
- overlap_components: same_slot=0.08

#### CHRD
- learned_rank: 6
- runtime_selected: no
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.048227
- opportunity_score: 0.6861
- signal_score: 35.1586
- runtime_exclusion_reason: portfolio cap / overlap selection loss
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0173), avg_abs_gap_pct_20 (+0.0124), signal_score (+0.0085)
- ranker_negative_reasons: sector_pct_above_200 (-0.0116), setup_quality_score (-0.0039), rsi_14 (-0.0030)
- overlap_components: same_slot=0.08

#### LYB
- learned_rank: 7
- runtime_selected_rank: 5
- runtime_selected: yes
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.045154
- opportunity_score: 0.6466
- signal_score: 31.6106
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0162), setup_quality_score (+0.0161), base_atr_contraction_20 (+0.0084)
- ranker_negative_reasons: signal_score (-0.0345), candidate_count_slot_day (-0.0019), candidate_count_sector_day (-0.0019)
- overlap_components: same_slot=0.08

#### DOW
- learned_rank: 8
- runtime_selected_rank: 6
- runtime_selected: yes
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.043593
- opportunity_score: 0.6257
- signal_score: 31.7414
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0221), setup_quality_score (+0.0142), avg_abs_gap_pct_20 (+0.0128)
- ranker_negative_reasons: signal_score (-0.0330), roc_63 (-0.0040), sma_200_dist (-0.0040)
- overlap_components: same_slot=0.08

### Best Learned Rejections
#### SM
- learned_rank: 5
- runtime_selected: no
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.050394
- opportunity_score: 0.6832
- signal_score: 35.5474
- runtime_exclusion_reason: portfolio cap / overlap selection loss
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0221), signal_score (+0.0133), avg_abs_gap_pct_20 (+0.0122)
- ranker_negative_reasons: sector_pct_above_200 (-0.0116), setup_quality_score (-0.0093), rsi_14 (-0.0031)
- overlap_components: same_slot=0.08

#### CHRD
- learned_rank: 6
- runtime_selected: no
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.048227
- opportunity_score: 0.6861
- signal_score: 35.1586
- runtime_exclusion_reason: portfolio cap / overlap selection loss
- ranker_positive_reasons: max_gap_down_pct_60 (+0.0173), avg_abs_gap_pct_20 (+0.0124), signal_score (+0.0085)
- ranker_negative_reasons: sector_pct_above_200 (-0.0116), setup_quality_score (-0.0039), rsi_14 (-0.0030)
- overlap_components: same_slot=0.08

### Runtime Overrides
#### CRGY
- learned_rank: 15
- runtime_selected_rank: 2
- runtime_selected: yes
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.031757
- opportunity_score: 0.7178
- signal_score: 37.7768
- ranker_positive_reasons: signal_score (+0.0403), max_gap_down_pct_60 (+0.0134), avg_abs_gap_pct_20 (+0.0123)
- ranker_negative_reasons: setup_quality_score (-0.0403), sector_pct_above_200 (-0.0116), roc_63 (-0.0036)
- overlap_components: same_slot=0.08

#### OVV
- learned_rank: 16
- runtime_selected_rank: 3
- runtime_selected: yes
- slot: energy
- sector: Energy
- predicted_alpha_vs_sector_10d: 0.030922
- opportunity_score: 0.6987
- signal_score: 36.1920
- ranker_positive_reasons: signal_score (+0.0211), max_gap_down_pct_60 (+0.0087), avg_abs_gap_pct_20 (+0.0079)
- ranker_negative_reasons: setup_quality_score (-0.0183), sector_pct_above_200 (-0.0116), rsi_14 (-0.0028)
- overlap_components: same_slot=0.08

## Slot-Level Selector Attribution

- target_column: alpha_vs_sector_10d
- train_rows: 7634
- train_dates: 368
- validation_dates: 158

### energy
- learned: mean_target=0.023565 hit_rate=0.6456 avg_pick_count=3.00 validation_days=158
- handcrafted: mean_target=0.004765 hit_rate=0.5612 avg_pick_count=3.00 validation_days=158
- runtime: mean_target=0.003204 hit_rate=0.5492 avg_pick_count=2.42 validation_days=158

### industrials
- learned: mean_target=0.012331 hit_rate=0.5417 avg_pick_count=2.89 validation_days=156
- handcrafted: mean_target=0.007451 hit_rate=0.5032 avg_pick_count=2.89 validation_days=156
- runtime: mean_target=0.005326 hit_rate=0.4988 avg_pick_count=2.06 validation_days=156

### materials
- learned: mean_target=0.034978 hit_rate=0.6346 avg_pick_count=2.93 validation_days=156
- handcrafted: mean_target=0.026943 hit_rate=0.6218 avg_pick_count=2.93 validation_days=156
- runtime: mean_target=0.040776 hit_rate=0.6797 avg_pick_count=1.54 validation_days=156

## Selected Candidates

### PARR
- selected_rank: 1
- slot: energy
- sector: Energy
- opportunity_score: 0.7050
- signal_score: 37.4646
- setup_quality_score: 0.7805
- expected_alpha_score: 0.8334
- freshness_score: 0.7047
- breadth_score: 0.8061
- overlap_penalty: 0.0800

### CRGY
- selected_rank: 2
- slot: energy
- sector: Energy
- opportunity_score: 0.7178
- signal_score: 37.7768
- setup_quality_score: 0.7870
- expected_alpha_score: 0.7565
- freshness_score: 0.8726
- breadth_score: 0.8061
- overlap_penalty: 0.0800

### OVV
- selected_rank: 3
- slot: energy
- sector: Energy
- opportunity_score: 0.6987
- signal_score: 36.1920
- setup_quality_score: 0.7540
- expected_alpha_score: 0.7984
- freshness_score: 0.7718
- breadth_score: 0.8061
- overlap_penalty: 0.0800

### CF
- selected_rank: 4
- slot: materials
- sector: Materials
- opportunity_score: 0.6530
- signal_score: 32.2863
- setup_quality_score: 0.7175
- expected_alpha_score: 0.8131
- freshness_score: 0.8658
- breadth_score: 0.4320
- overlap_penalty: 0.0800

### LYB
- selected_rank: 5
- slot: materials
- sector: Materials
- opportunity_score: 0.6466
- signal_score: 31.6106
- setup_quality_score: 0.7025
- expected_alpha_score: 0.8162
- freshness_score: 0.8553
- breadth_score: 0.4320
- overlap_penalty: 0.0800

### DOW
- selected_rank: 6
- slot: materials
- sector: Materials
- opportunity_score: 0.6257
- signal_score: 31.7414
- setup_quality_score: 0.7054
- expected_alpha_score: 0.7390
- freshness_score: 0.8618
- breadth_score: 0.4320
- overlap_penalty: 0.0800

## Best Excluded Candidates

### PR
- slot: energy
- sector: Energy
- opportunity_score: 0.7079
- signal_score: 37.4995
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### PBF
- slot: energy
- sector: Energy
- opportunity_score: 0.6996
- signal_score: 36.2583
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### OXY
- slot: energy
- sector: Energy
- opportunity_score: 0.6968
- signal_score: 37.9648
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### MUR
- slot: energy
- sector: Energy
- opportunity_score: 0.6960
- signal_score: 37.8977
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### FANG
- slot: energy
- sector: Energy
- opportunity_score: 0.6882
- signal_score: 36.9057
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### MTDR
- slot: energy
- sector: Energy
- opportunity_score: 0.6881
- signal_score: 37.0922
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### CHRD
- slot: energy
- sector: Energy
- opportunity_score: 0.6861
- signal_score: 35.1586
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### SM
- slot: energy
- sector: Energy
- opportunity_score: 0.6832
- signal_score: 35.5474
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### TDW
- slot: energy
- sector: Energy
- opportunity_score: 0.6831
- signal_score: 33.3030
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

### EOG
- slot: energy
- sector: Energy
- opportunity_score: 0.6792
- signal_score: 36.8423
- overlap_penalty: 0.0800
- exclusion_reason: portfolio cap / overlap selection loss
- overlap_components: same_slot=0.08

## Outcome Coverage

- 5d: forward=0 sector_alpha=0 spy_alpha=0
- 10d: forward=0 sector_alpha=0 spy_alpha=0
- 20d: forward=0 sector_alpha=0 spy_alpha=0
- mfe_20d_available: 0
- mae_20d_available: 0

## Forward Attribution

No forward return windows are fully available yet for this scan date.

## Candidate Ranker Validation

- target_column: alpha_vs_sector_10d
- train_rows: 7634
- validation_rows: 5442
- train_dates: 368
- validation_dates: 158
- feature_count: 43
- prediction_correlation: 0.116673

### Validation Averages
- learned_mean_target: 0.028037
- learned_hit_rate: 0.624578
- handcrafted_mean_target: 0.011667
- handcrafted_hit_rate: 0.556857
- runtime_mean_target: 0.011667
- runtime_hit_rate: 0.556857

### Latest Validation Date: 2026-04-24
- learned_tickers: SM, PARR, DINO, CF, DOW, LYB
- handcrafted_tickers: COP, PR, OII, WLK, LYB, CAR
- runtime_tickers: COP, PR, OII, WLK, LYB, CAR
