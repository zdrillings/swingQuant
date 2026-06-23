# Scan Analysis

- scan_date: 2026-06-11
- refreshed: no
- candidate_count: 240
- selected_count: 6
- min_opportunity_score: 0.55

## Selection Summary

### Eligible universe
- count: 240
- avg_opportunity_score: 0.3458
- avg_signal_score: 0.0000
- avg_setup_quality_score: 0.0000
- avg_expected_alpha_score: 0.7017
- avg_freshness_score: 0.6551
- avg_breadth_score: 0.6397
- avg_overlap_penalty: 0.0917

### Selected ideas
- count: 6
- avg_opportunity_score: 0.2706
- avg_signal_score: 0.0000
- avg_setup_quality_score: 0.0000
- avg_expected_alpha_score: 0.3429
- avg_freshness_score: 0.7694
- avg_breadth_score: 0.6258
- avg_overlap_penalty: 0.0800

### Top raw opportunity ideas
- count: 6
- avg_opportunity_score: 0.4621
- avg_signal_score: 0.0000
- avg_setup_quality_score: 0.0000
- avg_expected_alpha_score: 0.7795
- avg_freshness_score: 0.8181
- avg_breadth_score: 0.5201
- avg_overlap_penalty: 0.0133


## Threshold Diagnostics

- threshold: 0.55
- count_above_threshold_before_overlap: 0
- count_above_threshold_after_overlap: 0
- overlap_blocked_count: 0
- already_owned_count: 6

## Active Slot Coverage

### energy
- strategy_sector: Energy
- eligible_count: 34
- selected_count: 0
- status: candidates existed, but the slot was crowded out by higher-ranked names
- top_signal_names: SLB, WHD, KGS

### industrials
- strategy_sector: Industrials
- eligible_count: 69
- selected_count: 0
- status: candidates existed, but the slot was crowded out by higher-ranked names
- top_signal_names: FIX, AGX, WCC

### materials
- strategy_sector: Materials
- eligible_count: 23
- selected_count: 3
- status: slot contributed picks, but some eligible names lost in final ranking
- top_signal_names: ESI, KALU, CBT

### technology
- strategy_sector: Information Technology
- eligible_count: 114
- selected_count: 3
- status: slot contributed picks, but some eligible names lost in final ranking
- top_signal_names: MCHP, FORM, MPWR

## Live Slot Gate Waterfall

### energy
- strategy_sector: Energy
- gate_counts: regime_green=240, sector_scope=34, model_scored=34
- first_zero_gate: none
- component_positive_counts: unavailable

### industrials
- strategy_sector: Industrials
- gate_counts: regime_green=240, sector_scope=69, model_scored=69
- first_zero_gate: none
- component_positive_counts: unavailable

### materials
- strategy_sector: Materials
- gate_counts: regime_green=240, sector_scope=23, model_scored=23
- first_zero_gate: none
- component_positive_counts: unavailable

### technology
- strategy_sector: Information Technology
- gate_counts: regime_green=240, sector_scope=114, model_scored=114
- first_zero_gate: none
- component_positive_counts: unavailable

## Live Post-Gate Dropoff

- min_opportunity_score: 0.55

### energy
- strategy_sector: Energy
- gated_count: 34
- cleared_opportunity_count: 0
- dropped_after_opportunity_count: 34
- avg_gated_opportunity_score: 0.3880
- top_cleared: none
- top_dropped: SLB, WHD, KGS
- dropped: SLB opportunity=0.4815 signal=0.0000
- dropped: WHD opportunity=0.4611 signal=0.0000
- dropped: KGS opportunity=0.4609 signal=0.0000

### industrials
- strategy_sector: Industrials
- gated_count: 69
- cleared_opportunity_count: 0
- dropped_after_opportunity_count: 69
- avg_gated_opportunity_score: 0.3593
- top_cleared: none
- top_dropped: FIX, AGX, WCC
- dropped: FIX opportunity=0.4266 signal=0.0000
- dropped: AGX opportunity=0.4246 signal=0.0000
- dropped: WCC opportunity=0.4183 signal=0.0000

### materials
- strategy_sector: Materials
- gated_count: 23
- cleared_opportunity_count: 0
- dropped_after_opportunity_count: 23
- avg_gated_opportunity_score: 0.2526
- top_cleared: none
- top_dropped: ESI, KALU, CBT
- dropped: ESI opportunity=0.3872 signal=0.0000
- dropped: KALU opportunity=0.3741 signal=0.0000
- dropped: CBT opportunity=0.3517 signal=0.0000

### technology
- strategy_sector: Information Technology
- gated_count: 114
- cleared_opportunity_count: 0
- dropped_after_opportunity_count: 114
- avg_gated_opportunity_score: 0.3439
- top_cleared: none
- top_dropped: MCHP, FORM, MPWR
- dropped: MCHP opportunity=0.4513 signal=0.0000
- dropped: FORM opportunity=0.4502 signal=0.0000
- dropped: MPWR opportunity=0.4476 signal=0.0000

## Slot Summary

### technology
- strategy_sector: Information Technology
- eligible_count: 114
- selected_count: 3
- avg_opportunity: 0.3439
- avg_signal_score: 0.0000

### materials
- strategy_sector: Materials
- eligible_count: 23
- selected_count: 3
- avg_opportunity: 0.2526
- avg_signal_score: 0.0000

### energy
- strategy_sector: Energy
- eligible_count: 34
- selected_count: 0
- avg_opportunity: 0.3880
- avg_signal_score: 0.0000

### industrials
- strategy_sector: Industrials
- eligible_count: 69
- selected_count: 0
- avg_opportunity: 0.3593
- avg_signal_score: 0.0000

## Slot Allocation Drivers

### energy
- strategy_sector: Energy
- raw_slot_leaders: SLB, WHD, KGS
- adjusted_slot_leaders: AESI, TALO, CHRD
- overlay_promoted: AESI, TALO, CHRD
- overlay_demoted: SLB, WHD, KGS
- avg_slot_overlay_adjustment: -1.1930

### industrials
- strategy_sector: Industrials
- raw_slot_leaders: FIX, AGX, WCC
- adjusted_slot_leaders: ST, PBI, FIX
- overlay_promoted: ST, PBI
- overlay_demoted: AGX, WCC
- avg_slot_overlay_adjustment: -1.0015

### materials
- strategy_sector: Materials
- raw_slot_leaders: ESI, KALU, CBT
- adjusted_slot_leaders: CE, CC, AMR
- overlay_promoted: CE, CC, AMR
- overlay_demoted: ESI, KALU, CBT
- avg_slot_overlay_adjustment: +1.1634
- cutoff_pair: kept CF over CC

### technology
- strategy_sector: Information Technology
- raw_slot_leaders: MCHP, FORM, MPWR
- adjusted_slot_leaders: MXL, SMCI, GEN
- overlay_promoted: MXL, SMCI, GEN
- overlay_demoted: MCHP, FORM, MPWR
- avg_slot_overlay_adjustment: +0.0000
- cutoff_pair: kept INTC over MXL

## Owned Strength Watchlist

Owned names matched, but none cleared the buy threshold before ownership penalties.

## Portfolio Strength Coverage

- top_6_already_held: 0/6
- top_10_already_held: 0/10
- held_candidates_in_scan: 6
- strongest_held_candidate: ENTG
- strongest_unheld_candidate: MCHP
- open_holdings_not_in_candidate_set: MRVL, RKLB

### Top 10 Candidates
1. MCHP - not held, not selected, score=0.5313, post_penalty_score=0.4513, selection_score=0.0183, sector=Information Technology
2. FORM - not held, not selected, score=0.5302, post_penalty_score=0.4502, selection_score=0.0938, sector=Information Technology
3. MPWR - not held, not selected, score=0.5276, post_penalty_score=0.4476, selection_score=0.0128, sector=Information Technology
4. LITE - not held, not selected, score=0.5243, post_penalty_score=0.4443, selection_score=0.0163, sector=Information Technology
5. RMBS - not held, not selected, score=0.5213, post_penalty_score=0.4413, selection_score=0.0727, sector=Information Technology
6. TXN - not held, not selected, score=0.5183, post_penalty_score=0.4383, selection_score=0.0293, sector=Information Technology
7. CRUS - not held, not selected, score=0.5164, post_penalty_score=0.4364, selection_score=-0.0120, sector=Information Technology
8. TER - not held, not selected, score=0.5163, post_penalty_score=0.4363, selection_score=0.0442, sector=Information Technology
9. GLW - not held, not selected, score=0.5158, post_penalty_score=0.4358, selection_score=0.0733, sector=Information Technology
10. VIAV - not held, not selected, score=0.5153, post_penalty_score=0.4353, selection_score=0.0245, sector=Information Technology

## Learned Buy Review

- target_column: alpha_vs_sector_10d
- train_rows: 62385
- train_dates: 542
- learned_selected_count: 6
- runtime_selected_count: 6
- overlap_with_runtime_selected: 1/6
- runtime_selected_mean_predicted_alpha: 0.071076
- runtime_rejected_mean_predicted_alpha: 0.071120

### Top Learned Buys
#### SNDK
- learned_rank: 7
- runtime_selected: no
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.148796
- opportunity_score: 0.3781
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: atr_14 (+0.0800), opportunity_score (+0.0543), setup_quality_score (+0.0471)
- ranker_negative_reasons: signal_score (-0.0658), adj_close (-0.0398), sma_200_dist (-0.0296)
- overlap_components: same_slot=0.08

#### CIEN
- learned_rank: 8
- runtime_selected: no
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.115494
- opportunity_score: 0.4072
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: setup_quality_score (+0.0471), opportunity_score (+0.0349), sub_industry: Communications Equipment (+0.0258)
- ranker_negative_reasons: signal_score (-0.0658), base_range_pct_20 (-0.0176), candidate_count_sector_day (-0.0112)
- overlap_components: same_slot=0.08

#### DOW
- learned_rank: 9
- runtime_selected: no
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.115010
- opportunity_score: 0.2200
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: opportunity_score (+0.1604), setup_quality_score (+0.0471), relative_strength_index_vs_qqq (+0.0383)
- ranker_negative_reasons: expected_alpha_score (-0.1247), signal_score (-0.0658), relative_strength_index_vs_xlk (-0.0499)
- overlap_components: same_slot=0.08

#### LITE
- learned_rank: 10
- runtime_selected: no
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.112959
- opportunity_score: 0.4443
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: setup_quality_score (+0.0471), atr_14 (+0.0449), expected_alpha_score (+0.0283)
- ranker_negative_reasons: signal_score (-0.0658), adj_close (-0.0168), candidate_count_sector_day (-0.0112)
- overlap_components: same_slot=0.08

#### STRL
- learned_rank: 12
- runtime_selected: no
- slot: industrials
- sector: Industrials
- predicted_alpha_vs_sector_10d: 0.108660
- opportunity_score: 0.3768
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: opportunity_score (+0.0553), setup_quality_score (+0.0471), atr_14 (+0.0362)
- ranker_negative_reasons: signal_score (-0.0658), adj_close (-0.0157), base_range_pct_20 (-0.0124)
- overlap_components: same_slot=0.08

#### CF
- learned_rank: 15
- runtime_selected_rank: 6
- runtime_selected: yes
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.107376
- opportunity_score: 0.1743
- signal_score: 0.0000
- ranker_positive_reasons: opportunity_score (+0.1911), setup_quality_score (+0.0471), relative_strength_index_vs_qqq (+0.0430)
- ranker_negative_reasons: expected_alpha_score (-0.1396), signal_score (-0.0658), relative_strength_index_vs_xlk (-0.0560)
- overlap_components: same_slot=0.08

### Best Learned Rejections
#### SNDK
- learned_rank: 7
- runtime_selected: no
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.148796
- opportunity_score: 0.3781
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: atr_14 (+0.0800), opportunity_score (+0.0543), setup_quality_score (+0.0471)
- ranker_negative_reasons: signal_score (-0.0658), adj_close (-0.0398), sma_200_dist (-0.0296)
- overlap_components: same_slot=0.08

#### CIEN
- learned_rank: 8
- runtime_selected: no
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.115494
- opportunity_score: 0.4072
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: setup_quality_score (+0.0471), opportunity_score (+0.0349), sub_industry: Communications Equipment (+0.0258)
- ranker_negative_reasons: signal_score (-0.0658), base_range_pct_20 (-0.0176), candidate_count_sector_day (-0.0112)
- overlap_components: same_slot=0.08

#### DOW
- learned_rank: 9
- runtime_selected: no
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.115010
- opportunity_score: 0.2200
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: opportunity_score (+0.1604), setup_quality_score (+0.0471), relative_strength_index_vs_qqq (+0.0383)
- ranker_negative_reasons: expected_alpha_score (-0.1247), signal_score (-0.0658), relative_strength_index_vs_xlk (-0.0499)
- overlap_components: same_slot=0.08

#### LITE
- learned_rank: 10
- runtime_selected: no
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.112959
- opportunity_score: 0.4443
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: setup_quality_score (+0.0471), atr_14 (+0.0449), expected_alpha_score (+0.0283)
- ranker_negative_reasons: signal_score (-0.0658), adj_close (-0.0168), candidate_count_sector_day (-0.0112)
- overlap_components: same_slot=0.08

#### STRL
- learned_rank: 12
- runtime_selected: no
- slot: industrials
- sector: Industrials
- predicted_alpha_vs_sector_10d: 0.108660
- opportunity_score: 0.3768
- signal_score: 0.0000
- runtime_exclusion_reason: below opportunity threshold
- ranker_positive_reasons: opportunity_score (+0.0553), setup_quality_score (+0.0471), atr_14 (+0.0362)
- ranker_negative_reasons: signal_score (-0.0658), adj_close (-0.0157), base_range_pct_20 (-0.0124)
- overlap_components: same_slot=0.08

### Runtime Overrides
#### CE
- learned_rank: 25
- runtime_selected_rank: 3
- runtime_selected: yes
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.093886
- opportunity_score: 0.2077
- signal_score: 0.0000
- ranker_positive_reasons: opportunity_score (+0.1687), setup_quality_score (+0.0471), relative_strength_index_vs_qqq (+0.0396)
- ranker_negative_reasons: expected_alpha_score (-0.1491), signal_score (-0.0658), relative_strength_index_vs_xlk (-0.0516)
- overlap_components: same_slot=0.08

#### SMCI
- learned_rank: 70
- runtime_selected_rank: 1
- runtime_selected: yes
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.074845
- opportunity_score: 0.3034
- signal_score: 0.0000
- ranker_positive_reasons: opportunity_score (+0.1045), setup_quality_score (+0.0471), max_gap_down_pct_60 (+0.0407)
- ranker_negative_reasons: expected_alpha_score (-0.0891), signal_score (-0.0658), breakout_volume_ratio_50 (-0.0384)
- overlap_components: same_slot=0.08

#### INTC
- learned_rank: 97
- runtime_selected_rank: 4
- runtime_selected: yes
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.068028
- opportunity_score: 0.4198
- signal_score: 0.0000
- ranker_positive_reasons: expected_alpha_score (+0.0541), setup_quality_score (+0.0471), opportunity_score (+0.0264)
- ranker_negative_reasons: signal_score (-0.0658), sma_200_dist (-0.0135), sub_industry: Semiconductors (-0.0129)
- overlap_components: same_slot=0.08

#### AMR
- learned_rank: 106
- runtime_selected_rank: 5
- runtime_selected: yes
- slot: materials
- sector: Materials
- predicted_alpha_vs_sector_10d: 0.065269
- opportunity_score: 0.2205
- signal_score: 0.0000
- ranker_positive_reasons: opportunity_score (+0.1601), setup_quality_score (+0.0471), relative_strength_index_vs_qqq (+0.0270)
- ranker_negative_reasons: expected_alpha_score (-0.1300), signal_score (-0.0658), relative_strength_index_vs_xlk (-0.0352)
- overlap_components: same_slot=0.08

#### GEN
- learned_rank: 212
- runtime_selected_rank: 2
- runtime_selected: yes
- slot: technology
- sector: Information Technology
- predicted_alpha_vs_sector_10d: 0.017051
- opportunity_score: 0.2981
- signal_score: 0.0000
- ranker_positive_reasons: opportunity_score (+0.1081), setup_quality_score (+0.0471), relative_strength_index_vs_qqq (+0.0107)
- ranker_negative_reasons: expected_alpha_score (-0.0765), signal_score (-0.0658), relative_strength_index_vs_xlk (-0.0139)
- overlap_components: same_slot=0.08

## Slot-Level Selector Attribution

- target_column: alpha_vs_sector_10d
- validation_method: purged_walk_forward
- embargo_days: 10
- validation_blocks: 5
- train_rows: 41444
- train_dates: 435
- validation_dates: 163

### energy
- learned: mean_target=0.007133 hit_rate=0.5544 avg_pick_count=0.73 validation_days=163
- handcrafted: mean_target=-0.004986 hit_rate=0.4728 avg_pick_count=0.73 validation_days=163
- runtime: mean_target=0.040682 hit_rate=0.6905 avg_pick_count=1.37 validation_days=163

### industrials
- learned: mean_target=0.008377 hit_rate=0.5684 avg_pick_count=0.51 validation_days=159
- handcrafted: mean_target=0.005188 hit_rate=0.5684 avg_pick_count=0.51 validation_days=159
- runtime: mean_target=0.051375 hit_rate=0.6729 avg_pick_count=0.67 validation_days=159

### materials
- learned: mean_target=0.008170 hit_rate=0.5299 avg_pick_count=0.56 validation_days=163
- handcrafted: mean_target=-0.011751 hit_rate=0.4444 avg_pick_count=0.56 validation_days=163
- runtime: mean_target=0.061173 hit_rate=0.7261 avg_pick_count=1.75 validation_days=163

### technology
- learned: mean_target=0.080581 hit_rate=0.8333 avg_pick_count=0.08 validation_days=139
- handcrafted: mean_target=0.070221 hit_rate=0.7917 avg_pick_count=0.08 validation_days=139
- runtime: mean_target=0.088891 hit_rate=0.7218 avg_pick_count=2.60 validation_days=139

## Selected Candidates

### SMCI
- selected_rank: 1
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.2987
- model_reason_summary: limited recent downside gap risk, clear of near-term earnings, well above 50d trend
- model_comparison_summary: TDC in Information Technology on clear of near-term earnings and well above 50d trend
- opportunity_score: 0.3034
- selection_score: 0.2987
- raw_opportunity_rank: 208
- adjusted_selection_rank: 2
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- setup_quality_score: 0.0000
- expected_alpha_score: 0.3399
- freshness_score: 0.8409
- breadth_score: 0.7550
- overlap_penalty: 0.0800

### GEN
- selected_rank: 2
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.2738
- model_reason_summary: constructive ATR profile, clear of near-term earnings, supportive atr 14
- model_comparison_summary: TDC in Information Technology on constructive ATR profile and clear of near-term earnings
- opportunity_score: 0.2981
- selection_score: 0.2738
- raw_opportunity_rank: 215
- adjusted_selection_rank: 3
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- setup_quality_score: 0.0000
- expected_alpha_score: 0.3909
- freshness_score: 0.7377
- breadth_score: 0.7550
- overlap_penalty: 0.0800

### CE
- selected_rank: 3
- slot: materials
- sector: Materials
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.2520
- model_reason_summary: strong earnings volume, limited recent downside gap risk, holding above recent breakout
- model_comparison_summary: CF in Materials on strong earnings volume
- opportunity_score: 0.2077
- selection_score: 0.2520
- raw_opportunity_rank: 230
- adjusted_selection_rank: 4
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.8670
- slot_overlay_components: distance_above_20d_high=-0.0000, freshness_score=-1.3791, sector_pct_above_200=+1.2923, sector_pct_above_50=+0.9538, setup_quality_score=-0.0000, signal_score=-0.0000
- signal_score: n/a (model-sourced)
- setup_quality_score: 0.0000
- expected_alpha_score: 0.0976
- freshness_score: 0.9194
- breadth_score: 0.4966
- overlap_penalty: 0.0800

### INTC
- selected_rank: 4
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.2253
- model_reason_summary: well above 200d trend, supportive rsi 14, strong RS vs SPY
- model_comparison_summary: TDC in Information Technology on well above 200d trend and supportive rsi 14
- opportunity_score: 0.4198
- selection_score: 0.2253
- raw_opportunity_rank: 33
- adjusted_selection_rank: 5
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- setup_quality_score: 0.0000
- expected_alpha_score: 0.9184
- freshness_score: 0.5550
- breadth_score: 0.7550
- overlap_penalty: 0.0800

### AMR
- selected_rank: 5
- slot: materials
- sector: Materials
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.1678
- model_reason_summary: strong earnings volume, limited recent downside gap risk, supportive post-earnings hold
- model_comparison_summary: CF in Materials on strong earnings volume and supportive post-earnings hold
- opportunity_score: 0.2205
- selection_score: 0.1678
- raw_opportunity_rank: 228
- adjusted_selection_rank: 8
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.9437
- slot_overlay_components: distance_above_20d_high=-0.0000, freshness_score=-1.3024, sector_pct_above_200=+1.2923, sector_pct_above_50=+0.9538, setup_quality_score=-0.0000, signal_score=-0.0000
- signal_score: n/a (model-sourced)
- setup_quality_score: 0.0000
- expected_alpha_score: 0.1744
- freshness_score: 0.8683
- breadth_score: 0.4966
- overlap_penalty: 0.0800

### CF
- selected_rank: 6
- slot: materials
- sector: Materials
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.1200
- model_reason_summary: holding above earnings close, limited recent downside gap risk, holding above recent breakout
- opportunity_score: 0.1743
- selection_score: 0.1200
- raw_opportunity_rank: 234
- adjusted_selection_rank: 12
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +1.2037
- slot_overlay_components: distance_above_20d_high=-0.0000, freshness_score=-1.0425, sector_pct_above_200=+1.2923, sector_pct_above_50=+0.9538, setup_quality_score=-0.0000, signal_score=-0.0000
- signal_score: n/a (model-sourced)
- setup_quality_score: 0.0000
- expected_alpha_score: 0.1360
- freshness_score: 0.6950
- breadth_score: 0.4966
- overlap_penalty: 0.0800

## Best Excluded Candidates

### SLB
- slot: energy
- sector: Energy
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: -0.0027
- model_reason_summary: supportive atr 14, limited recent downside gap risk, healthy 50d sector breadth
- opportunity_score: 0.4815
- selection_score: -0.0027
- raw_opportunity_rank: 1
- adjusted_selection_rank: 151
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: -1.6994
- slot_overlay_components: distance_above_20d_high=-0.0000, roc_63=-0.7702, sector_pct_above_50=-0.4113, sma_200_dist=-0.5179
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0000
- exclusion_reason: below opportunity threshold

### WHD
- slot: energy
- sector: Energy
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: -0.0114
- model_reason_summary: healthy 50d sector breadth, strong 63d momentum, limited recent downside gap risk
- opportunity_score: 0.4611
- selection_score: -0.0114
- raw_opportunity_rank: 2
- adjusted_selection_rank: 190
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: -1.6445
- slot_overlay_components: distance_above_20d_high=-0.0000, roc_63=-0.8198, sector_pct_above_50=-0.4113, sma_200_dist=-0.4134
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0000
- exclusion_reason: below opportunity threshold

### KGS
- slot: energy
- sector: Energy
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: -0.0459
- model_reason_summary: active price discovery, strong 126d momentum, healthy 50d sector breadth
- opportunity_score: 0.4609
- selection_score: -0.0459
- raw_opportunity_rank: 3
- adjusted_selection_rank: 231
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: -1.9561
- slot_overlay_components: distance_above_20d_high=-0.0000, roc_63=-0.7049, sector_pct_above_50=-0.4113, sma_200_dist=-0.8399
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0000
- exclusion_reason: below opportunity threshold

### DINO
- slot: energy
- sector: Energy
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: -0.0201
- model_reason_summary: healthy 50d sector breadth, strong 63d momentum, supportive base tightness
- opportunity_score: 0.4595
- selection_score: -0.0201
- raw_opportunity_rank: 4
- adjusted_selection_rank: 214
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: -1.5652
- slot_overlay_components: distance_above_20d_high=-0.0000, roc_63=-0.6603, sector_pct_above_50=-0.4113, sma_200_dist=-0.4935
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0000
- exclusion_reason: below opportunity threshold

### SM
- slot: energy
- sector: Energy
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.0396
- model_reason_summary: limited recent downside gap risk, active price discovery, healthy 50d sector breadth
- opportunity_score: 0.4584
- selection_score: 0.0396
- raw_opportunity_rank: 5
- adjusted_selection_rank: 72
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: -1.6487
- slot_overlay_components: distance_above_20d_high=-0.0000, roc_63=-0.6829, sector_pct_above_50=-0.4113, sma_200_dist=-0.5545
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0000
- exclusion_reason: below opportunity threshold

### MCHP
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.0183
- model_reason_summary: holding above recent breakout, strong sector momentum backdrop, well above 200d trend
- opportunity_score: 0.4513
- selection_score: 0.0183
- raw_opportunity_rank: 6
- adjusted_selection_rank: 100
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0800
- exclusion_reason: below opportunity threshold
- overlap_components: same_slot=0.08

### FORM
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.0938
- model_reason_summary: well above 200d trend, supportive distance above 50d, clear of near-term earnings
- opportunity_score: 0.4502
- selection_score: 0.0938
- raw_opportunity_rank: 7
- adjusted_selection_rank: 19
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0800
- exclusion_reason: below opportunity threshold
- overlap_components: same_slot=0.08

### MPWR
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.0128
- model_reason_summary: clear of near-term earnings, well above 200d trend, limited recent downside gap risk
- opportunity_score: 0.4476
- selection_score: 0.0128
- raw_opportunity_rank: 8
- adjusted_selection_rank: 110
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0800
- exclusion_reason: below opportunity threshold
- overlap_components: same_slot=0.08

### LITE
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.0163
- model_reason_summary: well above 200d trend, constructive ATR profile, timely post-earnings setup
- opportunity_score: 0.4443
- selection_score: 0.0163
- raw_opportunity_rank: 9
- adjusted_selection_rank: 105
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0800
- exclusion_reason: below opportunity threshold
- overlap_components: same_slot=0.08

### RMBS
- slot: technology
- sector: Information Technology
- selection_source: shortlist_model
- model_name: xgboost_model
- model_predicted_alpha: 0.0727
- model_reason_summary: limited recent downside gap risk, well above 200d trend, strong sector momentum backdrop
- opportunity_score: 0.4413
- selection_score: 0.0727
- raw_opportunity_rank: 10
- adjusted_selection_rank: 30
- recent_feedback_adjustment: +0.0000
- slot_overlay_adjustment: +0.0000
- signal_score: n/a (model-sourced)
- overlap_penalty: 0.0800
- exclusion_reason: below opportunity threshold
- overlap_components: same_slot=0.08

## Outcome Coverage

- 5d: forward=0 sector_alpha=0 spy_alpha=0
- 10d: forward=0 sector_alpha=0 spy_alpha=0
- 20d: forward=0 sector_alpha=0 spy_alpha=0
- mfe_20d_available: 0
- mae_20d_available: 0

## Post-Change Selector Maturity

No scan dates show active recent-feedback selector adjustments yet.

## Recent Early Read

- target_column: alpha_vs_sector_1d
- dates: 2026-05-26, 2026-05-27, 2026-05-28, 2026-05-29, 2026-06-01

### 2026-05-26
- selected_mean_target: -0.017011
- excluded_mean_target: -0.004873
- selected_hit_rate: 0.1667
- selected_count: 6
- selected_tickers: AESI, CF, PBI, AR, ESI, CENX

### 2026-05-27
- selected_mean_target: -0.010192
- excluded_mean_target: 0.001386
- selected_hit_rate: 0.5000
- selected_count: 6
- selected_tickers: APA, CHRD, AESI, AGX, SXT, CTVA

### 2026-05-28
- selected_mean_target: -0.007987
- excluded_mean_target: 0.001328
- selected_hit_rate: 0.3333
- selected_count: 6
- selected_tickers: APA, TALO, AESI, AGX, SXT, CTVA

### 2026-05-29
- selected_mean_target: -0.003137
- excluded_mean_target: 0.003356
- selected_hit_rate: 0.3333
- selected_count: 6
- selected_tickers: APA, TALO, AESI, AGX, SXT, CTVA

### 2026-06-01
- selected_mean_target: 0.014268
- excluded_mean_target: 0.004034
- selected_hit_rate: 0.3333
- selected_count: 6
- selected_tickers: TALO, APA, AESI, AA, SXT, CTVA

## Signal-First Selector Early Read

- basis: recent live signal-first dates compared with opportunity-score counterfactual under the same caps
- scan_dates: 2026-06-05, 2026-06-08, 2026-06-09, 2026-06-10, 2026-06-11

### 2026-06-05
- runtime_tickers: INTC, VSH, ANET, AMR, CF, CE
- opportunity_counterfactual: none
- added_vs_opportunity: AMR, ANET, CE, CF, INTC, VSH
- removed_vs_opportunity: none
- 1d: pending
- 5d: pending
- 10d: pending

### 2026-06-08
- runtime_tickers: INTC, VSH, ANET, AMR, CF, CE
- opportunity_counterfactual: none
- added_vs_opportunity: AMR, ANET, CE, CF, INTC, VSH
- removed_vs_opportunity: none
- 1d: pending
- 5d: pending
- 10d: pending

### 2026-06-09
- runtime_tickers: INTC, VSH, SMCI, AMR, CE, CF
- opportunity_counterfactual: none
- added_vs_opportunity: AMR, CE, CF, INTC, SMCI, VSH
- removed_vs_opportunity: none
- 1d: pending
- 5d: pending
- 10d: pending

### 2026-06-10
- runtime_tickers: INTC, VSH, SMCI, AMR, CE, CF
- opportunity_counterfactual: none
- added_vs_opportunity: AMR, CE, CF, INTC, SMCI, VSH
- removed_vs_opportunity: none
- 1d: pending
- 5d: pending
- 10d: pending

### 2026-06-11
- runtime_tickers: INTC, SMCI, GEN, AMR, CE, CF
- opportunity_counterfactual: none
- added_vs_opportunity: AMR, CE, CF, GEN, INTC, SMCI
- removed_vs_opportunity: none
- 1d: pending
- 5d: pending
- 10d: pending

## Forward Attribution

No forward return windows are fully available yet for this scan date.

## Recent Selection Mistakes

- target_column: alpha_vs_sector_10d
- recent_scan_dates: 20
- date_range: 2026-04-21 -> 2026-05-18

### Selected ideas
- rows: 119
- mean_target: -0.024901
- median_target: -0.022999
- hit_rate: 0.3361

### Excluded eligible ideas
- rows: 755
- mean_target: -0.007771
- median_target: -0.009288
- hit_rate: 0.4066

### Selected Slot Breakdown
- energy: mean_target=-0.015873 hit_rate=0.3167 rows=60
- materials: mean_target=-0.033758 hit_rate=0.3256 rows=43
- industrials: mean_target=-0.034948 hit_rate=0.4375 rows=16

### Excluded Slot Breakdown
- industrials: mean_target=0.030616 hit_rate=0.4857 rows=35
- energy: mean_target=-0.007945 hit_rate=0.4071 rows=619
- materials: mean_target=-0.020007 hit_rate=0.3762 rows=101

### Biggest Missed Swaps
#### 2026-05-18 industrials
- selected: MTZ rank=5 target=-0.0718 opportunity=0.7248
- better_excluded: VICR target=0.3180 opportunity=0.6645
- performance_gap: 0.3898

#### 2026-04-29 industrials
- selected: CAR rank=6 target=-0.2001 opportunity=0.7083
- better_excluded: VRT target=0.1867 opportunity=0.6835
- performance_gap: 0.3868

#### 2026-04-28 industrials
- selected: CAR rank=5 target=-0.1926 opportunity=0.7155
- better_excluded: VRT target=0.1839 opportunity=0.7001
- performance_gap: 0.3765

#### 2026-05-01 materials
- selected: WLK rank=6 target=-0.2091 opportunity=0.7420
- better_excluded: MTRN target=0.1097 opportunity=0.6668
- performance_gap: 0.3188

#### 2026-05-14 industrials
- selected: VICR rank=6 target=0.1604 opportunity=0.6686
- better_excluded: RXO target=0.4688 opportunity=0.7230
- performance_gap: 0.3084

#### 2026-05-18 materials
- selected: CF rank=4 target=-0.1220 opportunity=0.7380
- better_excluded: STLD target=0.1666 opportunity=0.6626
- performance_gap: 0.2885

#### 2026-04-24 energy
- selected: COP rank=1 target=-0.0435 opportunity=0.8216
- better_excluded: DINO target=0.2157 opportunity=0.7742
- performance_gap: 0.2592

#### 2026-05-05 energy
- selected: PARR rank=2 target=-0.1366 opportunity=0.7470
- better_excluded: VAL target=0.1155 opportunity=0.7823
- performance_gap: 0.2521

#### 2026-04-29 energy
- selected: NOG rank=3 target=-0.1412 opportunity=0.7887
- better_excluded: OII target=0.1034 opportunity=0.8114
- performance_gap: 0.2447

#### 2026-05-07 materials
- selected: ALB rank=5 target=-0.1166 opportunity=0.7638
- better_excluded: MTRN target=0.1259 opportunity=0.6992
- performance_gap: 0.2425

#### 2026-04-28 materials
- selected: CE rank=6 target=-0.0945 opportunity=0.7247
- better_excluded: MTRN target=0.1344 opportunity=0.6645
- performance_gap: 0.2289

#### 2026-04-27 energy
- selected: COP rank=1 target=-0.0574 opportunity=0.8270
- better_excluded: DINO target=0.1711 opportunity=0.8102
- performance_gap: 0.2285

### Repeated Recent Drags
- CAR (industrials): mean_target=-0.2446 median_target=-0.2065 hit_rate=0.0000 picks=5
- WLK (materials): mean_target=-0.1633 median_target=-0.1433 hit_rate=0.0000 picks=5
- ALB (materials): mean_target=-0.1186 median_target=-0.1302 hit_rate=0.0000 picks=4
- MTDR (energy): mean_target=-0.0592 median_target=-0.0592 hit_rate=0.0000 picks=2
- CC (materials): mean_target=-0.0532 median_target=-0.0566 hit_rate=0.0000 picks=6
- MUR (energy): mean_target=-0.0478 median_target=-0.0478 hit_rate=0.0000 picks=3
- PARR (energy): mean_target=-0.0465 median_target=-0.0288 hit_rate=0.5000 picks=6
- CE (materials): mean_target=-0.0410 median_target=-0.0945 hit_rate=0.3333 picks=3

## Mediocre Setup Diagnostics

- target_column: alpha_vs_sector_10d
- recent_scan_dates: 20
- date_range: 2026-04-21 -> 2026-05-18
- signal_score_q25: 34.0000
- setup_quality_q25: 0.7445
- opportunity_score_q25: 0.7277

### All selected ideas
- rows: 119
- mean_target: -0.024901
- median_target: -0.022999
- hit_rate: 0.3361

### Low-signal selected ideas
- rows: 59
- mean_target: -0.034081
- median_target: -0.019178
- hit_rate: 0.3559

### Low-setup selected ideas
- rows: 30
- mean_target: -0.039181
- median_target: -0.029598
- hit_rate: 0.4000

### Low-opportunity selected ideas
- rows: 30
- mean_target: -0.028893
- median_target: -0.039802
- hit_rate: 0.3333

### Low-signal and low-setup selected ideas
- rows: 27
- mean_target: -0.038006
- median_target: -0.035913
- hit_rate: 0.4074

### Repeated Mediocre Drags
- CAR (industrials): mean_target=-0.2446 hit_rate=0.0000 picks=5 median_signal_score=34.00 median_setup_quality=0.7083
- ALB (materials): mean_target=-0.1456 hit_rate=0.0000 picks=2 median_signal_score=32.67 median_setup_quality=0.7259
- CE (materials): mean_target=-0.0133 hit_rate=0.5000 picks=2 median_signal_score=33.35 median_setup_quality=0.7411

## Regime Attribution

- target_column: alpha_vs_sector_10d
- recent_scan_dates: 40
- date_range: 2026-03-05 -> 2026-05-18
- purpose: compare selected vs excluded outcomes by market and sector regime

### SPY 200d Regime
#### green
### Selected ideas
- rows: 239
- mean_target: 0.023795
- median_target: 0.019689
- hit_rate: 0.5900

### Excluded eligible ideas
- rows: 4944
- mean_target: 0.019911
- median_target: 0.011976
- hit_rate: 0.5862

##### Slot Breakdown
- technology: selected_mean_target=0.048509 excluded_mean_target=0.054615 selected_rows=46 excluded_rows=1070
- industrials: selected_mean_target=0.027254 excluded_mean_target=0.015749 selected_rows=25 excluded_rows=1627
- energy: selected_mean_target=0.022397 excluded_mean_target=0.005614 selected_rows=104 excluded_rows=1651
- materials: selected_mean_target=0.006954 excluded_mean_target=0.008573 selected_rows=64 excluded_rows=596

### QQQ 200d Regime
#### green
### Selected ideas
- rows: 239
- mean_target: 0.023795
- median_target: 0.019689
- hit_rate: 0.5900

### Excluded eligible ideas
- rows: 4944
- mean_target: 0.019911
- median_target: 0.011976
- hit_rate: 0.5862

##### Slot Breakdown
- technology: selected_mean_target=0.048509 excluded_mean_target=0.054615 selected_rows=46 excluded_rows=1070
- industrials: selected_mean_target=0.027254 excluded_mean_target=0.015749 selected_rows=25 excluded_rows=1627
- energy: selected_mean_target=0.022397 excluded_mean_target=0.005614 selected_rows=104 excluded_rows=1651
- materials: selected_mean_target=0.006954 excluded_mean_target=0.008573 selected_rows=64 excluded_rows=596

### Sector ETF 200d Regime
#### green
### Selected ideas
- rows: 236
- mean_target: 0.023469
- median_target: 0.019451
- hit_rate: 0.5890

### Excluded eligible ideas
- rows: 4884
- mean_target: 0.019185
- median_target: 0.011388
- hit_rate: 0.5827

##### Slot Breakdown
- technology: selected_mean_target=0.048444 excluded_mean_target=0.053165 selected_rows=43 excluded_rows=1010
- industrials: selected_mean_target=0.027254 excluded_mean_target=0.015749 selected_rows=25 excluded_rows=1627
- energy: selected_mean_target=0.022397 excluded_mean_target=0.005614 selected_rows=104 excluded_rows=1651
- materials: selected_mean_target=0.006954 excluded_mean_target=0.008573 selected_rows=64 excluded_rows=596

#### red
### Selected ideas
- rows: 3
- mean_target: 0.049431
- median_target: 0.120059
- hit_rate: 0.6667

### Excluded eligible ideas
- rows: 60
- mean_target: 0.079014
- median_target: 0.077520
- hit_rate: 0.8667

##### Slot Breakdown
- technology: selected_mean_target=0.049431 excluded_mean_target=0.079014 selected_rows=3 excluded_rows=60

### Sector Breadth
#### high
### Selected ideas
- rows: 88
- mean_target: 0.023148
- median_target: 0.015372
- hit_rate: 0.6477

### Excluded eligible ideas
- rows: 1368
- mean_target: 0.004872
- median_target: 0.002401
- hit_rate: 0.5139

##### Slot Breakdown
- materials: selected_mean_target=0.029411 excluded_mean_target=0.009056 selected_rows=19 excluded_rows=115
- energy: selected_mean_target=0.021686 excluded_mean_target=0.004366 selected_rows=68 excluded_rows=1246
- industrials: selected_mean_target=0.003624 excluded_mean_target=0.026288 selected_rows=1 excluded_rows=7

#### mixed
### Selected ideas
- rows: 151
- mean_target: 0.024172
- median_target: 0.019882
- hit_rate: 0.5563

### Excluded eligible ideas
- rows: 3576
- mean_target: 0.025664
- median_target: 0.016732
- hit_rate: 0.6138

##### Slot Breakdown
- technology: selected_mean_target=0.048509 excluded_mean_target=0.054615 selected_rows=46 excluded_rows=1070
- industrials: selected_mean_target=0.028239 excluded_mean_target=0.015704 selected_rows=24 excluded_rows=1620
- energy: selected_mean_target=0.023739 excluded_mean_target=0.009454 selected_rows=36 excluded_rows=405
- materials: selected_mean_target=-0.002528 excluded_mean_target=0.008458 selected_rows=45 excluded_rows=481

## Slot Internal Attribution

- target_column: alpha_vs_sector_10d
- recent_scan_dates: 40
- date_range: 2026-03-05 -> 2026-05-18
- method: within each slot, compare high-half vs low-half of each feature on matured eligible rows

### energy
- eligible_rows: 1755
- selected_rows: 104
#### Strongest Positive Discriminators
- avg gap: threshold=0.0120 eligible_high_minus_low=0.014778 selected_high_minus_low=0.011447 rows=1755
- worst gap down: threshold=0.0543 eligible_high_minus_low=0.011408 selected_high_minus_low=0.002632 rows=1755
- sector breadth 200d: threshold=0.9508 eligible_high_minus_low=0.004100 selected_high_minus_low=0.019034 rows=1755
- RS vs QQQ: threshold=88.6525 eligible_high_minus_low=0.000122 selected_high_minus_low=0.022090 rows=1755
- RS vs SPY: threshold=88.6525 eligible_high_minus_low=0.000122 selected_high_minus_low=0.022090 rows=1755
- 63d momentum: threshold=0.2659 eligible_high_minus_low=-0.001820 selected_high_minus_low=0.015001 rows=1755

#### Strongest Negative Discriminators
- breakout extension: threshold=-0.0598 eligible_high_minus_low=-0.012502 selected_high_minus_low=0.013078 rows=1755
- 126d momentum: threshold=0.3875 eligible_high_minus_low=-0.008173 selected_high_minus_low=0.003531 rows=1755
- sector breadth 50d: threshold=0.7581 eligible_high_minus_low=-0.007823 selected_high_minus_low=0.006324 rows=1755
- expected alpha: threshold=0.8114 eligible_high_minus_low=-0.003963 selected_high_minus_low=0.005108 rows=1755
- freshness: threshold=0.7196 eligible_high_minus_low=-0.003457 selected_high_minus_low=-0.058708 rows=1755
- distance above 200d: threshold=0.2735 eligible_high_minus_low=-0.002623 selected_high_minus_low=0.027193 rows=1755

### industrials
- eligible_rows: 1652
- selected_rows: 25
#### Strongest Positive Discriminators
- avg gap: threshold=0.0100 eligible_high_minus_low=0.023489 selected_high_minus_low=0.084535 rows=1652
- worst gap down: threshold=0.0333 eligible_high_minus_low=0.019967 selected_high_minus_low=0.048628 rows=1652
- 126d momentum: threshold=0.2719 eligible_high_minus_low=0.015800 selected_high_minus_low=0.086689 rows=1652
- distance above 200d: threshold=0.1968 eligible_high_minus_low=0.014343 selected_high_minus_low=0.137943 rows=1652
- expected alpha: threshold=0.6806 eligible_high_minus_low=0.009014 selected_high_minus_low=nan rows=1652
- RS vs QQQ: threshold=81.6406 eligible_high_minus_low=0.008991 selected_high_minus_low=nan rows=1652

#### Strongest Negative Discriminators
- freshness: threshold=0.7239 eligible_high_minus_low=-0.023572 selected_high_minus_low=-0.107278 rows=1652
- sector breadth 200d: threshold=0.6288 eligible_high_minus_low=-0.003739 selected_high_minus_low=-0.179253 rows=1652
- breakout extension: threshold=-0.0693 eligible_high_minus_low=-0.002087 selected_high_minus_low=0.134568 rows=1652
- sector breadth 50d: threshold=0.3624 eligible_high_minus_low=0.000736 selected_high_minus_low=-0.192994 rows=1652
- 63d momentum: threshold=0.1722 eligible_high_minus_low=0.006310 selected_high_minus_low=nan rows=1652
- RS vs QQQ: threshold=81.6406 eligible_high_minus_low=0.008991 selected_high_minus_low=nan rows=1652

### materials
- eligible_rows: 660
- selected_rows: 64
#### Strongest Positive Discriminators
- breakout extension: threshold=-0.0790 eligible_high_minus_low=0.000590 selected_high_minus_low=-0.066331 rows=660
- sector breadth 200d: threshold=0.6308 eligible_high_minus_low=-0.000105 selected_high_minus_low=-0.002060 rows=660
- 63d momentum: threshold=0.2279 eligible_high_minus_low=-0.000456 selected_high_minus_low=-0.014956 rows=660
- RS vs QQQ: threshold=86.5195 eligible_high_minus_low=-0.000896 selected_high_minus_low=-0.004881 rows=660
- RS vs SPY: threshold=86.5195 eligible_high_minus_low=-0.000896 selected_high_minus_low=-0.004881 rows=660
- avg gap: threshold=0.0136 eligible_high_minus_low=-0.004092 selected_high_minus_low=0.076144 rows=660

#### Strongest Negative Discriminators
- 126d momentum: threshold=0.3682 eligible_high_minus_low=-0.033782 selected_high_minus_low=-0.138619 rows=660
- freshness: threshold=0.7087 eligible_high_minus_low=-0.027802 selected_high_minus_low=-0.047880 rows=660
- worst gap down: threshold=0.0497 eligible_high_minus_low=-0.021256 selected_high_minus_low=0.007481 rows=660
- distance above 200d: threshold=0.2649 eligible_high_minus_low=-0.019173 selected_high_minus_low=-0.103853 rows=660
- sector breadth 50d: threshold=0.4615 eligible_high_minus_low=-0.015622 selected_high_minus_low=-0.087795 rows=660
- expected alpha: threshold=0.7676 eligible_high_minus_low=-0.004354 selected_high_minus_low=-0.049199 rows=660

### technology
- eligible_rows: 1116
- selected_rows: 46
#### Strongest Positive Discriminators
- worst gap down: threshold=0.0449 eligible_high_minus_low=0.027695 selected_high_minus_low=nan rows=1116
- avg gap: threshold=0.0163 eligible_high_minus_low=0.024848 selected_high_minus_low=nan rows=1116
- 126d momentum: threshold=0.4436 eligible_high_minus_low=-0.004534 selected_high_minus_low=nan rows=1116
- distance above 200d: threshold=0.3332 eligible_high_minus_low=-0.006841 selected_high_minus_low=nan rows=1116
- sector breadth 200d: threshold=0.5058 eligible_high_minus_low=-0.008762 selected_high_minus_low=-0.000798 rows=1116
- freshness: threshold=0.6617 eligible_high_minus_low=-0.011672 selected_high_minus_low=-0.023239 rows=1116

#### Strongest Negative Discriminators
- breakout extension: threshold=-0.0818 eligible_high_minus_low=-0.030752 selected_high_minus_low=-0.155467 rows=1116
- sector breadth 50d: threshold=0.4128 eligible_high_minus_low=-0.024506 selected_high_minus_low=-0.030278 rows=1116
- 63d momentum: threshold=0.2862 eligible_high_minus_low=-0.019832 selected_high_minus_low=nan rows=1116
- RS vs QQQ: threshold=90.9446 eligible_high_minus_low=-0.014536 selected_high_minus_low=nan rows=1116
- RS vs SPY: threshold=90.9446 eligible_high_minus_low=-0.014536 selected_high_minus_low=nan rows=1116
- expected alpha: threshold=0.8197 eligible_high_minus_low=-0.014468 selected_high_minus_low=nan rows=1116

## Selector Bakeoff

- target_column: alpha_vs_sector_10d
- current_live_policy: signal_score primary with opportunity_score tie-break and portfolio caps
- available_matured_dates: 542

### Window 10
- recent_matured_dates: 10
- date_range: 2026-05-05 to 2026-05-18

#### Summary
- runtime: mean_target=-0.007752 hit_rate=0.3733 avg_pick_count=5.90 days=10 beats_runtime=n/a
- opportunity: mean_target=0.002229 hit_rate=0.3900 avg_pick_count=5.90 days=10 beats_runtime=0.6000
- signal: mean_target=-0.007752 hit_rate=0.3733 avg_pick_count=5.90 days=10 beats_runtime=0.0000
- learned: mean_target=0.002229 hit_rate=0.3900 avg_pick_count=5.90 days=10 beats_runtime=0.6000
- random: mean_target=-0.014997 hit_rate=0.3567 avg_pick_count=5.90 days=10 beats_runtime=0.6000

#### Slot Breakdown
##### energy
- runtime: mean_target=-0.022487 hit_rate=0.2667 days=10
- opportunity: mean_target=-0.006820 hit_rate=0.4000 days=10
- signal: mean_target=-0.022487 hit_rate=0.2667 days=10
- learned: mean_target=-0.006820 hit_rate=0.4000 days=10
- random: mean_target=-0.025005 hit_rate=0.3000 days=10

##### industrials
- runtime: mean_target=0.077724 hit_rate=0.8333 days=6
- opportunity: mean_target=0.276288 hit_rate=0.6667 days=3
- signal: mean_target=0.077724 hit_rate=0.8333 days=6
- learned: mean_target=0.276288 hit_rate=0.6667 days=3
- random: mean_target=0.035359 hit_rate=0.6667 days=5

##### materials
- runtime: mean_target=-0.016910 hit_rate=0.4000 days=10
- opportunity: mean_target=-0.020250 hit_rate=0.3500 days=10
- signal: mean_target=-0.016910 hit_rate=0.4000 days=10
- learned: mean_target=-0.020250 hit_rate=0.3500 days=10
- random: mean_target=-0.028282 hit_rate=0.3148 days=9

#### Biggest Runtime-vs-Opportunity Disagreements
##### 2026-05-14
- runtime: ALB, CENX, CRGY, PR, TALO, VICR
- opportunity: APA, CENX, CF, CRGY, OXY, RXO
- runtime_mean_target: 0.016225
- opportunity_mean_target: 0.079200
- delta: 0.062974

##### 2026-05-08
- runtime: ALB, APA, CC, GEV, MUR, PR
- opportunity: APA, CC, CF, CRGY, LYB, PR
- runtime_mean_target: -0.041609
- opportunity_mean_target: 0.001189
- delta: 0.042799

##### 2026-05-05
- runtime: CENX, CF, COP, PARR, VLO
- opportunity: CENX, CF, OXY, TDW, VAL
- runtime_mean_target: -0.024494
- opportunity_mean_target: 0.013617
- delta: 0.038111

##### 2026-05-07
- runtime: ALB, APA, CC, CENX, CRC, MTDR
- opportunity: ALB, APA, CC, CF, CRGY, PR
- runtime_mean_target: -0.033246
- opportunity_mean_target: -0.020481
- delta: 0.012764

##### 2026-05-13
- runtime: CC, CF, CRGY, OVV, PBI, TALO
- opportunity: APA, CENX, CF, CRGY, LYB, OVV
- runtime_mean_target: -0.031546
- opportunity_mean_target: -0.020775
- delta: 0.010771

##### 2026-05-11
- runtime: ALB, CC, CF, CRGY, MUR, OXY
- opportunity: APA, CC, CF, CRGY, LYB, OVV
- runtime_mean_target: -0.048625
- opportunity_mean_target: -0.038191
- delta: 0.010434

##### 2026-05-12
- runtime: CC, CENX, CRGY, OVV, PBI, VNOM
- opportunity: APA, CC, CENX, CRGY, LYB, OVV
- runtime_mean_target: -0.023999
- opportunity_mean_target: -0.031623
- delta: -0.007624

##### 2026-05-18
- runtime: APA, CF, CRGY, ESI, MTZ, PARR
- opportunity: CF, CRGY, DOW, MTZ, OXY, PARR
- runtime_mean_target: -0.034627
- opportunity_mean_target: -0.055949
- delta: -0.021322

##### 2026-05-15
- runtime: APA, CENX, CRGY, PARR, RXO, VICR
- opportunity: APA, CENX, CF, CRGY, OXY, RXO
- runtime_mean_target: 0.122081
- opportunity_mean_target: 0.072978
- delta: -0.049103

### Window 20
- recent_matured_dates: 20
- date_range: 2026-04-21 to 2026-05-18

#### Summary
- runtime: mean_target=-0.024897 hit_rate=0.3367 avg_pick_count=5.95 days=20 beats_runtime=n/a
- opportunity: mean_target=-0.015725 hit_rate=0.3617 avg_pick_count=5.95 days=20 beats_runtime=0.6000
- signal: mean_target=-0.024897 hit_rate=0.3367 avg_pick_count=5.95 days=20 beats_runtime=0.0000
- learned: mean_target=-0.015725 hit_rate=0.3617 avg_pick_count=5.95 days=20 beats_runtime=0.6000
- random: mean_target=-0.017031 hit_rate=0.3783 avg_pick_count=5.95 days=20 beats_runtime=0.7000

#### Slot Breakdown
##### energy
- runtime: mean_target=-0.015873 hit_rate=0.3167 days=20
- opportunity: mean_target=-0.006238 hit_rate=0.3833 days=20
- signal: mean_target=-0.015873 hit_rate=0.3167 days=20
- learned: mean_target=-0.006238 hit_rate=0.3833 days=20
- random: mean_target=-0.014549 hit_rate=0.3667 days=20

##### industrials
- runtime: mean_target=-0.049073 hit_rate=0.4615 days=13
- opportunity: mean_target=0.033527 hit_rate=0.3750 days=8
- signal: mean_target=-0.049073 hit_rate=0.4615 days=13
- learned: mean_target=0.033527 hit_rate=0.3750 days=8
- random: mean_target=-0.062682 hit_rate=0.3333 days=10

##### materials
- runtime: mean_target=-0.032479 hit_rate=0.3167 days=20
- opportunity: mean_target=-0.037536 hit_rate=0.3417 days=20
- signal: mean_target=-0.032479 hit_rate=0.3167 days=20
- learned: mean_target=-0.037536 hit_rate=0.3417 days=20
- random: mean_target=-0.019476 hit_rate=0.3596 days=19

### Window 40
- recent_matured_dates: 40
- date_range: 2026-03-05 to 2026-05-18

#### Summary
- runtime: mean_target=0.023594 hit_rate=0.5892 avg_pick_count=5.97 days=40 beats_runtime=n/a
- opportunity: mean_target=0.000808 hit_rate=0.5149 avg_pick_count=4.35 days=40 beats_runtime=0.3500
- signal: mean_target=-0.000614 hit_rate=0.5281 avg_pick_count=4.35 days=40 beats_runtime=0.0500
- learned: mean_target=0.000808 hit_rate=0.5149 avg_pick_count=4.35 days=40 beats_runtime=0.3500
- random: mean_target=0.000193 hit_rate=0.5149 avg_pick_count=4.35 days=40 beats_runtime=0.4250

#### Slot Breakdown
##### energy
- runtime: mean_target=0.028825 hit_rate=0.6316 days=38
- opportunity: mean_target=0.001222 hit_rate=0.5088 days=38
- signal: mean_target=0.003716 hit_rate=0.5175 days=38
- learned: mean_target=0.001222 hit_rate=0.5088 days=38
- random: mean_target=-0.002664 hit_rate=0.5000 days=38

##### industrials
- runtime: mean_target=0.019056 hit_rate=0.6500 days=20
- opportunity: mean_target=0.057091 hit_rate=0.4500 days=10
- signal: mean_target=-0.025548 hit_rate=0.5333 days=15
- learned: mean_target=0.057091 hit_rate=0.4500 days=10
- random: mean_target=-0.013953 hit_rate=0.4872 days=13

##### materials
- runtime: mean_target=0.020039 hit_rate=0.5556 days=33
- opportunity: mean_target=-0.023309 hit_rate=0.4097 days=24
- signal: mean_target=-0.019025 hit_rate=0.3841 days=23
- learned: mean_target=-0.023309 hit_rate=0.4097 days=24
- random: mean_target=-0.003629 hit_rate=0.4242 days=22

##### technology
- runtime: mean_target=0.044878 hit_rate=0.6354 days=16

## Selector Shadow Comparison

- target_column: alpha_vs_sector_10d
- recent_matured_dates: 10
- date_range: 2026-05-05 to 2026-05-18

### Summary
- runtime: mean_target=0.002784 hit_rate=0.4487 avg_pick_count=2.11 days=10
- shadow_old: mean_target=0.008133 hit_rate=0.4940 avg_pick_count=2.75 days=10
- shadow_new: mean_target=0.011571 hit_rate=0.4940 avg_pick_count=2.75 days=10

### Slot Breakdown
#### energy
- runtime: mean_target=-0.022487 hit_rate=0.2667 days=10
- shadow_old: mean_target=-0.006820 hit_rate=0.4000 days=10
- shadow_new: mean_target=-0.022487 hit_rate=0.2667 days=10

#### industrials
- runtime: mean_target=0.077724 hit_rate=0.8333 days=8
- shadow_old: mean_target=0.070088 hit_rate=0.8333 days=8
- shadow_new: mean_target=0.092529 hit_rate=0.9167 days=8

#### materials
- runtime: mean_target=-0.016910 hit_rate=0.4000 days=10
- shadow_old: mean_target=-0.026478 hit_rate=0.3167 days=10
- shadow_new: mean_target=-0.019138 hit_rate=0.3833 days=10

### Biggest Old-vs-New Swaps
#### 2026-05-18 materials
- shadow_old: CF, DOW, LYB
- shadow_new: CF, ESI, NUE
- old_mean_target: -0.122832
- new_mean_target: 0.014644
- delta: 0.137475

#### 2026-05-18 industrials
- shadow_old: MTZ, AGX, POWL
- shadow_new: MTZ, POWL, VICR
- old_mean_target: -0.000686
- new_mean_target: 0.115263
- delta: 0.115949

#### 2026-05-15 materials
- shadow_old: CF, CENX, LYB
- shadow_new: CENX, CF, STLD
- old_mean_target: -0.037684
- new_mean_target: 0.044362
- delta: 0.082045

#### 2026-05-15 industrials
- shadow_old: RXO, PBI, POWL
- shadow_new: RXO, VICR, PBI
- old_mean_target: 0.161945
- new_mean_target: 0.234315
- delta: 0.072370

#### 2026-05-11 energy
- shadow_old: APA, CRGY, OVV
- shadow_new: CRGY, MUR, OXY
- old_mean_target: -0.021009
- new_mean_target: -0.009572
- delta: 0.011437

#### 2026-05-05 materials
- shadow_old: CF, CENX
- shadow_new: CENX, CF
- old_mean_target: 0.013403
- new_mean_target: 0.013403
- delta: 0.000000

#### 2026-05-14 industrials
- shadow_old: RXO, PBI, VICR
- shadow_new: VICR, PBI, RXO
- old_mean_target: 0.220427
- new_mean_target: 0.220427
- delta: 0.000000

#### 2026-05-15 energy
- shadow_old: APA, OXY, CRGY
- shadow_new: APA, CRGY, PARR
- old_mean_target: 0.000415
- new_mean_target: -0.002532
- delta: -0.002947

#### 2026-05-12 materials
- shadow_old: CENX, LYB, CC
- shadow_new: CENX, CC, CF
- old_mean_target: -0.030928
- new_mean_target: -0.033919
- delta: -0.002990

#### 2026-05-14 materials
- shadow_old: CF, CENX, ALB
- shadow_new: CENX, ALB, CC
- old_mean_target: -0.018510
- new_mean_target: -0.026113
- delta: -0.007603

## Candidate Ranker Validation

- target_column: alpha_vs_sector_10d
- validation_method: purged_walk_forward
- embargo_days: 10
- validation_blocks: 5
- train_rows: 41444
- validation_rows: 33185
- train_dates: 435
- validation_dates: 163
- feature_count: 119
- prediction_correlation: 0.082055

### Validation Averages
- learned_mean_target: 0.035013
- learned_hit_rate: 0.584254
- handcrafted_mean_target: 0.007235
- handcrafted_hit_rate: 0.530061
- runtime_mean_target: 0.062641
- runtime_hit_rate: 0.698773

### Ranker Quintile Tear Sheet
- Q1: mean_target=0.023210 hit_rate=0.5531 mean_ranker_score=0.031283 rows=6699 dates=163
- Q2: mean_target=0.008238 hit_rate=0.5194 mean_ranker_score=0.009220 rows=6639 dates=163
- Q3: mean_target=0.004076 hit_rate=0.5227 mean_ranker_score=0.001343 rows=6640 dates=163
- Q4: mean_target=0.004000 hit_rate=0.4984 mean_ranker_score=-0.007730 rows=6639 dates=163
- Q5: mean_target=-0.000790 hit_rate=0.4823 mean_ranker_score=-0.022792 rows=6568 dates=163
- q1_minus_q5_spread: 0.024000
- learned_top_n_turnover_mean: 0.1903
- learned_top_n_turnover_pairs: 162

### Daily IC
- ic_mean: 0.085857
- ic_std: 0.151547
- ic_t_stat: 7.233037
- ic_dates: 163

### Slot Breakdown
- technology: q1_mean=0.035308 q5_mean=-0.003875 spread=0.039183 rows=10323 dates=139
- materials: q1_mean=0.020117 q5_mean=0.002556 spread=0.017561 rows=3721 dates=163
- industrials: q1_mean=0.004323 q5_mean=-0.007807 spread=0.012130 rows=13753 dates=159
- energy: q1_mean=0.003656 q5_mean=0.006899 spread=-0.003244 rows=5388 dates=163

### Sector Breakdown
- Information Technology: q1_mean=0.035308 q5_mean=-0.003875 spread=0.039183 rows=10323 dates=139
- Materials: q1_mean=0.020117 q5_mean=0.002556 spread=0.017561 rows=3721 dates=163
- Industrials: q1_mean=0.004323 q5_mean=-0.007807 spread=0.012130 rows=13753 dates=159
- Energy: q1_mean=0.003656 q5_mean=0.006899 spread=-0.003244 rows=5388 dates=163

### Latest Validation Date: 2026-05-18
- learned_tickers: FIX, POWL, AGX, DOW, KALU, LYB
- handcrafted_tickers: CRGY, PARR, OXY, CF, MTZ, DOW
- runtime_tickers: CRGY, PARR, APA, CF, MTZ, ESI
