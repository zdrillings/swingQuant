# Scan Performance

- benchmark: sector
- scope: all
- selection_source: all
- model_name: all
- model_generated_at: all
- latest_model_generated_at: n/a
- recent_scan_dates: 20
- scan_dates: 20
- selected_rows: 68
- scan_date_min: 2026-08-05
- scan_date_max: 2026-09-01

## Selection Source Coverage

- heuristic: 19 (27.9%)
- shortlist_model: 19 (27.9%)
- unknown: 30 (44.1%)

## Latest Model Selection Audit

- scan_date: 2026-09-01
- note: final picks are chosen after opportunity floor, recent rotation, current holdings, and portfolio caps; this section surfaces model-rank divergences.
- selected:
  - PANW: selected_rank=1, model_rank=9, slot=technology, selection_score=0.0881, opportunity=0.4590
  - NEU: selected_rank=2, model_rank=16, slot=materials, selection_score=0.0647, opportunity=0.4437
- higher_model_rank_unselected:
  - GDDY: model_rank=2, slot=technology, selection_score=0.1455, opportunity=0.2694
  - CDW: model_rank=3, slot=technology, selection_score=0.1406, opportunity=0.3016
  - RNG: model_rank=4, slot=technology, selection_score=0.1153, opportunity=-0.5797
  - ONTO: model_rank=7, slot=technology, selection_score=0.0959, opportunity=0.2841
  - NEO: model_rank=8, slot=healthcare, selection_score=0.0908, opportunity=-0.5088
  - PATH: model_rank=10, slot=technology, selection_score=0.0808, opportunity=0.3679
  - SPSC: model_rank=11, slot=technology, selection_score=0.0759, opportunity=0.3772
  - HPE: model_rank=15, slot=technology, selection_score=0.0684, opportunity=0.4103

## Horizon Summary

### 1d
- matured_picks: 64
- matured_scan_dates: 18
- latest_selected_scan_date: 2026-09-01
- latest_matured_scan_date: 2026-08-28
- mean_return: 0.28%
- median_return: 0.43%
- return_iqr: -1.42% to 2.36%
- return_p05_p95: -5.13% to 4.39%
- return_range: -7.16% to 5.97%
- hit_rate: 54.69%
- mean_alpha_vs_sector: 0.27%
- median_alpha_vs_sector: 0.27%
- alpha_iqr: -1.10% to 1.60%
- positive_alpha_rate: 56.25%

### 2d
- matured_picks: 58
- matured_scan_dates: 17
- latest_selected_scan_date: 2026-09-01
- latest_matured_scan_date: 2026-08-27
- mean_return: 1.11%
- median_return: 1.51%
- return_iqr: -1.41% to 3.38%
- return_p05_p95: -4.90% to 6.29%
- return_range: -10.42% to 9.76%
- hit_rate: 62.07%
- mean_alpha_vs_sector: 0.87%
- median_alpha_vs_sector: 0.91%
- alpha_iqr: -1.00% to 2.85%
- positive_alpha_rate: 65.52%

### 3d
- matured_picks: 52
- matured_scan_dates: 16
- latest_selected_scan_date: 2026-09-01
- latest_matured_scan_date: 2026-08-26
- mean_return: 1.35%
- median_return: 1.49%
- return_iqr: -1.32% to 4.00%
- return_p05_p95: -6.52% to 8.74%
- return_range: -11.22% to 11.59%
- hit_rate: 67.31%
- mean_alpha_vs_sector: 0.92%
- median_alpha_vs_sector: 0.61%
- alpha_iqr: -1.58% to 3.91%
- positive_alpha_rate: 59.62%

### 5d
- matured_picks: 40
- matured_scan_dates: 14
- latest_selected_scan_date: 2026-09-01
- latest_matured_scan_date: 2026-08-24
- mean_return: 1.16%
- median_return: 1.90%
- return_iqr: -1.58% to 5.56%
- return_p05_p95: -9.94% to 8.65%
- return_range: -18.97% to 14.32%
- hit_rate: 65.00%
- mean_alpha_vs_sector: 0.40%
- median_alpha_vs_sector: 0.09%
- alpha_iqr: -3.10% to 5.33%
- positive_alpha_rate: 52.50%

### 10d
- matured_picks: 23
- matured_scan_dates: 9
- latest_selected_scan_date: 2026-09-01
- latest_matured_scan_date: 2026-08-17
- mean_return: -2.90%
- median_return: -4.97%
- return_iqr: -9.05% to 4.68%
- return_p05_p95: -12.08% to 7.61%
- return_range: -18.62% to 10.89%
- hit_rate: 34.78%
- mean_alpha_vs_sector: -2.44%
- median_alpha_vs_sector: -3.07%
- alpha_iqr: -7.84% to 1.60%
- positive_alpha_rate: 30.43%

### 20d
- matured_picks: 0

### 60d
- matured_picks: 0

## 20d Timeframe Summary

- observations: 0

## 20d Opportunity Score Bands

- observations: 0

## Market Turn Diagnostics

- purpose: surface stale-leadership risk without hiding candidates from the scanner
- latest_scan_date: 2026-09-01

### Market ETF Snapshot
- latest_price_date: 2026-08-31
- SPY, 5d=0.47%, 10d=-0.73%, 20d=1.24%
- QQQ, 5d=1.48%, 10d=-1.80%, 20d=2.38%
- XLK, 5d=3.58%, 10d=-2.01%, 20d=4.75%
- SMH, 5d=1.80%, 10d=-6.30%, 20d=2.05%
- XLI, 5d=-2.16%, 10d=-6.01%, 20d=-4.38%
- XLB, 5d=-1.66%, 10d=0.86%, 20d=3.29%
- XLV, 5d=-2.38%, 10d=2.09%, 20d=5.12%

### Latest Selection Concentration
- selected_picks: 2
- by_sector: Information Technology=1, Materials=1
- by_slot: technology=1, materials=1
- median_opportunity_score: 0.4513
- opportunity_ge_0_40: 2/2
- opportunity_ge_0_45: 1/2
- opportunity_ge_0_50: 0/2

### Latest RS Deterioration
- observations: 2/2
- rs_deteriorating_ge_10pts: 1/2
- rs_deteriorating_ge_20pts: 0/2
- watch_only_candidates:
  - NEU: rs_vs_spy_5d_change=-14.9pts, rs_vs_group_5d_change=-13.0pts, opportunity=0.4437

### Matured 20d Outcomes By RS Change
- observations: 0
- note: RS change fields are unavailable until universe snapshots are refreshed.

## Portfolio Performance

- closed_trades: 47
- open_trades: 9
- realized_pnl: $22,524.42
- realized_return_on_cost: 6.53%
- unrealized_pnl: $2,211.30
- unrealized_return_on_cost: 3.50%
- total_pnl: $24,735.72
- total_return_on_cost: 6.06%

### Realized By Stock
- MRVL: trades=1, realized_pnl=$10,027.25, return_on_cost=59.82%, mean_trade_return=59.82%, range=59.82% to 59.82%, first_entry=2026-05-15, last_exit=2026-06-23
- MU: trades=1, realized_pnl=$3,900.25, return_on_cost=35.55%, mean_trade_return=35.55%, range=35.55% to 35.55%, first_entry=2026-05-15, last_exit=2026-06-04
- NBIS: trades=2, realized_pnl=$2,746.95, return_on_cost=8.36%, mean_trade_return=11.49%, range=1.17% to 21.80%, first_entry=2026-05-05, last_exit=2026-05-12
- TWLO: trades=1, realized_pnl=$2,262.00, return_on_cost=18.22%, mean_trade_return=18.22%, range=18.22% to 18.22%, first_entry=2026-07-01, last_exit=2026-08-10
- ASTS: trades=2, realized_pnl=$2,149.50, return_on_cost=14.04%, mean_trade_return=15.90%, range=-10.50% to 42.31%, first_entry=2026-05-11, last_exit=2026-06-01
- RKLB: trades=2, realized_pnl=$1,951.20, return_on_cost=10.48%, mean_trade_return=15.04%, range=5.67% to 24.40%, first_entry=2026-05-05, last_exit=2026-06-01
- OKTA: trades=1, realized_pnl=$1,113.50, return_on_cost=8.09%, mean_trade_return=8.09%, range=8.09% to 8.09%, first_entry=2026-07-28, last_exit=2026-08-14
- AGX: trades=2, realized_pnl=$1,007.65, return_on_cost=5.86%, mean_trade_return=5.94%, range=5.41% to 6.46%, first_entry=2026-06-01, last_exit=2026-06-29
- LFST: trades=1, realized_pnl=$909.00, return_on_cost=8.68%, mean_trade_return=8.68%, range=8.68% to 8.68%, first_entry=2026-07-21, last_exit=2026-08-14
- VICR: trades=2, realized_pnl=$901.00, return_on_cost=14.84%, mean_trade_return=14.08%, range=10.67% to 17.49%, first_entry=2026-05-12, last_exit=2026-06-30
- TALO: trades=1, realized_pnl=$867.30, return_on_cost=10.53%, mean_trade_return=10.53%, range=10.53% to 10.53%, first_entry=2026-06-22, last_exit=2026-07-23
- MXL: trades=1, realized_pnl=$867.00, return_on_cost=9.36%, mean_trade_return=9.36%, range=9.36% to 9.36%, first_entry=2026-06-05, last_exit=2026-06-29
- ENTG: trades=1, realized_pnl=$839.70, return_on_cost=19.48%, mean_trade_return=19.48%, range=19.48% to 19.48%, first_entry=2026-06-04, last_exit=2026-06-18
- CRSR: trades=1, realized_pnl=$655.50, return_on_cost=7.63%, mean_trade_return=7.63%, range=7.63% to 7.63%, first_entry=2026-06-29, last_exit=2026-07-20
- PANW: trades=1, realized_pnl=$486.45, return_on_cost=6.14%, mean_trade_return=6.14%, range=6.14% to 6.14%, first_entry=2026-07-03, last_exit=2026-07-20
- ANET: trades=1, realized_pnl=$392.43, return_on_cost=4.07%, mean_trade_return=4.07%, range=4.07% to 4.07%, first_entry=2026-06-16, last_exit=2026-08-03
- CRWD: trades=1, realized_pnl=$285.22, return_on_cost=4.54%, mean_trade_return=4.54%, range=4.54% to 4.54%, first_entry=2026-05-05, last_exit=2026-05-07
- APA: trades=1, realized_pnl=$230.00, return_on_cost=3.16%, mean_trade_return=3.16%, range=3.16% to 3.16%, first_entry=2026-05-11, last_exit=2026-05-15
- TFX: trades=1, realized_pnl=$229.27, return_on_cost=4.43%, mean_trade_return=4.43%, range=4.43% to 4.43%, first_entry=2026-05-05, last_exit=2026-05-07
- AA: trades=1, realized_pnl=$198.38, return_on_cost=10.61%, mean_trade_return=10.61%, range=10.61% to 10.61%, first_entry=2026-05-27, last_exit=2026-06-01
- OUST: trades=1, realized_pnl=$92.70, return_on_cost=2.13%, mean_trade_return=2.13%, range=2.13% to 2.13%, first_entry=2026-05-05, last_exit=2026-05-06
- CRGY: trades=1, realized_pnl=$60.00, return_on_cost=2.38%, mean_trade_return=2.38%, range=2.38% to 2.38%, first_entry=2026-05-12, last_exit=2026-05-15
- ICHR: trades=1, realized_pnl=$56.70, return_on_cost=2.88%, mean_trade_return=2.88%, range=2.88% to 2.88%, first_entry=2026-05-05, last_exit=2026-05-06
- AMZN: trades=1, realized_pnl=$52.50, return_on_cost=1.94%, mean_trade_return=1.94%, range=1.94% to 1.94%, first_entry=2026-05-05, last_exit=2026-05-06
- SM: trades=1, realized_pnl=$38.00, return_on_cost=3.08%, mean_trade_return=3.08%, range=3.08% to 3.08%, first_entry=2026-05-13, last_exit=2026-05-15
- LUNR: trades=1, realized_pnl=$0.00, return_on_cost=0.00%, mean_trade_return=0.00%, range=0.00% to 0.00%, first_entry=2026-06-01, last_exit=2026-06-01
- SLAB: trades=1, realized_pnl=$-35.50, return_on_cost=-0.32%, mean_trade_return=-0.32%, range=-0.32% to -0.32%, first_entry=2026-06-04, last_exit=2026-06-12
- ON: trades=1, realized_pnl=$-57.50, return_on_cost=-1.13%, mean_trade_return=-1.13%, range=-1.13% to -1.13%, first_entry=2026-05-08, last_exit=2026-05-12
- CTVA: trades=1, realized_pnl=$-132.30, return_on_cost=-4.82%, mean_trade_return=-4.82%, range=-4.82% to -4.82%, first_entry=2026-06-02, last_exit=2026-06-09
- AESI: trades=1, realized_pnl=$-156.00, return_on_cost=-2.37%, mean_trade_return=-2.37%, range=-2.37% to -2.37%, first_entry=2026-06-15, last_exit=2026-06-17
- GEN: trades=1, realized_pnl=$-177.18, return_on_cost=-7.10%, mean_trade_return=-7.10%, range=-7.10% to -7.10%, first_entry=2026-06-03, last_exit=2026-06-11
- XPO: trades=1, realized_pnl=$-347.40, return_on_cost=-7.90%, mean_trade_return=-7.90%, range=-7.90% to -7.90%, first_entry=2026-05-05, last_exit=2026-05-11
- CBT: trades=1, realized_pnl=$-348.50, return_on_cost=-7.54%, mean_trade_return=-7.54%, range=-7.54% to -7.54%, first_entry=2026-06-22, last_exit=2026-07-08
- CC: trades=1, realized_pnl=$-369.00, return_on_cost=-15.62%, mean_trade_return=-15.62%, range=-15.62% to -15.62%, first_entry=2026-05-06, last_exit=2026-06-30
- AMD: trades=1, realized_pnl=$-430.40, return_on_cost=-4.66%, mean_trade_return=-4.66%, range=-4.66% to -4.66%, first_entry=2026-05-11, last_exit=2026-05-12
- TDC: trades=1, realized_pnl=$-468.00, return_on_cost=-8.86%, mean_trade_return=-8.86%, range=-8.86% to -8.86%, first_entry=2026-06-04, last_exit=2026-06-17
- CHEOY: trades=1, realized_pnl=$-552.00, return_on_cost=-6.64%, mean_trade_return=-6.64%, range=-6.64% to -6.64%, first_entry=2026-06-29, last_exit=2026-07-23
- SMCI: trades=1, realized_pnl=$-910.00, return_on_cost=-19.21%, mean_trade_return=-19.21%, range=-19.21% to -19.21%, first_entry=2026-06-03, last_exit=2026-06-09
- MORN: trades=1, realized_pnl=$-921.15, return_on_cost=-14.87%, mean_trade_return=-14.87%, range=-14.87% to -14.87%, first_entry=2026-05-05, last_exit=2026-05-11
- HIMS: trades=1, realized_pnl=$-1,010.10, return_on_cost=-8.73%, mean_trade_return=-8.73%, range=-8.73% to -8.73%, first_entry=2026-08-05, last_exit=2026-08-18
- VSH: trades=1, realized_pnl=$-1,533.50, return_on_cost=-26.74%, mean_trade_return=-26.74%, range=-26.74% to -26.74%, first_entry=2026-06-11, last_exit=2026-07-08
- TTMI: trades=1, realized_pnl=$-2,346.50, return_on_cost=-24.90%, mean_trade_return=-24.90%, range=-24.90% to -24.90%, first_entry=2026-06-30, last_exit=2026-07-14

### Unrealized Open Positions
- NEO: shares=850, entry_date=2026-08-14, entry=16.68, latest=17.91 (2026-08-31), unrealized_pnl=$1,049.75, return=7.41%
- RNG: shares=200, entry_date=2026-08-18, entry=65.08, latest=69.62 (2026-08-31), unrealized_pnl=$908.00, return=6.98%
- DINO: shares=115, entry_date=2026-08-21, entry=95.55, latest=101.61 (2026-08-31), unrealized_pnl=$696.90, return=6.34%
- MU: shares=11, entry_date=2026-07-02, entry=964.63, latest=958.73 (2026-08-31), unrealized_pnl=$-64.90, return=-0.61%
- APP: shares=45, entry_date=2026-08-18, entry=320.47, latest=312.06 (2026-08-31), unrealized_pnl=$-378.45, return=-2.62%
- RKLB: shares=110, entry_date=2026-06-01, entry=108.44, latest=n/a (n/a), unrealized_pnl=n/a, return=n/a
- ASTS: shares=180, entry_date=2026-07-02, entry=74.85, latest=n/a (n/a), unrealized_pnl=n/a, return=n/a
- NBIS: shares=50, entry_date=2026-07-02, entry=237.00, latest=n/a (n/a), unrealized_pnl=n/a, return=n/a
- SLS: shares=150, entry_date=2026-07-02, entry=15.25, latest=n/a (n/a), unrealized_pnl=n/a, return=n/a

## Best And Worst Picks

### 1d
- best:
  - MAN (2026-08-17): return=4.77%, alpha_vs_sector=6.25%
  - INSW (2026-08-26): return=5.97%, alpha_vs_sector=6.19%
  - NSP (2026-08-17): return=3.71%, alpha_vs_sector=5.18%
- worst:
  - PARR (2026-08-19): return=-6.07%, alpha_vs_sector=-6.34%
  - MOG-A (2026-08-19): return=-7.16%, alpha_vs_sector=-5.96%
  - PENG (2026-08-07): return=-5.56%, alpha_vs_sector=-4.68%

### 2d
- best:
  - AMN (2026-08-05): return=9.76%, alpha_vs_sector=8.83%
  - PENG (2026-08-12): return=8.63%, alpha_vs_sector=8.02%
  - MAN (2026-08-17): return=4.67%, alpha_vs_sector=7.01%
- worst:
  - PGNY (2026-08-05): return=-10.42%, alpha_vs_sector=-11.35%
  - MOG-A (2026-08-19): return=-5.30%, alpha_vs_sector=-4.37%
  - PANW (2026-08-21): return=-5.02%, alpha_vs_sector=-4.16%

### 3d
- best:
  - MAN (2026-08-17): return=6.99%, alpha_vs_sector=10.51%
  - PENG (2026-08-11): return=11.29%, alpha_vs_sector=9.18%
  - AMN (2026-08-05): return=11.59%, alpha_vs_sector=8.98%
- worst:
  - PGNY (2026-08-05): return=-11.22%, alpha_vs_sector=-13.83%
  - PARR (2026-08-13): return=-1.98%, alpha_vs_sector=-6.28%
  - MOG-A (2026-08-19): return=-7.58%, alpha_vs_sector=-5.96%

### 5d
- best:
  - PENG (2026-08-10): return=14.32%, alpha_vs_sector=12.17%
  - MAN (2026-08-17): return=8.14%, alpha_vs_sector=12.07%
  - CXW (2026-08-13): return=6.04%, alpha_vs_sector=9.28%
- worst:
  - PGNY (2026-08-05): return=-18.97%, alpha_vs_sector=-21.57%
  - PARR (2026-08-13): return=-11.80%, alpha_vs_sector=-16.20%
  - PENG (2026-08-12): return=-9.84%, alpha_vs_sector=-7.08%

### 10d
- best:
  - MAN (2026-08-17): return=10.89%, alpha_vs_sector=16.89%
  - NSP (2026-08-17): return=5.73%, alpha_vs_sector=11.73%
  - CXW (2026-08-13): return=6.93%, alpha_vs_sector=10.69%
- worst:
  - PGNY (2026-08-05): return=-18.62%, alpha_vs_sector=-25.63%
  - CLF (2026-08-05): return=-11.70%, alpha_vs_sector=-11.47%
  - PENG (2026-08-11): return=-12.12%, alpha_vs_sector=-9.79%

### 20d
No matured picks.

### 60d
No matured picks.

## Repeated Winners And Losers

### 1d
- repeated_winners:
  - PBF: n=3, mean_return=1.67%, mean_alpha_vs_sector=1.66%
  - PENG: n=6, mean_return=1.55%, mean_alpha_vs_sector=1.11%
  - CXW: n=4, mean_return=0.40%, mean_alpha_vs_sector=1.03%
- repeated_losers:
  - DINO: n=2, mean_return=-0.02%, mean_alpha_vs_sector=-0.85%
  - PARR: n=7, mean_return=-0.14%, mean_alpha_vs_sector=-0.57%
  - DD: n=5, mean_return=-0.63%, mean_alpha_vs_sector=-0.14%

### 2d
- repeated_winners:
  - PENG: n=6, mean_return=3.44%, mean_alpha_vs_sector=2.65%
  - CXW: n=3, mean_return=1.35%, mean_alpha_vs_sector=2.53%
  - PBF: n=3, mean_return=2.70%, mean_alpha_vs_sector=1.98%
- repeated_losers:
  - DD: n=4, mean_return=-0.03%, mean_alpha_vs_sector=0.19%
  - OII: n=6, mean_return=1.66%, mean_alpha_vs_sector=0.83%
  - GEO: n=3, mean_return=0.56%, mean_alpha_vs_sector=1.02%

### 3d
- repeated_winners:
  - PENG: n=6, mean_return=4.79%, mean_alpha_vs_sector=3.79%
  - GEO: n=3, mean_return=1.17%, mean_alpha_vs_sector=2.61%
  - CXW: n=2, mean_return=0.13%, mean_alpha_vs_sector=2.18%
- repeated_losers:
  - DD: n=4, mean_return=-0.86%, mean_alpha_vs_sector=-0.14%
  - OII: n=5, mean_return=1.27%, mean_alpha_vs_sector=0.13%
  - DINO: n=2, mean_return=2.38%, mean_alpha_vs_sector=0.60%

### 5d
- repeated_winners:
  - GEO: n=2, mean_return=1.08%, mean_alpha_vs_sector=4.12%
  - PENG: n=6, mean_return=4.54%, mean_alpha_vs_sector=3.81%
  - PBF: n=2, mean_return=3.67%, mean_alpha_vs_sector=1.60%
- repeated_losers:
  - DD: n=2, mean_return=-3.00%, mean_alpha_vs_sector=-3.45%
  - PARR: n=3, mean_return=-2.01%, mean_alpha_vs_sector=-3.33%
  - OII: n=3, mean_return=-0.34%, mean_alpha_vs_sector=-1.66%

### 10d
- repeated_winners:
  - PENG: n=6, mean_return=-9.59%, mean_alpha_vs_sector=-7.24%
- repeated_losers:
  - PENG: n=6, mean_return=-9.59%, mean_alpha_vs_sector=-7.24%

### 20d
No matured picks.

### 60d
No matured picks.

## Recent Scan Dates

### 2026-09-01
- picks: PANW, NEU

### 2026-08-31
- picks: CDW, HPE

### 2026-08-28
- picks: OII, WHD, PARR, DD, CXW, AVNT
- 1d: median_return=-0.20%, median_alpha_vs_sector=-0.65%, winners=3/6, range=-1.75% to 2.31%

### 2026-08-27
- picks: OII, WHD, PARR, CXW, DXPE, AVNT
- 1d: median_return=0.49%, median_alpha_vs_sector=1.00%, winners=4/6, range=-3.57% to 2.35%
- 2d: median_return=0.31%, median_alpha_vs_sector=0.15%, winners=3/6, range=-2.63% to 3.74%

### 2026-08-26
- picks: OII, PARR, INSW, VSEC, DD, CXW
- 1d: median_return=1.14%, median_alpha_vs_sector=1.66%, winners=4/6, range=-2.60% to 5.97%
- 2d: median_return=0.71%, median_alpha_vs_sector=1.40%, winners=3/6, range=-4.88% to 5.49%
- 3d: median_return=1.02%, median_alpha_vs_sector=1.24%, winners=3/6, range=-8.03% to 5.48%

### 2026-08-25
- picks: OII, PARR, PBF, DD, GEO, SW
- 1d: median_return=1.66%, median_alpha_vs_sector=1.03%, winners=5/6, range=-0.58% to 4.84%
- 2d: median_return=2.32%, median_alpha_vs_sector=2.46%, winners=5/6, range=-1.62% to 4.36%
- 3d: median_return=1.03%, median_alpha_vs_sector=1.26%, winners=5/6, range=-2.31% to 7.17%

### 2026-08-24
- picks: PARR, OII, PBF, AXON, DD, GEO
- 1d: median_return=-0.57%, median_alpha_vs_sector=0.42%, winners=1/6, range=-4.32% to 2.62%
- 2d: median_return=1.31%, median_alpha_vs_sector=1.00%, winners=5/6, range=-1.66% to 1.94%
- 3d: median_return=1.52%, median_alpha_vs_sector=2.30%, winners=5/6, range=-1.32% to 3.65%
- 5d: median_return=0.71%, median_alpha_vs_sector=1.12%, winners=3/6, range=-5.19% to 7.52%

### 2026-08-21
- picks: PANW, CDW
- 1d: median_return=-2.93%, median_alpha_vs_sector=-1.15%, winners=0/2, range=-3.91% to -1.95%
- 2d: median_return=-0.75%, median_alpha_vs_sector=0.11%, winners=1/2, range=-5.02% to 3.53%
- 3d: median_return=-1.06%, median_alpha_vs_sector=-0.80%, winners=1/2, range=-5.19% to 3.07%
- 5d: median_return=5.72%, median_alpha_vs_sector=4.42%, winners=2/2, range=3.83% to 7.61%

### 2026-08-20
- picks: GEN, DELL
- 1d: median_return=2.18%, median_alpha_vs_sector=2.06%, winners=2/2, range=1.68% to 2.68%
- 2d: median_return=1.48%, median_alpha_vs_sector=3.14%, winners=1/2, range=-0.37% to 3.32%
- 3d: median_return=3.94%, median_alpha_vs_sector=4.68%, winners=2/2, range=3.85% to 4.03%
- 5d: median_return=8.44%, median_alpha_vs_sector=5.43%, winners=2/2, range=8.25% to 8.62%

### 2026-08-19
- picks: DINO, PARR, OII, MOG-A
- 1d: median_return=-4.04%, median_alpha_vs_sector=-4.12%, winners=0/4, range=-7.16% to -0.38%
- 2d: median_return=1.53%, median_alpha_vs_sector=1.43%, winners=3/4, range=-5.30% to 2.85%
- 3d: median_return=-3.93%, median_alpha_vs_sector=-3.19%, winners=1/4, range=-7.58% to 0.59%
- 5d: median_return=-3.79%, median_alpha_vs_sector=-1.99%, winners=1/4, range=-6.55% to 2.01%

## Recent Picks

### PANW
- scan_date: 2026-09-01
- sector: Information Technology
- selected_rank: 1

### NEU
- scan_date: 2026-09-01
- sector: Materials
- selected_rank: 2

### CDW
- scan_date: 2026-08-31
- sector: Information Technology
- selected_rank: 1

### HPE
- scan_date: 2026-08-31
- sector: Information Technology
- selected_rank: 2

### OII
- scan_date: 2026-08-28
- sector: Energy
- selected_rank: 1
- 1d: return=2.31%, alpha_vs_sector=0.27%

### WHD
- scan_date: 2026-08-28
- sector: Energy
- selected_rank: 2
- 1d: return=0.96%, alpha_vs_sector=-1.08%

### PARR
- scan_date: 2026-08-28
- sector: Energy
- selected_rank: 3
- 1d: return=1.36%, alpha_vs_sector=-0.69%

### DD
- scan_date: 2026-08-28
- sector: Materials
- selected_rank: 4
- 1d: return=-1.74%, alpha_vs_sector=-0.82%

### CXW
- scan_date: 2026-08-28
- sector: Industrials
- selected_rank: 5
- 1d: return=-1.75%, alpha_vs_sector=-0.61%

### AVNT
- scan_date: 2026-08-28
- sector: Materials
- selected_rank: 6
- 1d: return=-1.37%, alpha_vs_sector=-0.44%

### OII
- scan_date: 2026-08-27
- sector: Energy
- selected_rank: 1
- 1d: return=-1.21%, alpha_vs_sector=-1.84%
- 2d: return=1.07%, alpha_vs_sector=-1.61%

### WHD
- scan_date: 2026-08-27
- sector: Energy
- selected_rank: 2
- 1d: return=1.63%, alpha_vs_sector=1.01%
- 2d: return=2.61%, alpha_vs_sector=-0.07%

### PARR
- scan_date: 2026-08-27
- sector: Energy
- selected_rank: 3
- 1d: return=2.35%, alpha_vs_sector=1.72%
- 2d: return=3.74%, alpha_vs_sector=1.06%

### CXW
- scan_date: 2026-08-27
- sector: Industrials
- selected_rank: 4
- 1d: return=0.06%, alpha_vs_sector=0.99%
- 2d: return=-1.69%, alpha_vs_sector=0.37%

### DXPE
- scan_date: 2026-08-27
- sector: Industrials
- selected_rank: 5
- 1d: return=-3.57%, alpha_vs_sector=-2.65%
- 2d: return=-2.63%, alpha_vs_sector=-0.58%

### AVNT
- scan_date: 2026-08-27
- sector: Materials
- selected_rank: 6
- 1d: return=0.93%, alpha_vs_sector=1.02%
- 2d: return=-0.45%, alpha_vs_sector=0.56%

### OII
- scan_date: 2026-08-26
- sector: Energy
- selected_rank: 1
- 1d: return=3.20%, alpha_vs_sector=3.43%
- 2d: return=1.95%, alpha_vs_sector=1.55%
- 3d: return=4.31%, alpha_vs_sector=1.86%

### PARR
- scan_date: 2026-08-26
- sector: Energy
- selected_rank: 2
- 1d: return=1.68%, alpha_vs_sector=1.90%
- 2d: return=4.06%, alpha_vs_sector=3.66%
- 3d: return=5.48%, alpha_vs_sector=3.03%

### INSW
- scan_date: 2026-08-26
- sector: Energy
- selected_rank: 3
- 1d: return=5.97%, alpha_vs_sector=6.19%
- 2d: return=5.49%, alpha_vs_sector=5.09%
- 3d: return=5.31%, alpha_vs_sector=2.86%

### VSEC
- scan_date: 2026-08-26
- sector: Industrials
- selected_rank: 4
- 1d: return=-2.60%, alpha_vs_sector=-1.75%
- 2d: return=-4.88%, alpha_vs_sector=-3.11%
- 3d: return=-8.03%, alpha_vs_sector=-5.14%
