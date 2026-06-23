# Scan Performance

- benchmark: sector
- scope: latest_model
- selection_source: shortlist_model
- model_name: xgboost_model
- model_generated_at: 2026-06-23T01:41:57+00:00
- recent_scan_dates: 28
- scan_dates: 28
- selected_rows: 168
- scan_date_min: 2026-04-15
- scan_date_max: 2026-06-05

## Horizon Summary

### 2d
- matured_picks: 168
- matured_scan_dates: 28
- mean_return: 1.66%
- median_return: 1.05%
- return_iqr: -2.66% to 4.59%
- return_p05_p95: -7.93% to 12.44%
- return_range: -20.98% to 27.84%
- hit_rate: 59.52%
- mean_alpha_vs_sector: 0.93%
- median_alpha_vs_sector: -0.11%
- alpha_iqr: -2.39% to 3.51%
- positive_alpha_rate: 47.62%

### 5d
- matured_picks: 168
- matured_scan_dates: 28
- mean_return: 5.71%
- median_return: 5.04%
- return_iqr: -1.91% to 9.93%
- return_p05_p95: -9.07% to 26.74%
- return_range: -13.86% to 53.98%
- hit_rate: 69.64%
- mean_alpha_vs_sector: 3.60%
- median_alpha_vs_sector: 2.18%
- alpha_iqr: -2.41% to 8.07%
- positive_alpha_rate: 60.12%

### 10d
- matured_picks: 168
- matured_scan_dates: 28
- mean_return: 12.12%
- median_return: 8.60%
- return_iqr: 0.40% to 19.19%
- return_p05_p95: -8.41% to 37.82%
- return_range: -21.90% to 206.77%
- hit_rate: 75.60%
- mean_alpha_vs_sector: 7.91%
- median_alpha_vs_sector: 3.85%
- alpha_iqr: -1.98% to 12.66%
- positive_alpha_rate: 66.07%

### 20d
- matured_picks: 162
- matured_scan_dates: 27
- mean_return: 16.19%
- median_return: 9.33%
- return_iqr: -0.19% to 24.71%
- return_p05_p95: -10.75% to 55.96%
- return_range: -20.28% to 298.59%
- hit_rate: 74.07%
- mean_alpha_vs_sector: 9.65%
- median_alpha_vs_sector: 4.08%
- alpha_iqr: -3.24% to 13.36%
- positive_alpha_rate: 66.05%

### 60d
- matured_picks: 0

## 20d Opportunity Score Bands

- score: opportunity_score
- return: fwd_return_20d
- alpha: alpha_vs_sector_20d
- observations: 162

- score < 0.30: n=0, pick_share=0.00%
- 0.30 <= score < 0.35: n=7, pick_share=4.32%, mean_return=14.00%, median_return=15.24%, hit_rate=71.43%, mean_alpha=6.81%, median_alpha=3.65%, positive_alpha_rate=71.43%
- 0.35 <= score < 0.40: n=31, pick_share=19.14%, mean_return=19.17%, median_return=5.86%, hit_rate=61.29%, mean_alpha=13.68%, median_alpha=-0.17%, positive_alpha_rate=48.39%
- 0.40 <= score < 0.45: n=45, pick_share=27.78%, mean_return=20.92%, median_return=15.07%, hit_rate=88.89%, mean_alpha=13.58%, median_alpha=8.16%, positive_alpha_rate=80.00%
- 0.45 <= score < 0.50: n=53, pick_share=32.72%, mean_return=15.99%, median_return=14.02%, hit_rate=75.47%, mean_alpha=7.80%, median_alpha=4.70%, positive_alpha_rate=67.92%
- score >= 0.50: n=26, pick_share=16.05%, mean_return=5.42%, median_return=4.89%, hit_rate=61.54%, mean_alpha=2.59%, median_alpha=1.18%, positive_alpha_rate=57.69%

## Best And Worst Picks

### 2d
- best:
  - RAMP (2026-05-15): return=27.24%, alpha_vs_sector=28.05%
  - AOSL (2026-04-16): return=27.84%, alpha_vs_sector=26.17%
  - MXL (2026-04-15): return=19.35%, alpha_vs_sector=16.66%
- worst:
  - CC (2026-05-05): return=-20.98%, alpha_vs_sector=-20.73%
  - MXL (2026-06-05): return=-10.27%, alpha_vs_sector=-10.53%
  - DBX (2026-05-08): return=-8.93%, alpha_vs_sector=-9.15%

### 5d
- best:
  - MXL (2026-04-15): return=53.98%, alpha_vs_sector=48.79%
  - VICR (2026-04-15): return=36.46%, alpha_vs_sector=36.54%
  - VSH (2026-05-18): return=36.32%, alpha_vs_sector=30.14%
- worst:
  - LITE (2026-05-01): return=-7.56%, alpha_vs_sector=-15.55%
  - CC (2026-05-05): return=-13.01%, alpha_vs_sector=-14.20%
  - CRSR (2026-05-08): return=-12.22%, alpha_vs_sector=-13.04%

### 10d
- best:
  - MXL (2026-04-15): return=206.77%, alpha_vs_sector=200.91%
  - VSH (2026-05-19): return=74.35%, alpha_vs_sector=62.11%
  - OKTA (2026-05-18): return=57.12%, alpha_vs_sector=43.66%
- worst:
  - CC (2026-05-05): return=-21.90%, alpha_vs_sector=-17.54%
  - LITE (2026-05-04): return=-9.34%, alpha_vs_sector=-16.94%
  - AMR (2026-06-05): return=-13.48%, alpha_vs_sector=-15.44%

### 20d
- best:
  - MXL (2026-04-15): return=298.59%, alpha_vs_sector=280.93%
  - VSH (2026-05-15): return=70.87%, alpha_vs_sector=62.06%
  - VICR (2026-04-15): return=59.25%, alpha_vs_sector=57.83%
- worst:
  - LITE (2026-05-01): return=-4.73%, alpha_vs_sector=-25.67%
  - CC (2026-05-05): return=-20.28%, alpha_vs_sector=-20.47%
  - NOG (2026-05-04): return=-18.19%, alpha_vs_sector=-15.89%

### 60d
No matured picks.

## Repeated Winners And Losers

### 2d
- repeated_winners:
  - RAMP: n=2, mean_return=14.09%, mean_alpha_vs_sector=13.27%
  - AOSL: n=2, mean_return=15.50%, mean_alpha_vs_sector=12.25%
  - VICR: n=2, mean_return=8.44%, mean_alpha_vs_sector=7.35%
- repeated_losers:
  - CRSR: n=4, mean_return=-4.19%, mean_alpha_vs_sector=-4.78%
  - ALB: n=3, mean_return=-5.80%, mean_alpha_vs_sector=-3.86%
  - MARA: n=2, mean_return=-3.45%, mean_alpha_vs_sector=-3.42%

### 5d
- repeated_winners:
  - MXL: n=2, mean_return=29.18%, mean_alpha_vs_sector=25.34%
  - VICR: n=2, mean_return=24.27%, mean_alpha_vs_sector=23.73%
  - VSH: n=5, mean_return=22.41%, mean_alpha_vs_sector=19.67%
- repeated_losers:
  - CRSR: n=4, mean_return=-10.47%, mean_alpha_vs_sector=-10.75%
  - ALB: n=3, mean_return=-5.52%, mean_alpha_vs_sector=-4.96%
  - HLIT: n=4, mean_return=-0.76%, mean_alpha_vs_sector=-4.85%

### 10d
- repeated_winners:
  - MXL: n=2, mean_return=112.97%, mean_alpha_vs_sector=106.76%
  - VSH: n=5, mean_return=53.50%, mean_alpha_vs_sector=43.73%
  - VICR: n=2, mean_return=33.52%, mean_alpha_vs_sector=31.70%
- repeated_losers:
  - CE: n=3, mean_return=-11.78%, mean_alpha_vs_sector=-8.37%
  - CC: n=8, mean_return=-8.47%, mean_alpha_vs_sector=-6.69%
  - ALB: n=3, mean_return=-4.98%, mean_alpha_vs_sector=-5.96%

### 20d
- repeated_winners:
  - VSH: n=5, mean_return=61.30%, mean_alpha_vs_sector=55.59%
  - SYNA: n=3, mean_return=53.55%, mean_alpha_vs_sector=37.66%
  - SNDK: n=18, mean_return=38.14%, mean_alpha_vs_sector=27.99%
- repeated_losers:
  - LITE: n=10, mean_return=4.62%, mean_alpha_vs_sector=-11.73%
  - CE: n=3, mean_return=-13.41%, mean_alpha_vs_sector=-11.46%
  - ALB: n=3, mean_return=-7.56%, mean_alpha_vs_sector=-11.00%

### 60d
No matured picks.

## Recent Scan Dates

### 2026-06-05
- picks: CRSR, SLAB, MXL, AMR, VICR, HCC
- 2d: median_return=-4.80%, median_alpha_vs_sector=-5.08%, winners=2/6, range=-10.27% to 4.59%
- 5d: median_return=-0.06%, median_alpha_vs_sector=-2.84%, winners=3/6, range=-6.44% to 12.08%
- 10d: median_return=1.66%, median_alpha_vs_sector=-4.91%, winners=4/6, range=-13.48% to 34.86%

### 2026-05-21
- picks: SNDK, ADEA, DDOG, CC, AA, CAR
- 2d: median_return=4.33%, median_alpha_vs_sector=2.26%, winners=6/6, range=2.57% to 12.52%
- 5d: median_return=10.70%, median_alpha_vs_sector=4.72%, winners=6/6, range=1.52% to 17.16%
- 10d: median_return=7.93%, median_alpha_vs_sector=6.84%, winners=5/6, range=-4.38% to 12.10%
- 20d: median_return=9.82%, median_alpha_vs_sector=4.20%, winners=5/6, range=-11.95% to 47.43%

### 2026-05-20
- picks: SNDK, ADEA, DOCN, AA, SXT, CC
- 2d: median_return=1.01%, median_alpha_vs_sector=-0.49%, winners=4/6, range=-4.04% to 11.29%
- 5d: median_return=2.70%, median_alpha_vs_sector=-1.18%, winners=5/6, range=-5.12% to 17.89%
- 10d: median_return=17.19%, median_alpha_vs_sector=8.70%, winners=4/6, range=-2.69% to 26.36%
- 20d: median_return=3.84%, median_alpha_vs_sector=-2.30%, winners=3/6, range=-7.44% to 56.89%

### 2026-05-19
- picks: VSH, SNDK, AA, CLSK, SXT, ESI
- 2d: median_return=7.40%, median_alpha_vs_sector=5.58%, winners=6/6, range=0.71% to 15.12%
- 5d: median_return=14.90%, median_alpha_vs_sector=10.23%, winners=6/6, range=4.90% to 33.50%
- 10d: median_return=25.34%, median_alpha_vs_sector=16.07%, winners=6/6, range=0.52% to 74.35%
- 20d: median_return=16.40%, median_alpha_vs_sector=10.49%, winners=4/6, range=-3.26% to 64.43%

### 2026-05-18
- picks: VSH, SNDK, OKTA, AA, ALB, ESI
- 2d: median_return=2.39%, median_alpha_vs_sector=2.68%, winners=5/6, range=-3.15% to 8.69%
- 5d: median_return=13.61%, median_alpha_vs_sector=9.80%, winners=5/6, range=-0.60% to 36.32%
- 10d: median_return=32.25%, median_alpha_vs_sector=24.16%, winners=5/6, range=-2.13% to 70.07%
- 20d: median_return=22.66%, median_alpha_vs_sector=16.71%, winners=5/6, range=-5.48% to 63.22%

### 2026-05-15
- picks: VSH, RAMP, CLSK, ALB, AA, MTRN
- 2d: median_return=0.22%, median_alpha_vs_sector=1.63%, winners=3/6, range=-6.20% to 27.24%
- 5d: median_return=17.98%, median_alpha_vs_sector=16.82%, winners=5/6, range=-4.88% to 27.11%
- 10d: median_return=25.53%, median_alpha_vs_sector=19.38%, winners=5/6, range=-5.13% to 50.39%
- 20d: median_return=24.83%, median_alpha_vs_sector=18.23%, winners=5/6, range=-6.36% to 70.87%

### 2026-05-14
- picks: VSH, SNDK, MARA, SXT, AA, ALB
- 2d: median_return=-4.40%, median_alpha_vs_sector=-1.56%, winners=0/6, range=-8.35% to -2.74%
- 5d: median_return=1.17%, median_alpha_vs_sector=3.01%, winners=4/6, range=-11.09% to 11.54%
- 10d: median_return=12.90%, median_alpha_vs_sector=8.97%, winners=4/6, range=-7.68% to 37.01%
- 20d: median_return=5.90%, median_alpha_vs_sector=4.03%, winners=5/6, range=-10.82% to 56.05%

### 2026-05-13
- picks: VSH, SNDK, CLSK, SXT, AA, ESI
- 2d: median_return=-3.02%, median_alpha_vs_sector=-1.75%, winners=0/6, range=-8.64% to -1.43%
- 5d: median_return=-2.61%, median_alpha_vs_sector=0.63%, winners=2/6, range=-8.49% to 15.79%
- 10d: median_return=11.93%, median_alpha_vs_sector=9.78%, winners=5/6, range=-5.40% to 36.39%
- 20d: median_return=14.43%, median_alpha_vs_sector=13.44%, winners=5/6, range=-2.96% to 51.96%

### 2026-05-12
- picks: APA, CRSR, SNDK, RMBS, CC, CRGY
- 2d: median_return=-0.86%, median_alpha_vs_sector=-2.48%, winners=3/6, range=-4.77% to 0.41%
- 5d: median_return=-4.65%, median_alpha_vs_sector=-4.44%, winners=2/6, range=-10.21% to 9.36%
- 10d: median_return=4.12%, median_alpha_vs_sector=1.99%, winners=3/6, range=-8.80% to 30.93%
- 20d: median_return=4.24%, median_alpha_vs_sector=3.24%, winners=4/6, range=-17.67% to 15.07%

### 2026-05-11
- picks: APA, CRSR, CE, SNDK, RMBS, CC
- 2d: median_return=-0.21%, median_alpha_vs_sector=-0.18%, winners=3/6, range=-6.48% to 0.74%
- 5d: median_return=-8.59%, median_alpha_vs_sector=-5.65%, winners=1/6, range=-13.86% to 9.40%
- 10d: median_return=2.23%, median_alpha_vs_sector=-1.59%, winners=4/6, range=-12.51% to 16.89%
- 20d: median_return=3.04%, median_alpha_vs_sector=2.03%, winners=3/6, range=-19.95% to 9.17%

## Recent Picks

### CRSR
- scan_date: 2026-06-05
- sector: Information Technology
- selected_rank: 1
- 2d: return=-6.66%, alpha_vs_sector=-6.92%
- 5d: return=-6.44%, alpha_vs_sector=-8.93%
- 10d: return=2.55%, alpha_vs_sector=-4.02%

### SLAB
- scan_date: 2026-06-05
- sector: Information Technology
- selected_rank: 2
- 2d: return=0.32%, alpha_vs_sector=0.06%
- 5d: return=0.64%, alpha_vs_sector=-1.85%
- 10d: return=0.77%, alpha_vs_sector=-5.81%

### MXL
- scan_date: 2026-06-05
- sector: Information Technology
- selected_rank: 3
- 2d: return=-10.27%, alpha_vs_sector=-10.53%
- 5d: return=4.37%, alpha_vs_sector=1.88%
- 10d: return=19.18%, alpha_vs_sector=12.61%

### AMR
- scan_date: 2026-06-05
- sector: Materials
- selected_rank: 4
- 2d: return=-4.35%, alpha_vs_sector=-4.62%
- 5d: return=-0.77%, alpha_vs_sector=-3.83%
- 10d: return=-13.48%, alpha_vs_sector=-15.44%

### VICR
- scan_date: 2026-06-05
- sector: Industrials
- selected_rank: 5
- 2d: return=4.59%, alpha_vs_sector=3.77%
- 5d: return=12.08%, alpha_vs_sector=10.93%
- 10d: return=34.86%, alpha_vs_sector=30.49%

### HCC
- scan_date: 2026-06-05
- sector: Materials
- selected_rank: 6
- 2d: return=-5.25%, alpha_vs_sector=-5.53%
- 5d: return=-1.81%, alpha_vs_sector=-4.87%
- 10d: return=-11.97%, alpha_vs_sector=-13.92%

### SNDK
- scan_date: 2026-05-21
- sector: Information Technology
- selected_rank: 1
- 2d: return=3.07%, alpha_vs_sector=-0.59%
- 5d: return=9.90%, alpha_vs_sector=2.95%
- 10d: return=1.11%, alpha_vs_sector=0.16%
- 20d: return=47.43%, alpha_vs_sector=39.84%

### ADEA
- scan_date: 2026-05-21
- sector: Information Technology
- selected_rank: 2
- 2d: return=7.37%, alpha_vs_sector=3.71%
- 5d: return=1.52%, alpha_vs_sector=-5.44%
- 10d: return=10.14%, alpha_vs_sector=9.19%
- 20d: return=20.90%, alpha_vs_sector=13.31%

### DDOG
- scan_date: 2026-05-21
- sector: Information Technology
- selected_rank: 3
- 2d: return=2.57%, alpha_vs_sector=-1.09%
- 5d: return=13.44%, alpha_vs_sector=6.49%
- 10d: return=7.37%, alpha_vs_sector=6.42%
- 20d: return=1.53%, alpha_vs_sector=-6.06%

### CC
- scan_date: 2026-05-21
- sector: Materials
- selected_rank: 4
- 2d: return=3.08%, alpha_vs_sector=1.14%
- 5d: return=3.36%, alpha_vs_sector=1.10%
- 10d: return=-4.38%, alpha_vs_sector=-5.60%
- 20d: return=0.09%, alpha_vs_sector=-3.11%

### AA
- scan_date: 2026-05-21
- sector: Materials
- selected_rank: 5
- 2d: return=12.52%, alpha_vs_sector=10.59%
- 5d: return=17.16%, alpha_vs_sector=14.90%
- 10d: return=8.48%, alpha_vs_sector=7.26%
- 20d: return=-11.95%, alpha_vs_sector=-15.15%

### CAR
- scan_date: 2026-05-21
- sector: Industrials
- selected_rank: 6
- 2d: return=5.59%, alpha_vs_sector=3.38%
- 5d: return=11.50%, alpha_vs_sector=9.97%
- 10d: return=12.10%, alpha_vs_sector=9.96%
- 20d: return=18.12%, alpha_vs_sector=11.51%

### SNDK
- scan_date: 2026-05-20
- sector: Information Technology
- selected_rank: 1
- 2d: return=6.19%, alpha_vs_sector=4.35%
- 5d: return=17.89%, alpha_vs_sector=12.40%
- 10d: return=26.36%, alpha_vs_sector=17.31%
- 20d: return=56.89%, alpha_vs_sector=48.81%

### ADEA
- scan_date: 2026-05-20
- sector: Information Technology
- selected_rank: 2
- 2d: return=1.70%, alpha_vs_sector=-0.13%
- 5d: return=3.18%, alpha_vs_sector=-2.30%
- 10d: return=22.76%, alpha_vs_sector=13.71%
- 20d: return=20.67%, alpha_vs_sector=12.60%

### DOCN
- scan_date: 2026-05-20
- sector: Information Technology
- selected_rank: 3
- 2d: return=-1.03%, alpha_vs_sector=-2.87%
- 5d: return=-5.12%, alpha_vs_sector=-10.60%
- 10d: return=12.73%, alpha_vs_sector=3.69%
- 20d: return=8.22%, alpha_vs_sector=0.15%

### AA
- scan_date: 2026-05-20
- sector: Materials
- selected_rank: 4
- 2d: return=11.29%, alpha_vs_sector=10.14%
- 5d: return=17.84%, alpha_vs_sector=14.54%
- 10d: return=21.64%, alpha_vs_sector=17.82%
- 20d: return=-7.44%, alpha_vs_sector=-11.64%

### SXT
- scan_date: 2026-05-20
- sector: Materials
- selected_rank: 5
- 2d: return=0.31%, alpha_vs_sector=-0.84%
- 5d: return=2.21%, alpha_vs_sector=-1.09%
- 10d: return=-1.14%, alpha_vs_sector=-4.96%
- 20d: return=-0.53%, alpha_vs_sector=-4.74%

### CC
- scan_date: 2026-05-20
- sector: Materials
- selected_rank: 6
- 2d: return=-4.04%, alpha_vs_sector=-5.19%
- 5d: return=2.02%, alpha_vs_sector=-1.28%
- 10d: return=-2.69%, alpha_vs_sector=-6.52%
- 20d: return=-2.25%, alpha_vs_sector=-6.45%

### VSH
- scan_date: 2026-05-19
- sector: Information Technology
- selected_rank: 1
- 2d: return=15.12%, alpha_vs_sector=12.97%
- 5d: return=33.50%, alpha_vs_sector=28.01%
- 10d: return=74.35%, alpha_vs_sector=62.11%
- 20d: return=64.43%, alpha_vs_sector=58.15%

### SNDK
- scan_date: 2026-05-19
- sector: Information Technology
- selected_rank: 2
- 2d: return=11.70%, alpha_vs_sector=9.55%
- 5d: return=15.16%, alpha_vs_sector=9.67%
- 10d: return=32.66%, alpha_vs_sector=20.42%
- 20d: return=41.88%, alpha_vs_sector=35.60%
