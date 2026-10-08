# Gated Selection Dry Run - 2026-10-07

- generated_at: 2026-10-08T02:58:11+00:00
- data_access: read-only CSV artifacts; no `./sq` write command; no `data/` mutation
- oos_artifact: reports/shortlist_model_oos_predictions.csv
- live_artifact: reports/shortlist_model_live_predictions.csv
- model: ridge_adaptive
- target_column: alpha_vs_sector_60d
- selection_gate: score_quantile
- selection_gate_enabled: True
- selection_gate_quantile: 0.9500
- selection_gate_lookback_sessions: 126
- oos_rows_loaded: 221167
- oos_dates_loaded: 522
- oos_date_min: 2024-04-12
- oos_date_max: 2026-07-14
- latest_live_date: 2026-10-07
- latest_live_rows: 647
- live_gate_threshold: +0.446391
- gate_qualified_live_count: 388
- runtime_top_n_after_gate: 2
- would_be_champion: n/a
- numeric_verdict: FAIL: gate-enabled ridge_adaptive does not clear tonight's promotion floors

## Gated Acceptance Windows

| window | dates | active_dates | empty_gated_dates | avg_pick_count | hit_rate_excess | beat_universe_rate | mean_target_excess | spearman | top_ticker | top_ticker_date_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ridge_adaptive_full_oos | 180 | 180 | 342 | +1.9333 | -0.0135 | +0.4944 | +0.0297 | -0.0161 | SNDK | +0.2611 |
| ridge_adaptive_last_fold | 18 | 18 | 2 | +1.7778 | -0.2359 | +0.1667 | -0.1278 | +0.1052 | SNDK | +0.7778 |
| ridge_adaptive_trailing_3folds | 58 | 58 | 2 | +1.9138 | -0.0663 | +0.4483 | -0.0174 | -0.0068 | SNDK | +0.5690 |

## Floor Verdict

| window | metric | value | floor | pass |
|---|---|---:|---:|---|
| last_fold | hit_rate_excess | -0.2359 | >= +0.0200 | no |
| last_fold | beat_universe_rate | +0.1667 | >= +0.5000 | no |
| last_fold | mean_target_excess | -0.1278 | >= +0.0000 | no |
| last_fold | spearman | +0.1052 | >= +0.0000 | yes |
| last_fold | top_ticker_date_rate | +0.7778 | <= +0.4000 | no |
| trailing_3folds | hit_rate_excess | -0.0663 | >= +0.0200 | no |
| trailing_3folds | beat_universe_rate | +0.4483 | >= +0.5000 | no |
| trailing_3folds | mean_target_excess | -0.0174 | >= +0.0000 | no |
| trailing_3folds | spearman | -0.0068 | >= +0.0000 | no |
| trailing_3folds | top_ticker_date_rate | +0.5690 | <= +0.4000 | no |
| full_oos | spearman | -0.0161 | >= +0.0000 | no |

## Would-Be Live Picks

- note: 388 latest-live rows clear the rolling score threshold; the runtime model path remains capped at top-2 picks.

| rank_after_gate | ticker | sector | score | threshold |
|---:|---|---|---:|---:|
| 1 | KRG | Real Estate | +0.957689 | +0.446391 |
| 2 | VTR | Real Estate | +0.949382 | +0.446391 |

## Gate-Qualified Live Roster

| rank_after_gate | ticker | sector | score | threshold |
|---:|---|---|---:|---:|
| 1 | KRG | Real Estate | +0.957689 | +0.446391 |
| 2 | VTR | Real Estate | +0.949382 | +0.446391 |
| 3 | GD | Industrials | +0.944745 | +0.446391 |
| 4 | DOC | Real Estate | +0.943972 | +0.446391 |
| 5 | FRT | Real Estate | +0.934699 | +0.446391 |
| 6 | CALY | Consumer Discretionary | +0.925811 | +0.446391 |
| 7 | KHC | Consumer Staples | +0.920788 | +0.446391 |
| 8 | NNN | Real Estate | +0.916924 | +0.446391 |
| 9 | CTRE | Real Estate | +0.910935 | +0.446391 |
| 10 | INVH | Real Estate | +0.910935 | +0.446391 |
| 11 | PNC | Financials | +0.909196 | +0.446391 |
| 12 | VLY | Financials | +0.907264 | +0.446391 |
| 13 | BBT | Financials | +0.904946 | +0.446391 |
| 14 | JBHT | Industrials | +0.904560 | +0.446391 |
| 15 | BNL | Real Estate | +0.903400 | +0.446391 |
| 16 | MHO | Consumer Discretionary | +0.900309 | +0.446391 |
| 17 | RTX | Industrials | +0.896832 | +0.446391 |
| 18 | BMRN | Health Care | +0.889490 | +0.446391 |
| 19 | USB | Financials | +0.887944 | +0.446391 |
| 20 | BAC | Financials | +0.886012 | +0.446391 |
| 21 | AHR | Real Estate | +0.884853 | +0.446391 |
| 22 | BRX | Real Estate | +0.884853 | +0.446391 |
| 23 | KIM | Real Estate | +0.884080 | +0.446391 |
| 24 | PLD | Real Estate | +0.879057 | +0.446391 |
| 25 | PSA | Real Estate | +0.878284 | +0.446391 |
| 26 | OHI | Real Estate | +0.877898 | +0.446391 |
| 27 | FAF | Financials | +0.874034 | +0.446391 |
| 28 | RF | Financials | +0.872488 | +0.446391 |
| 29 | TFC | Financials | +0.872488 | +0.446391 |
| 30 | GS | Financials | +0.871716 | +0.446391 |
| 31 | KEY | Financials | +0.868238 | +0.446391 |
| 32 | FR | Real Estate | +0.866306 | +0.446391 |
| 33 | HXL | Industrials | +0.865533 | +0.446391 |
| 34 | ALL | Financials | +0.865533 | +0.446391 |
| 35 | SPG | Real Estate | +0.864374 | +0.446391 |
| 36 | GE | Industrials | +0.862828 | +0.446391 |
| 37 | OC | Industrials | +0.861283 | +0.446391 |
| 38 | BXP | Real Estate | +0.857805 | +0.446391 |
| 39 | GNTX | Consumer Discretionary | +0.857419 | +0.446391 |
| 40 | EPR | Real Estate | +0.857032 | +0.446391 |
| 41 | TCBI | Financials | +0.853941 | +0.446391 |
| 42 | AMH | Real Estate | +0.853748 | +0.446391 |
| 43 | SBRA | Real Estate | +0.850850 | +0.446391 |
| 44 | REG | Real Estate | +0.846600 | +0.446391 |
| 45 | FIBK | Financials | +0.845054 | +0.446391 |
| 46 | ZION | Financials | +0.843895 | +0.446391 |
| 47 | CLSK | Information Technology | +0.841963 | +0.446391 |
| 48 | ABCB | Financials | +0.840031 | +0.446391 |
| 49 | GL | Financials | +0.839258 | +0.446391 |
| 50 | SNDR | Industrials | +0.839258 | +0.446391 |
| 51 | UDR | Real Estate | +0.837713 | +0.446391 |
| 52 | PRI | Financials | +0.836940 | +0.446391 |
| 53 | CURB | Real Estate | +0.836553 | +0.446391 |
| 54 | SON | Materials | +0.836167 | +0.446391 |
| 55 | EGP | Real Estate | +0.835587 | +0.446391 |
| 56 | KFY | Industrials | +0.834621 | +0.446391 |
| 57 | CFG | Financials | +0.834235 | +0.446391 |
| 58 | FITB | Financials | +0.829985 | +0.446391 |
| 59 | ELS | Real Estate | +0.829598 | +0.446391 |
| 60 | HTO | Utilities | +0.829212 | +0.446391 |
| 61 | UMBF | Financials | +0.827280 | +0.446391 |
| 62 | GATX | Industrials | +0.825348 | +0.446391 |
| 63 | SIGI | Financials | +0.824961 | +0.446391 |
| 64 | KNX | Industrials | +0.823416 | +0.446391 |
| 65 | CNS | Financials | +0.823029 | +0.446391 |
| 66 | COLB | Financials | +0.823029 | +0.446391 |
| 67 | SCHW | Financials | +0.822257 | +0.446391 |
| 68 | SRPT | Health Care | +0.820711 | +0.446391 |
| 69 | PB | Financials | +0.820325 | +0.446391 |
| 70 | SNA | Industrials | +0.820325 | +0.446391 |
| 71 | SW | Materials | +0.819938 | +0.446391 |
| 72 | RLI | Financials | +0.818779 | +0.446391 |
| 73 | SLM | Financials | +0.816847 | +0.446391 |
| 74 | ORI | Financials | +0.816461 | +0.446391 |
| 75 | SBUX | Consumer Discretionary | +0.815301 | +0.446391 |
| 76 | CBSH | Financials | +0.814142 | +0.446391 |
| 77 | ALKS | Health Care | +0.812597 | +0.446391 |
| 78 | MS | Financials | +0.808733 | +0.446391 |
| 79 | BKU | Financials | +0.807573 | +0.446391 |
| 80 | JLL | Real Estate | +0.805641 | +0.446391 |
| 81 | LAD | Consumer Discretionary | +0.804096 | +0.446391 |
| 82 | STAG | Real Estate | +0.804096 | +0.446391 |
| 83 | FHB | Financials | +0.802550 | +0.446391 |
| 84 | BG | Consumer Staples | +0.801777 | +0.446391 |
| 85 | PHM | Consumer Discretionary | +0.801005 | +0.446391 |
| 86 | FULT | Financials | +0.800232 | +0.446391 |
| 87 | SKT | Real Estate | +0.799459 | +0.446391 |
| 88 | FFIN | Financials | +0.794822 | +0.446391 |
| 89 | MARA | Information Technology | +0.794822 | +0.446391 |
| 90 | MTB | Financials | +0.793663 | +0.446391 |
| 91 | CBT | Materials | +0.789413 | +0.446391 |
| 92 | AUB | Financials | +0.787094 | +0.446391 |
| 93 | INCY | Health Care | +0.784776 | +0.446391 |
| 94 | NTRS | Financials | +0.783617 | +0.446391 |
| 95 | TTWO | Communication Services | +0.783230 | +0.446391 |
| 96 | WDC | Information Technology | +0.782844 | +0.446391 |
| 97 | TRNO | Real Estate | +0.782457 | +0.446391 |
| 98 | FNB | Financials | +0.782071 | +0.446391 |
| 99 | GRBK | Consumer Discretionary | +0.781298 | +0.446391 |
| 100 | ROIV | Health Care | +0.780526 | +0.446391 |
| 101 | ALV | Consumer Discretionary | +0.779366 | +0.446391 |
| 102 | EXR | Real Estate | +0.778980 | +0.446391 |
| 103 | EXPE | Consumer Discretionary | +0.776275 | +0.446391 |
| 104 | CUBE | Real Estate | +0.773570 | +0.446391 |
| 105 | TKR | Industrials | +0.771252 | +0.446391 |
| 106 | ESE | Industrials | +0.767002 | +0.446391 |
| 107 | UNF | Industrials | +0.765456 | +0.446391 |
| 108 | GEN | Information Technology | +0.763138 | +0.446391 |
| 109 | KBR | Industrials | +0.761592 | +0.446391 |
| 110 | LAMR | Real Estate | +0.759660 | +0.446391 |
| 111 | RJF | Financials | +0.758114 | +0.446391 |
| 112 | AJG | Financials | +0.756569 | +0.446391 |
| 113 | OMC | Communication Services | +0.756182 | +0.446391 |
| 114 | IBOC | Financials | +0.754637 | +0.446391 |
| 115 | PAYX | Industrials | +0.753478 | +0.446391 |
| 116 | FHN | Financials | +0.752705 | +0.446391 |
| 117 | VNO | Real Estate | +0.749227 | +0.446391 |
| 118 | NEO | Health Care | +0.747682 | +0.446391 |
| 119 | AWK | Utilities | +0.746909 | +0.446391 |
| 120 | ASB | Financials | +0.744590 | +0.446391 |
| 121 | CINF | Financials | +0.744590 | +0.446391 |
| 122 | FFBC | Financials | +0.744204 | +0.446391 |
| 123 | DVA | Health Care | +0.739181 | +0.446391 |
| 124 | REYN | Consumer Staples | +0.737635 | +0.446391 |
| 125 | NJR | Utilities | +0.737249 | +0.446391 |
| 126 | UNFI | Consumer Staples | +0.736090 | +0.446391 |
| 127 | AMP | Financials | +0.734158 | +0.446391 |
| 128 | APO | Financials | +0.734158 | +0.446391 |
| 129 | WFC | Financials | +0.732998 | +0.446391 |
| 130 | WELL | Real Estate | +0.732226 | +0.446391 |
| 131 | WEX | Financials | +0.731453 | +0.446391 |
| 132 | ES | Utilities | +0.730294 | +0.446391 |
| 133 | PNFP | Financials | +0.729907 | +0.446391 |
| 134 | MAC | Real Estate | +0.729521 | +0.446391 |
| 135 | WTFC | Financials | +0.729134 | +0.446391 |
| 136 | FTI | Energy | +0.728362 | +0.446391 |
| 137 | PFG | Financials | +0.728362 | +0.446391 |
| 138 | AXP | Financials | +0.727589 | +0.446391 |
| 139 | CDP | Real Estate | +0.725270 | +0.446391 |
| 140 | JAZZ | Health Care | +0.723725 | +0.446391 |
| 141 | SWX | Utilities | +0.721020 | +0.446391 |
| 142 | WSFS | Financials | +0.715997 | +0.446391 |
| 143 | WAL | Financials | +0.715224 | +0.446391 |
| 144 | ATR | Materials | +0.714838 | +0.446391 |
| 145 | AAL | Industrials | +0.714065 | +0.446391 |
| 146 | JPM | Financials | +0.712519 | +0.446391 |
| 147 | PENN | Consumer Discretionary | +0.711747 | +0.446391 |
| 148 | SYY | Consumer Staples | +0.711360 | +0.446391 |
| 149 | TXRH | Consumer Discretionary | +0.711360 | +0.446391 |
| 150 | AMCR | Materials | +0.710587 | +0.446391 |
| 151 | RCUS | Health Care | +0.710587 | +0.446391 |
| 152 | D | Utilities | +0.709815 | +0.446391 |
| 153 | OII | Energy | +0.709428 | +0.446391 |
| 154 | ADEA | Information Technology | +0.708655 | +0.446391 |
| 155 | AKAM | Information Technology | +0.707883 | +0.446391 |
| 156 | UBER | Industrials | +0.707110 | +0.446391 |
| 157 | KMB | Consumer Staples | +0.707110 | +0.446391 |
| 158 | ARCB | Industrials | +0.705564 | +0.446391 |
| 159 | RNST | Financials | +0.705564 | +0.446391 |
| 160 | KTB | Consumer Discretionary | +0.704405 | +0.446391 |
| 161 | JKHY | Financials | +0.703632 | +0.446391 |
| 162 | KRYS | Health Care | +0.703246 | +0.446391 |
| 163 | DAR | Consumer Staples | +0.703246 | +0.446391 |
| 164 | NBIX | Health Care | +0.702087 | +0.446391 |
| 165 | PRU | Financials | +0.699768 | +0.446391 |
| 166 | BRO | Financials | +0.698995 | +0.446391 |
| 167 | AX | Financials | +0.697063 | +0.446391 |
| 168 | EG | Financials | +0.696677 | +0.446391 |
| 169 | SSB | Financials | +0.696291 | +0.446391 |
| 170 | STT | Financials | +0.694359 | +0.446391 |
| 171 | HOMB | Financials | +0.693972 | +0.446391 |
| 172 | PECO | Real Estate | +0.692427 | +0.446391 |
| 173 | WTRG | Utilities | +0.692427 | +0.446391 |
| 174 | SHC | Health Care | +0.691654 | +0.446391 |
| 175 | ONB | Financials | +0.690495 | +0.446391 |
| 176 | ZBH | Health Care | +0.685085 | +0.446391 |
| 177 | LSTR | Industrials | +0.684699 | +0.446391 |
| 178 | FHI | Financials | +0.684312 | +0.446391 |
| 179 | INDB | Financials | +0.683153 | +0.446391 |
| 180 | DGX | Health Care | +0.682380 | +0.446391 |
| 181 | FBK | Financials | +0.682380 | +0.446391 |
| 182 | CATY | Financials | +0.681221 | +0.446391 |
| 183 | SAIC | Industrials | +0.678903 | +0.446391 |
| 184 | R | Industrials | +0.678516 | +0.446391 |
| 185 | INVX | Energy | +0.678130 | +0.446391 |
| 186 | MTG | Financials | +0.674652 | +0.446391 |
| 187 | CFR | Financials | +0.674266 | +0.446391 |
| 188 | KNSL | Financials | +0.673879 | +0.446391 |
| 189 | CUZ | Real Estate | +0.673879 | +0.446391 |
| 190 | UCB | Financials | +0.673493 | +0.446391 |
| 191 | MUSA | Consumer Discretionary | +0.672720 | +0.446391 |
| 192 | CVBF | Financials | +0.671947 | +0.446391 |
| 193 | VMI | Industrials | +0.671561 | +0.446391 |
| 194 | ANDE | Consumer Staples | +0.670788 | +0.446391 |
| 195 | SBCF | Financials | +0.670402 | +0.446391 |
| 196 | KGS | Energy | +0.669243 | +0.446391 |
| 197 | AYI | Industrials | +0.668083 | +0.446391 |
| 198 | H | Consumer Discretionary | +0.666151 | +0.446391 |
| 199 | L | Financials | +0.664992 | +0.446391 |
| 200 | DTM | Energy | +0.664606 | +0.446391 |
| 201 | SHW | Materials | +0.664606 | +0.446391 |
| 202 | AFL | Financials | +0.663833 | +0.446391 |
| 203 | SSD | Industrials | +0.663447 | +0.446391 |
| 204 | AIZ | Financials | +0.660742 | +0.446391 |
| 205 | ANIP | Health Care | +0.658037 | +0.446391 |
| 206 | HWC | Financials | +0.656878 | +0.446391 |
| 207 | UBSI | Financials | +0.655719 | +0.446391 |
| 208 | BALL | Materials | +0.650309 | +0.446391 |
| 209 | KVUE | Consumer Staples | +0.649150 | +0.446391 |
| 210 | MCO | Financials | +0.647604 | +0.446391 |
| 211 | BKR | Energy | +0.646832 | +0.446391 |
| 212 | ESS | Real Estate | +0.646445 | +0.446391 |
| 213 | CVS | Health Care | +0.645286 | +0.446391 |
| 214 | RYAN | Financials | +0.644900 | +0.446391 |
| 215 | C | Financials | +0.642968 | +0.446391 |
| 216 | CZR | Consumer Discretionary | +0.642195 | +0.446391 |
| 217 | TGT | Consumer Staples | +0.640649 | +0.446391 |
| 218 | CROX | Consumer Discretionary | +0.639876 | +0.446391 |
| 219 | SKY | Consumer Discretionary | +0.639876 | +0.446391 |
| 220 | HIG | Financials | +0.634080 | +0.446391 |
| 221 | DUK | Utilities | +0.633308 | +0.446391 |
| 222 | PEN | Health Care | +0.632148 | +0.446391 |
| 223 | FCFS | Financials | +0.629444 | +0.446391 |
| 224 | AGCO | Industrials | +0.627512 | +0.446391 |
| 225 | MHK | Consumer Discretionary | +0.626739 | +0.446391 |
| 226 | MOG-A | Industrials | +0.626739 | +0.446391 |
| 227 | AEE | Utilities | +0.625966 | +0.446391 |
| 228 | SFNC | Financials | +0.625966 | +0.446391 |
| 229 | LEA | Consumer Discretionary | +0.623261 | +0.446391 |
| 230 | PLMR | Financials | +0.621716 | +0.446391 |
| 231 | FBP | Financials | +0.618624 | +0.446391 |
| 232 | CUBI | Financials | +0.617465 | +0.446391 |
| 233 | J | Industrials | +0.617465 | +0.446391 |
| 234 | EXC | Utilities | +0.616306 | +0.446391 |
| 235 | TRV | Financials | +0.615147 | +0.446391 |
| 236 | RHP | Real Estate | +0.612442 | +0.446391 |
| 237 | CACI | Industrials | +0.611669 | +0.446391 |
| 238 | ACGL | Financials | +0.611283 | +0.446391 |
| 239 | ATO | Utilities | +0.610896 | +0.446391 |
| 240 | SPGI | Financials | +0.610896 | +0.446391 |
| 241 | CSL | Industrials | +0.609351 | +0.446391 |
| 242 | UNP | Industrials | +0.609351 | +0.446391 |
| 243 | KRC | Real Estate | +0.608964 | +0.446391 |
| 244 | TKO | Communication Services | +0.608964 | +0.446391 |
| 245 | HIW | Real Estate | +0.608578 | +0.446391 |
| 246 | NMIH | Financials | +0.608192 | +0.446391 |
| 247 | LNTH | Health Care | +0.608192 | +0.446391 |
| 248 | ESNT | Financials | +0.606646 | +0.446391 |
| 249 | CSX | Industrials | +0.605100 | +0.446391 |
| 250 | COF | Financials | +0.604714 | +0.446391 |
| 251 | CWST | Industrials | +0.604328 | +0.446391 |
| 252 | TGTX | Health Care | +0.604328 | +0.446391 |
| 253 | TOL | Consumer Discretionary | +0.604328 | +0.446391 |
| 254 | SEZL | Financials | +0.602396 | +0.446391 |
| 255 | LNT | Utilities | +0.601623 | +0.446391 |
| 256 | MAN | Industrials | +0.601236 | +0.446391 |
| 257 | ELAN | Health Care | +0.599304 | +0.446391 |
| 258 | ALG | Industrials | +0.597759 | +0.446391 |
| 259 | PBI | Industrials | +0.595827 | +0.446391 |
| 260 | LIVN | Health Care | +0.595440 | +0.446391 |
| 261 | WM | Industrials | +0.591963 | +0.446391 |
| 262 | DTE | Utilities | +0.591577 | +0.446391 |
| 263 | HSIC | Health Care | +0.591577 | +0.446391 |
| 264 | MGY | Energy | +0.591190 | +0.446391 |
| 265 | NEM | Materials | +0.590417 | +0.446391 |
| 266 | CCK | Materials | +0.587326 | +0.446391 |
| 267 | ITW | Industrials | +0.584235 | +0.446391 |
| 268 | REGN | Health Care | +0.584235 | +0.446391 |
| 269 | LTC | Real Estate | +0.584042 | +0.446391 |
| 270 | EW | Health Care | +0.583849 | +0.446391 |
| 271 | AFG | Financials | +0.580371 | +0.446391 |
| 272 | APD | Materials | +0.579212 | +0.446391 |
| 273 | HASI | Financials | +0.577666 | +0.446391 |
| 274 | BCO | Industrials | +0.574189 | +0.446391 |
| 275 | NSC | Industrials | +0.573416 | +0.446391 |
| 276 | ENVA | Financials | +0.573029 | +0.446391 |
| 277 | MSI | Information Technology | +0.572257 | +0.446391 |
| 278 | WEC | Utilities | +0.571870 | +0.446391 |
| 279 | APPF | Information Technology | +0.571870 | +0.446391 |
| 280 | GPN | Financials | +0.571484 | +0.446391 |
| 281 | RPM | Materials | +0.569165 | +0.446391 |
| 282 | RAMP | Information Technology | +0.568393 | +0.446391 |
| 283 | PPG | Materials | +0.566461 | +0.446391 |
| 284 | COLM | Consumer Discretionary | +0.566074 | +0.446391 |
| 285 | WHD | Energy | +0.565301 | +0.446391 |
| 286 | JNJ | Health Care | +0.562597 | +0.446391 |
| 287 | UNH | Health Care | +0.562597 | +0.446391 |
| 288 | TROW | Financials | +0.562210 | +0.446391 |
| 289 | SO | Utilities | +0.559892 | +0.446391 |
| 290 | DORM | Consumer Discretionary | +0.559119 | +0.446391 |
| 291 | PSMT | Consumer Staples | +0.556414 | +0.446391 |
| 292 | ED | Utilities | +0.555641 | +0.446391 |
| 293 | EBAY | Consumer Discretionary | +0.555641 | +0.446391 |
| 294 | VNOM | Energy | +0.554869 | +0.446391 |
| 295 | SOLV | Health Care | +0.553709 | +0.446391 |
| 296 | ETSY | Consumer Discretionary | +0.547527 | +0.446391 |
| 297 | SR | Utilities | +0.544822 | +0.446391 |
| 298 | CHRD | Energy | +0.540958 | +0.446391 |
| 299 | TFX | Health Care | +0.540572 | +0.446391 |
| 300 | BILL | Information Technology | +0.540185 | +0.446391 |
| 301 | COKE | Consumer Staples | +0.538253 | +0.446391 |
| 302 | IPAR | Consumer Staples | +0.538253 | +0.446391 |
| 303 | OKE | Energy | +0.538253 | +0.446391 |
| 304 | RDN | Financials | +0.538253 | +0.446391 |
| 305 | CB | Financials | +0.537481 | +0.446391 |
| 306 | IDA | Utilities | +0.535549 | +0.446391 |
| 307 | OPLN | Industrials | +0.535549 | +0.446391 |
| 308 | WDAY | Information Technology | +0.534776 | +0.446391 |
| 309 | AM | Energy | +0.533230 | +0.446391 |
| 310 | ACIW | Information Technology | +0.532071 | +0.446391 |
| 311 | GWW | Industrials | +0.529753 | +0.446391 |
| 312 | GVA | Industrials | +0.529753 | +0.446391 |
| 313 | FE | Utilities | +0.528594 | +0.446391 |
| 314 | MOH | Health Care | +0.528594 | +0.446391 |
| 315 | BEN | Financials | +0.527048 | +0.446391 |
| 316 | JBL | Information Technology | +0.526662 | +0.446391 |
| 317 | NFG | Utilities | +0.526662 | +0.446391 |
| 318 | LRN | Consumer Discretionary | +0.522411 | +0.446391 |
| 319 | KDP | Consumer Staples | +0.522025 | +0.446391 |
| 320 | TDY | Information Technology | +0.522025 | +0.446391 |
| 321 | UAL | Industrials | +0.520866 | +0.446391 |
| 322 | HAE | Health Care | +0.520479 | +0.446391 |
| 323 | NI | Utilities | +0.520479 | +0.446391 |
| 324 | XEL | Utilities | +0.519706 | +0.446391 |
| 325 | UCTT | Information Technology | +0.518161 | +0.446391 |
| 326 | BRC | Industrials | +0.517774 | +0.446391 |
| 327 | MAS | Industrials | +0.516615 | +0.446391 |
| 328 | ARMK | Consumer Discretionary | +0.516229 | +0.446391 |
| 329 | STX | Information Technology | +0.515070 | +0.446391 |
| 330 | CRC | Energy | +0.513910 | +0.446391 |
| 331 | SKYW | Industrials | +0.511978 | +0.446391 |
| 332 | VRT | Industrials | +0.511978 | +0.446391 |
| 333 | BJRI | Consumer Discretionary | +0.511206 | +0.446391 |
| 334 | GEHC | Health Care | +0.510819 | +0.446391 |
| 335 | HQY | Health Care | +0.510046 | +0.446391 |
| 336 | DXCM | Health Care | +0.508114 | +0.446391 |
| 337 | DXPE | Industrials | +0.508114 | +0.446391 |
| 338 | XPO | Industrials | +0.507342 | +0.446391 |
| 339 | SEIC | Financials | +0.504250 | +0.446391 |
| 340 | ICUI | Health Care | +0.503864 | +0.446391 |
| 341 | ORLY | Consumer Discretionary | +0.501159 | +0.446391 |
| 342 | STE | Health Care | +0.501159 | +0.446391 |
| 343 | CDW | Information Technology | +0.499614 | +0.446391 |
| 344 | CPK | Utilities | +0.494590 | +0.446391 |
| 345 | MDU | Utilities | +0.494204 | +0.446391 |
| 346 | ADP | Industrials | +0.490726 | +0.446391 |
| 347 | MRSH | Financials | +0.490726 | +0.446391 |
| 348 | RBC | Industrials | +0.490340 | +0.446391 |
| 349 | PATH | Information Technology | +0.489567 | +0.446391 |
| 350 | DBX | Information Technology | +0.488022 | +0.446391 |
| 351 | LH | Health Care | +0.487249 | +0.446391 |
| 352 | MATX | Industrials | +0.481453 | +0.446391 |
| 353 | EWBC | Financials | +0.480294 | +0.446391 |
| 354 | KALU | Materials | +0.480294 | +0.446391 |
| 355 | DE | Industrials | +0.477975 | +0.446391 |
| 356 | IBKR | Financials | +0.476816 | +0.446391 |
| 357 | CNP | Utilities | +0.476043 | +0.446391 |
| 358 | DRI | Consumer Discretionary | +0.474884 | +0.446391 |
| 359 | CRM | Information Technology | +0.473725 | +0.446391 |
| 360 | DASH | Consumer Discretionary | +0.473725 | +0.446391 |
| 361 | ETR | Utilities | +0.472566 | +0.446391 |
| 362 | HRB | Consumer Discretionary | +0.471020 | +0.446391 |
| 363 | CNH | Industrials | +0.469474 | +0.446391 |
| 364 | TDW | Energy | +0.469088 | +0.446391 |
| 365 | CAT | Industrials | +0.468702 | +0.446391 |
| 366 | EVRG | Utilities | +0.468702 | +0.446391 |
| 367 | TREX | Industrials | +0.467929 | +0.446391 |
| 368 | CL | Consumer Staples | +0.465997 | +0.446391 |
| 369 | HON | Industrials | +0.464838 | +0.446391 |
| 370 | OGE | Utilities | +0.464838 | +0.446391 |
| 371 | MMSI | Health Care | +0.464065 | +0.446391 |
| 372 | QCOM | Information Technology | +0.462906 | +0.446391 |
| 373 | ACMR | Information Technology | +0.460587 | +0.446391 |
| 374 | DLTR | Consumer Staples | +0.460587 | +0.446391 |
| 375 | CVCO | Consumer Discretionary | +0.459042 | +0.446391 |
| 376 | MSA | Industrials | +0.457496 | +0.446391 |
| 377 | DCI | Industrials | +0.457110 | +0.446391 |
| 378 | GRMN | Consumer Discretionary | +0.456723 | +0.446391 |
| 379 | NEOG | Health Care | +0.456337 | +0.446391 |
| 380 | CART | Consumer Staples | +0.453246 | +0.446391 |
| 381 | TDC | Information Technology | +0.453246 | +0.446391 |
| 382 | AGYS | Information Technology | +0.451700 | +0.446391 |
| 383 | CVX | Energy | +0.448609 | +0.446391 |
| 384 | AEP | Utilities | +0.448223 | +0.446391 |
| 385 | INDV | Health Care | +0.447450 | +0.446391 |
| 386 | LYV | Communication Services | +0.447450 | +0.446391 |
| 387 | MANH | Information Technology | +0.447063 | +0.446391 |
| 388 | ST | Industrials | +0.447063 | +0.446391 |
