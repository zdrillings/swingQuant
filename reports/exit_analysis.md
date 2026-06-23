# Exit Analysis

- closed_trade_count: 16
- linked_trade_count: 2
- excluded_non_recommendation_trades: 14
- analyzed_trade_count: 2
- horizons: 5, 10, 15, 20
- note: this report includes only trades that can be linked to a selected SwingQuant scan recommendation on the entry date.
- note: exact exit reason is not stored in the ledger yet; this report compares realized exits to simple counterfactual holds.

## Actual Exit Summary

- mean_actual_return: 0.027688
- median_actual_return: 0.027688
- hit_rate: 1.0000
- mean_holding_days: 3.50
- mean_mfe_actual: 0.055404
- mean_mae_actual: -0.011442
- mean_actual_giveback: 0.027715
- mean_actual_alpha_vs_spy: 0.027106
- mean_actual_alpha_vs_sector: -0.008406

## Horizon Comparison

### 5d
- comparable_trades: 2
- mean_counterfactual_return: 0.096692
- mean_delta_vs_actual: 0.069003
- better_than_actual_rate: 1.0000
- mean_counterfactual_alpha_vs_sector: 0.035863

### 10d
- comparable_trades: 0

### 15d
- comparable_trades: 0

### 20d
- comparable_trades: 0

## Slot Breakdown

### energy
- trade_count: 2
- mean_actual_return: 0.027688
- mean_actual_giveback: 0.027715
- mean_delta_vs_actual_5d: 0.069003

## Biggest Givebacks

### APA
- slot: energy
- sector: Energy
- entry_date: 2026-05-11
- exit_date: 2026-05-15
- actual_return: 0.031567
- mfe_actual: 0.071919
- actual_giveback: 0.040351

### CRGY
- slot: energy
- sector: Energy
- entry_date: 2026-05-12
- exit_date: 2026-05-15
- actual_return: 0.023810
- mfe_actual: 0.038889
- actual_giveback: 0.015079
