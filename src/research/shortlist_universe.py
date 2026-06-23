from __future__ import annotations

import pandas as pd


VALID_ELIGIBLE_UNIVERSE_MODES = (
    "passed_only",
    "passed_or_trend",
)

VALID_MODEL_SCOPES = (
    "global",
    "sector_specific",
)


def normalize_eligible_universe_mode(value: str | None) -> str:
    mode = str(value or "passed_only").strip().lower()
    if mode not in VALID_ELIGIBLE_UNIVERSE_MODES:
        raise ValueError(
            f"Unsupported eligible_universe_mode={value!r}. "
            f"Choose one of: {', '.join(VALID_ELIGIBLE_UNIVERSE_MODES)}."
        )
    return mode


def normalize_model_scope(value: str | None) -> str:
    scope = str(value or "global").strip().lower()
    if scope not in VALID_MODEL_SCOPES:
        raise ValueError(
            f"Unsupported model_scope={value!r}. "
            f"Choose one of: {', '.join(VALID_MODEL_SCOPES)}."
        )
    return scope


def eligible_universe_mode_description(mode: str) -> str:
    normalized = normalize_eligible_universe_mode(mode)
    if normalized == "passed_only":
        return "passed_any_strategy plus liquidity and price floor"
    return (
        "passed_any_strategy or trend-qualified liquid names "
        "(green regime, above 200d, positive 63d momentum, RS vs SPY >= 60)"
    )


def filter_eligible_universe(frame: pd.DataFrame, *, eligible_universe_mode: str) -> pd.DataFrame:
    mode = normalize_eligible_universe_mode(eligible_universe_mode)
    working = frame.copy()
    working["md_volume_30d"] = pd.to_numeric(working.get("md_volume_30d"), errors="coerce")
    working["adj_close"] = pd.to_numeric(working.get("adj_close"), errors="coerce")
    passed_any_strategy = (
        working["passed_any_strategy"]
        if "passed_any_strategy" in working.columns
        else pd.Series(False, index=working.index)
    )
    working["passed_any_strategy"] = passed_any_strategy.fillna(False).astype(bool)

    base_mask = working["md_volume_30d"].ge(20_000_000.0) & working["adj_close"].gt(0.0)
    if mode == "passed_only":
        return working.loc[base_mask & working["passed_any_strategy"]].copy()

    regime_green = (
        working["regime_green"]
        if "regime_green" in working.columns
        else pd.Series(False, index=working.index)
    )
    working["regime_green"] = regime_green.fillna(False).astype(bool)
    working["sma_200_dist"] = pd.to_numeric(working.get("sma_200_dist"), errors="coerce")
    working["roc_63"] = pd.to_numeric(working.get("roc_63"), errors="coerce")
    working["relative_strength_index_vs_spy"] = pd.to_numeric(
        working.get("relative_strength_index_vs_spy"),
        errors="coerce",
    )

    trend_mask = (
        working["regime_green"]
        & working["sma_200_dist"].gt(0.0)
        & working["roc_63"].gt(0.0)
        & working["relative_strength_index_vs_spy"].ge(60.0)
    )
    return working.loc[base_mask & (working["passed_any_strategy"] | trend_mask)].copy()
