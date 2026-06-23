from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json

import pandas as pd

from src.settings import load_feature_config
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import regime_etf_for_sector
from src.utils.shortlist_runtime import load_live_shortlist_model_context
from src.utils.signal_engine import build_analysis_frame, latest_rsi_2_with_intraday, overlay_price_history
from src.utils.strategy import (
    load_active_strategies,
    profit_target_price,
    resolve_trade_strategy,
    rsi_2_exit_triggered,
    summarize_buy_setup,
    trailing_stop_price,
)


@dataclass(frozen=True)
class QuoteReport:
    ticker: str
    current_price: float | None
    current_price_source: str
    last_close: float | None
    close_change_pct: float | None
    latest_scan_date: str | None
    latest_scan_selected: bool | None
    latest_scan_selected_rank: int | None
    latest_scan_selection_source: str | None
    model_name: str | None
    model_rank: int | None
    model_predicted_alpha: float | None
    model_reason_summary: str | None
    model_comparison_summary: str | None
    has_open_trade: bool
    resolved_slot: str | None
    strategy_source: str | None
    entry_price: float | None
    shares: int | None
    unrealized_pct: float | None
    stop_price: float | None
    target_price: float | None
    days_in_trade: int | None
    regime_etf: str | None
    regime_green: bool | None
    buy_setup_status: str | None
    buy_setup_note: str | None
    sell_now: bool
    exit_flags: tuple[str, ...]
    notes: tuple[str, ...]

    def render_console(self) -> str:
        lines = [f"Ticker: {self.ticker}"]
        price_text = self._fmt_price(self.current_price)
        if self.current_price is not None:
            lines.append(f"Current price: {price_text} ({self.current_price_source})")
        else:
            lines.append("Current price: unavailable")
        lines.append(f"Last close: {self._fmt_price(self.last_close)}")
        if self.close_change_pct is not None:
            lines.append(f"Vs last close: {self._fmt_pct(self.close_change_pct)}")

        if self.latest_scan_date is not None:
            selected_label = (
                "selected"
                if self.latest_scan_selected is True
                else ("excluded" if self.latest_scan_selected is False else "unknown")
            )
            scan_bits = [self.latest_scan_date, selected_label]
            if self.latest_scan_selected_rank is not None:
                scan_bits.append(f"rank #{self.latest_scan_selected_rank}")
            if self.latest_scan_selection_source:
                scan_bits.append(self.latest_scan_selection_source)
            lines.append(f"Latest scan: {' | '.join(scan_bits)}")

        if self.model_name is not None:
            model_bits = [self.model_name]
            if self.model_rank is not None:
                model_bits.append(f"model rank #{self.model_rank}")
            if self.model_predicted_alpha is not None:
                model_bits.append(
                    f"predicted 20d alpha vs sector {self._fmt_pct(self.model_predicted_alpha)}"
                )
            lines.append(f"Model: {' | '.join(model_bits)}")
            if self.model_reason_summary:
                lines.append(f"Why: {self.model_reason_summary}")
            if self.model_comparison_summary:
                lines.append(f"Won over: {self.model_comparison_summary}")

        if self.has_open_trade:
            lines.append("")
            lines.append("Open trade:")
            lines.append(
                "  "
                + " | ".join(
                    [
                        f"slot={self.resolved_slot or '-'}",
                        f"source={self.strategy_source or '-'}",
                        f"entry={self._fmt_price(self.entry_price)}",
                        f"shares={self.shares if self.shares is not None else '-'}",
                        f"pnl={self._fmt_pct(self.unrealized_pct)}",
                    ]
                )
            )
            lines.append(
                "  "
                + " | ".join(
                    [
                        f"stop={self._fmt_price(self.stop_price)}",
                        f"target={self._fmt_price(self.target_price)}",
                        f"days={self.days_in_trade if self.days_in_trade is not None else '-'}",
                        f"regime={self._fmt_regime()}",
                    ]
                )
            )
            lines.append(f"  Sell now: {'yes' if self.sell_now else 'no'}")
            lines.append(
                "  Exit flags: "
                + (", ".join(self.exit_flags) if self.exit_flags else "none")
            )
            if self.buy_setup_status is not None:
                lines.append(
                    f"  Strategy setup: {self.buy_setup_status}"
                    + (f" | {self.buy_setup_note}" if self.buy_setup_note else "")
                )
            if self.notes:
                lines.append(f"  Notes: {', '.join(self.notes)}")

        return "\n".join(lines)

    def _fmt_price(self, value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "-"

    def _fmt_pct(self, value: float | None) -> str:
        return f"{value * 100.0:+.2f}%" if value is not None else "-"

    def _fmt_regime(self) -> str:
        if self.regime_etf is None:
            return "-"
        if self.regime_green is None:
            return f"{self.regime_etf} ?"
        return f"{self.regime_etf} {'green' if self.regime_green else 'red'}"


class QuoteService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.logger = get_logger("quote")

    def run(self, *, ticker: str) -> QuoteReport:
        self.db_manager.initialize()
        normalized_ticker = str(ticker).upper().strip()
        open_trade = self.db_manager.get_latest_open_trade(normalized_ticker)
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        sector_map = {str(row["ticker"]).upper(): row["sector"] for row in universe_rows}
        strategies = load_active_strategies()

        trade_sector = (
            sector_map.get(normalized_ticker)
            or (open_trade["strategy_slot"] if open_trade is not None and hasattr(open_trade, "keys") else None)
        )
        regime_etf = regime_etf_for_sector(trade_sector) if trade_sector else "SPY"
        history_tickers = sorted({normalized_ticker, "SPY", "QQQ", regime_etf})
        intraday_prices = self._load_intraday_last_prices(history_tickers)
        base_history = self.db_manager.load_price_history(history_tickers)
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = (
            earnings_loader([normalized_ticker]) if callable(earnings_loader) else pd.DataFrame()
        )
        recent_history = self._download_recent_daily_history(history_tickers)
        price_history = overlay_price_history(base_history, recent_history)
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        latest_snapshot = (
            analysis_frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
            if not analysis_frame.empty
            else pd.DataFrame()
        )
        latest_rows_by_ticker = (
            latest_snapshot.set_index("ticker").to_dict(orient="index")
            if not latest_snapshot.empty
            else {}
        )
        latest_ticker_row = latest_rows_by_ticker.get(normalized_ticker, {})
        latest_scan = self._load_latest_scan_candidate(normalized_ticker)
        current_price, current_price_source, price_note = self._resolve_current_price(
            ticker=normalized_ticker,
            intraday_prices=intraday_prices,
            latest_ticker_row=latest_ticker_row,
            latest_scan=latest_scan,
            open_trade=open_trade,
        )
        last_close = (
            float(latest_ticker_row["adj_close"])
            if pd.notna(latest_ticker_row.get("adj_close"))
            else None
        )
        close_change_pct = None
        if current_price is not None and last_close not in (None, 0):
            close_change_pct = (current_price / float(last_close)) - 1.0

        shortlist_model_context = self._load_shortlist_model_context()
        live_prediction = self._load_live_prediction(
            ticker=normalized_ticker,
            shortlist_model_context=shortlist_model_context,
        )

        model_name = shortlist_model_context.champion_model if shortlist_model_context else None
        model_rank = self._optional_int(live_prediction.get("model_rank"))
        model_predicted_alpha = self._optional_float(live_prediction.get("predicted_alpha"))
        model_reason_summary = self._optional_text(live_prediction.get("model_reason_summary"))
        model_comparison_summary = self._optional_text(live_prediction.get("model_comparison_summary"))

        if model_predicted_alpha is None and latest_scan:
            ranking_components = latest_scan.get("ranking_components", {})
            model_predicted_alpha = self._optional_float(
                ranking_components.get("model_predicted_alpha")
            )
            model_reason_summary = model_reason_summary or self._optional_text(
                ranking_components.get("model_reason_summary")
            )
            model_comparison_summary = model_comparison_summary or self._optional_text(
                ranking_components.get("model_comparison_summary")
            )
            model_name = model_name or self._optional_text(ranking_components.get("model_name"))

        if open_trade is None:
            return QuoteReport(
                ticker=normalized_ticker,
                current_price=current_price,
                current_price_source=current_price_source,
                last_close=last_close,
                close_change_pct=close_change_pct,
                latest_scan_date=self._optional_text(latest_scan.get("scan_date")) if latest_scan else None,
                latest_scan_selected=bool(latest_scan.get("selected")) if latest_scan is not None else None,
                latest_scan_selected_rank=self._optional_int(latest_scan.get("selected_rank")) if latest_scan else None,
                latest_scan_selection_source=self._optional_text(latest_scan.get("selection_source")) if latest_scan else None,
                model_name=model_name,
                model_rank=model_rank,
                model_predicted_alpha=model_predicted_alpha,
                model_reason_summary=model_reason_summary,
                model_comparison_summary=model_comparison_summary,
                has_open_trade=False,
                resolved_slot=None,
                strategy_source=None,
                entry_price=None,
                shares=None,
                unrealized_pct=None,
                stop_price=None,
                target_price=None,
                days_in_trade=None,
                regime_etf=None,
                regime_green=None,
                buy_setup_status=None,
                buy_setup_note=None,
                sell_now=False,
                exit_flags=(),
                notes=(price_note,) if price_note else (),
            )

        resolution = resolve_trade_strategy(
            trade=open_trade,
            strategies=strategies,
            sector_map=sector_map,
            backtest_lookup=getattr(self.db_manager, "get_backtest_result_by_strategy_id", None),
        )
        strategy = resolution.strategy
        entry_price = float(open_trade["entry_price"])
        shares = int(open_trade["shares"])
        unrealized_pct = (
            ((current_price / entry_price) - 1.0)
            if current_price is not None
            else None
        )
        days_in_trade = len(pd.bdate_range(open_trade["entry_date"], date.today().isoformat())) - 1
        notes: list[str] = []
        stop_price = None
        target_price = None
        regime_green = None
        buy_setup_status = None
        buy_setup_note = None
        exit_flags_map = {
            "trailing_stop": False,
            "profit_target": False,
            "rsi_2": False,
            "time_limit": False,
            "regime_flip": False,
            "pre_earnings_exit": False,
        }
        if price_note:
            notes.append(price_note)

        if strategy is None:
            notes.append("strategy unresolved")
        else:
            ticker_sector = sector_map.get(normalized_ticker) or strategy.sector
            regime_etf = regime_etf_for_sector(ticker_sector)
            regime_green = self._resolve_regime_state(
                regime_etf=regime_etf,
                latest_rows_by_ticker=latest_rows_by_ticker,
                intraday_prices=intraday_prices,
            )
            entry_atr = self._row_value(open_trade, "entry_atr")
            if (entry_atr is None or pd.isna(entry_atr)) and pd.notna(latest_ticker_row.get("atr_14")):
                entry_atr = float(latest_ticker_row["atr_14"])
            current_max = max(float(open_trade["max_price_seen"]), current_price or float(open_trade["max_price_seen"]))
            try:
                stop_price = trailing_stop_price(
                    max_price_seen=current_max,
                    entry_atr=float(entry_atr) if entry_atr is not None else None,
                    exit_rules=strategy.exit_rules,
                )
                target_price = profit_target_price(
                    entry_price=entry_price,
                    entry_atr=float(entry_atr) if entry_atr is not None else None,
                    exit_rules=strategy.exit_rules,
                )
            except Exception:
                notes.append("exit thresholds unavailable")

            buy_setup_status, buy_setup_note = summarize_buy_setup(
                strategy_indicators=strategy.indicators,
                latest_row=latest_ticker_row,
            )

            if current_price is not None:
                try:
                    rsi_2 = latest_rsi_2_with_intraday(
                        price_history=price_history,
                        ticker=normalized_ticker,
                        current_price=current_price,
                        as_of=date.today(),
                    )
                    exit_flags_map["rsi_2"] = rsi_2_exit_triggered(
                        rsi_2=rsi_2,
                        unrealized_pct=unrealized_pct,
                        days_in_trade=days_in_trade,
                        strategy_slot=strategy.slot if strategy is not None else None,
                    )
                except Exception:
                    notes.append("intraday RSI unavailable")

                exit_flags_map["trailing_stop"] = bool(
                    stop_price is not None and current_price < stop_price
                )
                exit_flags_map["profit_target"] = bool(
                    target_price is not None and current_price >= target_price
                )

            exit_flags_map["time_limit"] = bool(
                days_in_trade > int(strategy.exit_rules.time_limit_days)
            )
            exit_flags_map["regime_flip"] = regime_green is False
            if (
                strategy.exit_rules.exit_before_earnings_days is not None
                and pd.notna(latest_ticker_row.get("days_to_next_earnings"))
            ):
                exit_flags_map["pre_earnings_exit"] = bool(
                    float(latest_ticker_row["days_to_next_earnings"])
                    <= float(strategy.exit_rules.exit_before_earnings_days)
                )

        exit_flags = tuple(name for name, active in exit_flags_map.items() if active)
        return QuoteReport(
            ticker=normalized_ticker,
            current_price=current_price,
            current_price_source=current_price_source,
            last_close=last_close,
            close_change_pct=close_change_pct,
            latest_scan_date=self._optional_text(latest_scan.get("scan_date")) if latest_scan else None,
            latest_scan_selected=bool(latest_scan.get("selected")) if latest_scan is not None else None,
            latest_scan_selected_rank=self._optional_int(latest_scan.get("selected_rank")) if latest_scan else None,
            latest_scan_selection_source=self._optional_text(latest_scan.get("selection_source")) if latest_scan else None,
            model_name=model_name,
            model_rank=model_rank,
            model_predicted_alpha=model_predicted_alpha,
            model_reason_summary=model_reason_summary,
            model_comparison_summary=model_comparison_summary,
            has_open_trade=True,
            resolved_slot=strategy.slot if strategy is not None else None,
            strategy_source=resolution.source,
            entry_price=entry_price,
            shares=shares,
            unrealized_pct=unrealized_pct,
            stop_price=stop_price,
            target_price=target_price,
            days_in_trade=days_in_trade,
            regime_etf=regime_etf,
            regime_green=regime_green,
            buy_setup_status=buy_setup_status,
            buy_setup_note=buy_setup_note,
            sell_now=bool(exit_flags),
            exit_flags=exit_flags,
            notes=tuple(notes),
        )

    def _load_shortlist_model_context(self):
        try:
            config = load_feature_config()
            shortlist_model_config = (
                config.get("scan_policy", {}).get("shortlist_model", {})
                if isinstance(config, dict)
                else {}
            )
            preferred_model_name = shortlist_model_config.get("production_model_name", "xgboost_model")
            eligible_universe_mode = shortlist_model_config.get(
                "production_eligible_universe_mode",
                shortlist_model_config.get("eligible_universe_mode", "passed_only"),
            )
            model_scope = shortlist_model_config.get("production_model_scope", "global")
            xgboost_config = shortlist_model_config.get("production_xgboost_config", "baseline")
            return load_live_shortlist_model_context(
                self.db_manager,
                preferred_model_name=str(preferred_model_name) if preferred_model_name not in (None, "") else None,
                eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
                model_scope=str(model_scope or "global"),
                xgboost_config=str(xgboost_config or "baseline"),
            )
        except Exception as exc:
            self.logger.warning("Unable to load shortlist model context in quote: %s", exc)
            return None

    def _load_live_prediction(self, *, ticker: str, shortlist_model_context) -> dict:
        if not shortlist_model_context or shortlist_model_context.live_predictions.empty:
            return {}
        predictions = shortlist_model_context.live_predictions
        matched = predictions.loc[predictions["ticker"] == ticker]
        if matched.empty:
            return {}
        return matched.iloc[0].to_dict()

    def _load_latest_scan_candidate(self, ticker: str) -> dict | None:
        loader = getattr(self.db_manager, "load_scan_candidates", None)
        if not callable(loader):
            return None
        frame = loader()
        if frame is None or frame.empty or "ticker" not in frame.columns:
            return None
        matched = frame.loc[frame["ticker"].astype(str).str.upper() == ticker].copy()
        if matched.empty:
            return None
        matched["scan_date"] = pd.to_datetime(matched["scan_date"], errors="coerce")
        matched = matched.sort_values(
            ["scan_date", "selected", "selected_rank", "ticker"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        row = matched.iloc[0].to_dict()
        details = self._parse_details(row.get("details_json"))
        ranking_components = details.get("ranking_components", {}) if isinstance(details, dict) else {}
        return {
            "scan_date": row.get("scan_date").date().isoformat() if pd.notna(row.get("scan_date")) else None,
            "adj_close": self._optional_float(row.get("adj_close")),
            "selected": bool(row.get("selected")) if row.get("selected") is not None else None,
            "selected_rank": self._optional_int(row.get("selected_rank")),
            "selection_source": self._optional_text(
                ranking_components.get("selection_source")
            ),
            "ranking_components": ranking_components,
        }

    def _resolve_current_price(
        self,
        *,
        ticker: str,
        intraday_prices: dict[str, float],
        latest_ticker_row: dict,
        latest_scan: dict | None,
        open_trade,
    ) -> tuple[float | None, str, str | None]:
        intraday_price = intraday_prices.get(ticker)
        if intraday_price is not None:
            return float(intraday_price), "intraday", None

        candidates: list[tuple[date, float, str]] = []
        analysis_date = self._coerce_date(latest_ticker_row.get("date"))
        if analysis_date is not None and pd.notna(latest_ticker_row.get("adj_close")):
            candidates.append((analysis_date, float(latest_ticker_row["adj_close"]), "daily_close"))

        scan_date = self._coerce_date(latest_scan.get("scan_date") if latest_scan else None)
        scan_price = latest_scan.get("adj_close") if latest_scan else None
        if scan_date is not None and scan_price is not None and pd.notna(scan_price):
            candidates.append((scan_date, float(scan_price), "scan_close"))

        if not candidates:
            return None, "unavailable", "current price unavailable"

        price_date, price, source = max(candidates, key=lambda item: item[0])
        entry_date = self._coerce_date(open_trade["entry_date"]) if open_trade is not None else None
        if entry_date is not None and price_date < entry_date:
            return (
                None,
                "unavailable",
                f"current price unavailable; freshest close {price_date.isoformat()} predates entry {entry_date.isoformat()}",
            )
        if (date.today() - price_date).days > 7:
            return (
                None,
                "unavailable",
                f"current price unavailable; freshest close is stale ({price_date.isoformat()})",
            )
        return price, f"{source}:{price_date.isoformat()}", None

    def _coerce_date(self, value) -> date | None:
        if value in (None, "") or pd.isna(value):
            return None
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            return None
        return timestamp.date()

    def _download_recent_daily_history(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        start_date = date.today() - timedelta(days=450)
        try:
            for ticker_batch in chunked(tickers, 50):
                raw_batch = self.market_data_client.download_daily_history(ticker_batch, start_date)
                for ticker in ticker_batch:
                    history = extract_ticker_history(raw_batch, ticker)
                    if not history.empty:
                        frames.append(history)
        except Exception as exc:
            self.logger.warning("Falling back to DuckDB-only daily history in quote: %s", exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_intraday_last_prices(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        try:
            raw_frame = self.market_data_client.download_intraday_history(tickers)
        except Exception as exc:
            self.logger.warning("Unable to load intraday prices in quote: %s", exc)
            return prices
        for ticker in tickers:
            history = extract_ticker_history(raw_frame, ticker)
            if history.empty:
                continue
            prices[ticker] = float(history.iloc[-1]["close"])
        return prices

    def _resolve_regime_state(
        self,
        *,
        regime_etf: str,
        latest_rows_by_ticker: dict[str, dict],
        intraday_prices: dict[str, float],
    ) -> bool | None:
        regime_row = latest_rows_by_ticker.get(regime_etf)
        if not regime_row:
            return None
        regime_price = intraday_prices.get(regime_etf)
        if regime_price is None and pd.notna(regime_row.get("adj_close")):
            regime_price = float(regime_row["adj_close"])
        if regime_price is None:
            return None
        sma_column = "qqq_sma_200" if regime_etf == "QQQ" else "spy_sma_200"
        sma_value = regime_row.get(sma_column)
        if pd.isna(sma_value):
            return None
        return regime_price >= float(sma_value)

    def _optional_int(self, value) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(value)

    def _optional_float(self, value) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _optional_text(self, value) -> str | None:
        if value in (None, "") or pd.isna(value):
            return None
        return str(value)

    def _parse_details(self, value) -> dict:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _row_value(self, row, key: str):
        if hasattr(row, "keys"):
            return row[key] if key in row.keys() else None
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return None
