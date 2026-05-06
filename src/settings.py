from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os


class SettingsError(RuntimeError):
    """Raised when the runtime configuration cannot be loaded."""


@dataclass(frozen=True)
class AppPaths:
    root_dir: Path
    data_dir: Path
    duckdb_path: Path
    sqlite_path: Path
    reports_dir: Path
    logs_dir: Path
    config_path: Path
    env_path: Path
    production_strategy_path: Path
    production_strategies_path: Path | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    paths: AppPaths
    env: dict[str, str]
    total_capital: float | None
    risk_per_trade: float | None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SettingsError(f"Invalid .env line: {raw_line}")

        key, raw_value = line.split("=", 1)
        values[key.strip()] = raw_value.strip().strip("'").strip('"')
    return values


def _optional_float(values: dict[str, str], key: str) -> float | None:
    value = values.get(key)
    if value is None or value == "":
        return None
    return float(value)


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    root_dir = _project_root()
    paths = AppPaths(
        root_dir=root_dir,
        data_dir=root_dir / "data",
        duckdb_path=root_dir / "data" / "market_data.duckdb",
        sqlite_path=root_dir / "data" / "ledger.sqlite",
        reports_dir=root_dir / "reports",
        logs_dir=root_dir / "logs",
        config_path=root_dir / "config.yaml",
        env_path=root_dir / ".env",
        production_strategy_path=root_dir / "production_strategy.json",
        production_strategies_path=root_dir / "production_strategies.json",
    )
    env_values = _parse_env_file(paths.env_path)
    return RuntimeSettings(
        paths=paths,
        env=env_values,
        total_capital=_optional_float(env_values, "TOTAL_CAPITAL"),
        risk_per_trade=_optional_float(env_values, "RISK_PER_TRADE"),
    )


def load_feature_config(config_path: Path | None = None) -> dict:
    import yaml

    path = config_path or get_settings().paths.config_path
    if not path.exists():
        raise SettingsError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_production_strategy(path: Path | None = None) -> dict:
    strategy_path = path or get_settings().paths.production_strategy_path
    if not strategy_path.exists():
        raise SettingsError(f"Missing production strategy: {strategy_path}")
    return json.loads(strategy_path.read_text(encoding="utf-8"))


def load_production_strategies(path: Path | None = None) -> dict:
    default_path = get_settings().paths.production_strategies_path
    strategies_path = path or default_path
    if strategies_path is None:
        raise SettingsError("Missing production strategies path.")
    if not strategies_path.exists():
        raise SettingsError(f"Missing production strategies: {strategies_path}")
    return json.loads(strategies_path.read_text(encoding="utf-8"))
