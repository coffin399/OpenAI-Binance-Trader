from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.default.yaml"
USER_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigSection:
    """Thin wrapper that provides attribute access to nested dicts."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item not in self._data:
            raise AttributeError(f"Config option '{item}' not found")
        value = self._data[item]
        if isinstance(value, dict):
            return ConfigSection(value)
        return value

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._data)


class Config(ConfigSection):
    """Loads configuration from YAML with environment overrides."""

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or USER_CONFIG_PATH
        ensure_user_config_exists(self._config_path)

        default_config = load_yaml_config(DEFAULT_CONFIG_PATH)
        user_config = load_yaml_config(self._config_path)
        data = merge_dicts(default_config, user_config)
        data = apply_environment_overrides(data)
        super().__init__(data)

    @property
    def path(self) -> Path:
        return self._config_path


def ensure_user_config_exists(config_path: Path) -> None:
    if config_path.exists():
        return
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Default configuration not found at {DEFAULT_CONFIG_PATH}."
        )
    config_path.write_bytes(DEFAULT_CONFIG_PATH.read_bytes())


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def merge_dicts(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    keys = set(default) | set(override)
    for key in keys:
        default_value = default.get(key)
        override_value = override.get(key)

        if isinstance(default_value, dict) and isinstance(override_value, dict):
            result[key] = merge_dicts(default_value, override_value)
        elif override_value is not None:
            result[key] = override_value
        else:
            result[key] = default_value
    return result


def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    overrides = {
        ("binance", "api_key"): os.getenv("BINANCE_API_KEY"),
        ("binance", "api_secret"): os.getenv("BINANCE_API_SECRET"),
        ("binance", "testnet"): _env_bool("BINANCE_TESTNET"),
        ("discord", "bot_token"): os.getenv("DISCORD_BOT_TOKEN"),
        ("discord", "channel_ids"): _env_list("DISCORD_CHANNEL_IDS"),
        ("runtime", "dry_run"): _env_bool("TRADING_DRY_RUN"),
        ("trading", "symbols"): _env_list("TRADING_SYMBOLS"),
    }

    for path, value in overrides.items():
        if value in (None, []):
            continue
        section = config
        for key in path[:-1]:
            section = section.setdefault(key, {})
        section[path[-1]] = value

    ai_config = config.get("ai", {})
    providers = ai_config.get("providers", [])
    for provider in providers:
        name = provider.get("name")
        if not name:
            continue
        env_key = f"{name.upper()}_API_KEYS"
        keys = _env_list(env_key)
        if keys:
            provider["api_keys"] = keys

    return config


def _env_bool(env_var: str) -> Optional[bool]:
    value = os.getenv(env_var)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _env_list(env_var: str) -> Optional[list[str]]:
    raw_value = os.getenv(env_var)
    if not raw_value:
        return None
    separators = [",", "\n", ";"]
    values = [raw_value]
    for separator in separators:
        if separator in raw_value:
            values = [piece for piece in raw_value.replace("\r", "").split(separator)]
    cleaned = [value.strip() for value in values if value.strip()]
    return cleaned or None
