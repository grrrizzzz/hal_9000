"""Load the site, threshold and tariff YAML files from the config directory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from hal9000 import config

__all__ = ["Site", "Thresholds", "load_site", "load_thresholds", "load_yaml"]


def load_yaml(name: str, config_dir: Path | None = None) -> dict:
    directory = config_dir or config.settings().config_dir
    path = directory / name
    if not path.exists():
        raise config.ConfigError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text()) or {}


@dataclass(frozen=True)
class Site:
    latitude: float
    longitude: float
    altitude_m: float
    timezone: str
    tilt_deg: float
    azimuth_deg: float
    dc_kw: float
    degradation_factor: float
    bootstrap_performance_ratio: float

    @property
    def openmeteo_azimuth(self) -> float:
        """Open-Meteo's convention: 0 = south, -90 = east, 90 = west."""
        return self.azimuth_deg - 180.0

    @property
    def max_kwh_per_5min(self) -> float:
        """Physical ceiling for one 5-minute bucket, with 20% headroom."""
        return self.dc_kw * 1.2 * 5 / 60


@dataclass(frozen=True)
class Thresholds:
    normal_min: float
    low_min: float
    zero_max: float
    min_expected_kwh: float
    net_position_warn_fraction: float


@lru_cache(maxsize=4)
def _site_cached(config_dir: str) -> Site:
    raw = load_yaml("site.yaml", Path(config_dir))["site"]
    return Site(
        latitude=float(raw["latitude"]),
        longitude=float(raw["longitude"]),
        altitude_m=float(raw.get("altitude_m", 0)),
        timezone=raw["timezone"],
        tilt_deg=float(raw["tilt_deg"]),
        azimuth_deg=float(raw["azimuth_deg"]),
        dc_kw=float(raw["dc_kw"]),
        degradation_factor=float(raw.get("degradation_factor", 1.0)),
        bootstrap_performance_ratio=float(raw.get("bootstrap_performance_ratio", 0.85)),
    )


def load_site(config_dir: Path | None = None) -> Site:
    return _site_cached(str(config_dir or config.settings().config_dir))


def load_thresholds(config_dir: Path | None = None) -> Thresholds:
    raw = load_yaml("thresholds.yaml", config_dir)
    bands = raw["bands"]
    return Thresholds(
        normal_min=float(bands["normal_min"]),
        low_min=float(bands["low_min"]),
        zero_max=float(bands["zero_max"]),
        min_expected_kwh=float(raw.get("min_expected_kwh", 1.0)),
        net_position_warn_fraction=float(raw.get("net_position_warn_fraction", 0.1)),
    )
