"""
Development-level tables v1 (R mock §1 and §3)
v1: presale and brokerage developments, with distance columns added
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from reme.core.codes import ID_DEVELOPMENT
from .config import MockConfig
from .geo import add_distance_cols
from .sampling import (
    categorical,
    dates_without_replacement,
    int_between,
    quarter_labels,
    uniform_between,
    zone_labels,
)
from .schemas import DEVELOPMENTS, DEVELOPMENTS_BROKERAGE


def build_presale_developments(cfg: MockConfig, rng: np.random.Generator) -> pd.DataFrame:
    """§1 — presale developments (development == 1, id_type == 0)."""
    c = cfg.presale_devs
    n = c.n
    dev_type = categorical(rng, c.dev_type_probs, n)                       # §1.1.2
    total_units = int_between(rng, *c.total_units, n)                      # §1.1.3
    amenities = int_between(rng, *c.amenities_score, n)
    price_sqm = rng.normal(c.price_sqm_mean, c.price_sqm_sd, n)            # §1.1.4
    lat = uniform_between(rng, *cfg.geo.presale_bbox.lat, n)               # §1.1.5
    lon = uniform_between(rng, *cfg.geo.presale_bbox.lon, n)
    zone = zone_labels(rng, cfg.zone_ids, n)
    sale_start = dates_without_replacement(rng, c.sale_start_from, cfg.current_date, n)  # §1.1.6
    date_price = quarter_labels(rng, cfg.quarter_years, n)                 # §1.1.7
    # §1.2 — floors: 1 for horizontal / mixed, drawn for vertical dev types
    vertical_floors = int_between(rng, *c.vertical_floors, n)
    floors = np.where(np.isin(dev_type, c.vertical_dev_types), vertical_floors, 1)
    # §1.3 — sold units: share of total, capped at total (R's round is half-to-even, as numpy's)
    share = uniform_between(rng, *c.sold_share, n)
    sold_units = np.minimum(np.round(total_units * share), total_units)

    df = pd.DataFrame({
        "dev_id": [f"DEV_{i}" for i in range(1, n + 1)],
        "development": 1,
        "id_type": ID_DEVELOPMENT,
        "dev_type": dev_type,
        "total_units": total_units,
        "amenities_score": amenities,
        "floors": floors,
        "price_sqm_avg": price_sqm,
        "lat": lat,
        "lon": lon,
        "zone_id": zone,
        "sale_start_date": sale_start,
        "date_price": date_price,
        "sold_units": sold_units.astype(np.int64),
    })
    return DEVELOPMENTS.coerce(add_distance_cols(df, cfg.geo))           # §1.4


def build_brokerage_developments(cfg: MockConfig, rng: np.random.Generator) -> pd.DataFrame:
    """§3 — delivered developments that host resale (brokerage) units."""
    c = cfg.brokerage_devs
    n = c.n
    dev_type = categorical(rng, c.dev_type_probs, n)                       # §3.1
    total_units = int_between(rng, *c.total_units, n)                      # §3.2
    amenities = int_between(rng, *c.amenities_score, n)
    floors = int_between(rng, *c.floors, n)
    price_sqm = rng.normal(c.price_sqm_mean, c.price_sqm_sd, n)
    lat = uniform_between(rng, *cfg.geo.brokerage_bbox.lat, n)
    lon = uniform_between(rng, *cfg.geo.brokerage_bbox.lon, n)
    zone = zone_labels(rng, cfg.zone_ids, n)
    start = cfg.current_date - dt.timedelta(days=c.delivery_lookback_days)
    est_delivery = dates_without_replacement(rng, start, cfg.current_date, n)
    date_price = quarter_labels(rng, cfg.quarter_years, n)

    df = pd.DataFrame({
        "dev_id": [f"BDEV_{i}" for i in range(1, n + 1)],
        "development": 1,
        "id_type": ID_DEVELOPMENT,
        "dev_type": dev_type,
        "total_units": total_units,
        "amenities_score": amenities,
        "floors": floors,
        "price_sqm_avg": price_sqm,
        "lat": lat,
        "lon": lon,
        "zone_id": zone,
        "est_delivery": est_delivery,
        "date_price": date_price,
    })
    return DEVELOPMENTS_BROKERAGE.coerce(add_distance_cols(df, cfg.geo))  # §3.3
