"""Normalized space-weather observation schema shared by every provider, the model and the API.

A provider returns `Observations`: a list of `Record`s, each a UTC timestamp plus any subset of the fields below.
Feeds have different cadences (5-min protons, 1-min X-rays, hourly or 3-hourly Kp), so one record does not need
every field; missing values are None. Units are fixed here, and every adapter converts into them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

# field name -> (unit, lower bound, upper bound); bounds reject corrupt values, not unusual-but-real ones
FIELDS: dict[str, tuple[str, float, float]] = {
    "proton_10mev_pfu": ("pfu (protons cm-2 s-1 sr-1), integral >=10 MeV", 0.0, 1e7),
    "proton_50mev_pfu": ("pfu, integral >=50 MeV", 0.0, 1e7),
    "proton_100mev_pfu": ("pfu, integral >=100 MeV", 0.0, 1e7),
    "xray_long_wm2": ("W m-2, 0.1-0.8 nm (XRS-B), true irradiance (no 0.7 scaling)", 0.0, 1e-1),
    "xray_short_wm2": ("W m-2, 0.05-0.4 nm (XRS-A)", 0.0, 1e-1),
    "kp": ("planetary K index, 0-9", 0.0, 9.0),
    "sw_speed_kms": ("solar wind bulk speed, km s-1", 0.0, 5000.0),
    "sw_density_cm3": ("solar wind proton density, cm-3", 0.0, 1000.0),
    "bz_gsm_nt": ("IMF Bz in GSM, nT", -500.0, 500.0),
    "dose_rate_usv_h": ("measured onboard dose-equivalent rate, microSv h-1", 0.0, 1e6),
    "proton_satellite": ("GOES satellite number of the proton values (selects the background correction)", 1.0, 99.0),
}


def utc(t: datetime | str) -> datetime:
    """Parse ISO-8601 (with 'Z', a space separator, or no zone) into an aware UTC datetime."""
    if isinstance(t, str):
        s = t.strip().replace("Z", "+00:00")
        if " " in s and "T" not in s:
            s = s.replace(" ", "T", 1)
        t = datetime.fromisoformat(s)
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def clean(name: str, v) -> float | None:
    """Return a finite in-range float, or None (fill values such as -99999 and out-of-range values drop out)."""
    if v is None or name not in FIELDS:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    lo, hi = FIELDS[name][1], FIELDS[name][2]
    if not math.isfinite(x) or x < lo or x > hi:
        return None
    return x


@dataclass
class Record:
    t: datetime
    values: dict[str, float] = field(default_factory=dict)

    @classmethod
    def make(cls, t, **vals) -> "Record":
        return cls(utc(t), {k: x for k, v in vals.items() if (x := clean(k, v)) is not None})


@dataclass
class Observations:
    source: str
    records: list[Record] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def series(self, name: str) -> list[tuple[datetime, float]]:
        """Time-sorted (t, value) pairs for one field; duplicates at the same time keep the last value."""
        d = {r.t: r.values[name] for r in self.records if name in r.values}
        return sorted(d.items())

    def latest(self, name: str) -> tuple[datetime, float] | None:
        s = self.series(name)
        return s[-1] if s else None

    def has(self, name: str) -> bool:
        return any(name in r.values for r in self.records)

    def merged(self, other: "Observations") -> "Observations":
        return Observations(self.source, self.records + other.records, self.fetched_at,
                            self.warnings + other.warnings, {**self.meta, **other.meta})

    def to_json(self) -> dict:
        return {"source": self.source, "fetched_at": self.fetched_at.isoformat(), "warnings": self.warnings,
                "meta": self.meta, "records": [{"t_utc": r.t.isoformat(), **r.values} for r in self.records]}

    @classmethod
    def from_json(cls, d: dict, source: str | None = None) -> "Observations":
        recs = []
        for row in d.get("records", []):
            row = dict(row)
            t = row.pop("t_utc", None) or row.pop("time_tag", None)
            if t is None:
                continue
            try:
                recs.append(Record.make(t, **{k: v for k, v in row.items() if k in FIELDS}))
            except (TypeError, ValueError):
                continue
        return cls(source or str(d.get("source", "file")), recs, warnings=list(d.get("warnings", [])),
                   meta=dict(d.get("meta", {})))
