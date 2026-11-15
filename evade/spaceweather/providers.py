"""Space-weather providers: one small interface, a NOAA SWPC reference adapter, a file replay adapter, and a cache.

To plug in another API, subclass `SpaceWeatherProvider`, implement `fetch(hours)` to return `Observations` in the
normalized schema (schema.FIELDS), and register it under a name with `register()`. See README "What your API needs
to return". Clients choose providers by allow-listed name only; they never pass a URL or a file path.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .schema import Observations, Record, utc


class ProviderUnavailable(RuntimeError):
    """The provider could not deliver usable data (network error, timeout, bad payload, or no proton data)."""


class SpaceWeatherProvider:
    name = "base"

    def fetch(self, hours: int = 72) -> Observations:     # pragma: no cover - interface
        raise NotImplementedError


# ---------------- HTTP (allow-listed hosts, timeout, size cap) ----------------
ALLOWED_HOSTS = {"services.swpc.noaa.gov"}
USER_AGENT = "evade-spaceweather/0.1 (research prototype)"


def http_get_json(url: str, timeout: float = 8.0, max_bytes: int = 8_000_000) -> Any:
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS:
        raise ProviderUnavailable(f"host not allowed: {u.hostname}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:     # noqa: S310 (https and host allow-listed above)
            body = r.read(max_bytes + 1)
    except Exception as e:                                           # URLError, timeout, HTTP errors
        raise ProviderUnavailable(f"fetch failed: {type(e).__name__}") from e
    if len(body) > max_bytes:
        raise ProviderUnavailable("response too large")
    try:
        # SWPC sometimes serves a file with trailing NUL padding (seen 2026-11-15 while a feed was being rewritten)
        return json.loads(body.rstrip(b"\x00 \t\r\n"))
    except ValueError as e:
        raise ProviderUnavailable("response is not JSON") from e


def rows_as_dicts(payload: Any) -> list[dict]:
    """SWPC serves both a list of objects and a list of lists with a header row; accept either."""
    if not isinstance(payload, list) or not payload:
        return []
    if isinstance(payload[0], dict):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload[0], list):
        head = [str(h) for h in payload[0]]
        return [dict(zip(head, r)) for r in payload[1:] if isinstance(r, list)]
    return []


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


# ---------------- NOAA SWPC reference provider (public, no key) ----------------
class NoaaSwpcProvider(SpaceWeatherProvider):
    """NOAA SWPC JSON feeds (services.swpc.noaa.gov), primary GOES satellite plus DSCOVR/ACE real-time solar wind.

    Protons are required (no protons -> ProviderUnavailable); X-rays, Kp and solar wind are optional and missing
    feeds become warnings. GOES-R X-ray fluxes are true irradiance, matching the training data after unscaling.
    """
    name = "noaa_swpc"
    BASE = "https://services.swpc.noaa.gov"
    PROTON_ENERGY = {">=10 MeV": "proton_10mev_pfu", ">=50 MeV": "proton_50mev_pfu", ">=100 MeV": "proton_100mev_pfu"}
    XRAY_ENERGY = {"0.1-0.8nm": "xray_long_wm2", "0.05-0.4nm": "xray_short_wm2"}

    def __init__(self, fetch_json: Callable[[str], Any] | None = None, timeout: float = 8.0):
        self.fetch_json = fetch_json or (lambda url: http_get_json(url, timeout=timeout))

    @staticmethod
    def _span(hours: int) -> str:
        return "6-hour" if hours <= 6 else "1-day" if hours <= 24 else "3-day" if hours <= 72 else "7-day"

    def fetch(self, hours: int = 72) -> Observations:
        span = self._span(hours)
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        recs, warn, meta = [], [], {"feeds": {}}

        def get(key: str, path: str):
            try:
                rows = rows_as_dicts(self.fetch_json(f"{self.BASE}{path}"))
                meta["feeds"][key] = len(rows)
                return rows
            except ProviderUnavailable as e:
                meta["feeds"][key] = f"unavailable ({e})"
                return None

        protons = get("protons", f"/json/goes/primary/integral-protons-{span}.json")
        if not protons:
            raise ProviderUnavailable(f"NOAA SWPC proton feed unavailable or empty ({meta['feeds'].get('protons')})")
        sats = set()
        for r in protons:
            name = self.PROTON_ENERGY.get(str(r.get("energy", "")).replace(" ", "").replace("MeV", " MeV"))
            if name and r.get("time_tag"):
                recs.append(Record.make(r["time_tag"], **{name: r.get("flux")}))
                sats.add(r.get("satellite"))
        meta["proton_satellites"] = sorted(str(s) for s in sats if s is not None)

        xr = get("xrays", f"/json/goes/primary/xrays-{span}.json")
        if xr:
            for r in xr:
                name = self.XRAY_ENERGY.get(str(r.get("energy", "")).replace(" ", ""))
                if name and r.get("time_tag"):
                    recs.append(Record.make(r["time_tag"], **{name: r.get("flux")}))
        else:
            warn.append("x-ray feed unavailable: R-scale and forecast X-ray features missing")

        kp = get("kp", "/products/noaa-planetary-k-index.json")
        if kp:
            for r in kp:
                t = _first(r, "time_tag", "TimeTag")
                if t:
                    recs.append(Record.make(t, kp=_first(r, "Kp", "kp", "kp_index", "estimated_kp")))
        else:
            warn.append("Kp feed unavailable")

        # real-time solar wind (DSCOVR/ACE/SWFO, 1-min); several spacecraft per timestamp, keep the active one
        active = lambda r: r.get("active", True) in (True, "true", 1)
        plasma = get("plasma", "/json/rtsw/rtsw_wind_1m.json")
        for r in plasma or []:
            if r.get("time_tag") and active(r):
                recs.append(Record.make(r["time_tag"], sw_speed_kms=_first(r, "proton_speed", "speed"),
                                        sw_density_cm3=_first(r, "proton_density", "density")))
        mag = get("mag", "/json/rtsw/rtsw_mag_1m.json")
        for r in mag or []:
            if r.get("time_tag") and active(r):
                recs.append(Record.make(r["time_tag"], bz_gsm_nt=r.get("bz_gsm")))
        if not plasma or not mag:
            warn.append("solar wind feed unavailable")

        recs = [r for r in recs if r.values and r.t >= since]
        if not any("proton_10mev_pfu" in r.values for r in recs):
            raise ProviderUnavailable("NOAA SWPC returned no usable >=10 MeV proton values")
        return Observations(self.name, recs, warnings=warn, meta=meta)


# ---------------- file replay provider ----------------
class FileProvider(SpaceWeatherProvider):
    """Replays a JSON file in the normalized schema ({"source", "records": [{"t_utc", <fields>...}]}).

    With `shift_to_now`, the last record is moved to the current time so a historical storm replays as live.
    The path comes from server configuration only (EVADE_SPACEWX_REPLAY_FILE), never from a request.
    """
    name = "replay"

    def __init__(self, path: str | Path, shift_to_now: bool = True, max_bytes: int = 20_000_000):
        self.path, self.shift, self.max_bytes = Path(path), shift_to_now, max_bytes

    def fetch(self, hours: int = 72) -> Observations:
        try:
            if self.path.stat().st_size > self.max_bytes:
                raise ProviderUnavailable("replay file too large")
            obs = Observations.from_json(json.loads(self.path.read_text(encoding="utf-8")), source=self.name)
        except (OSError, ValueError) as e:
            raise ProviderUnavailable(f"replay file unreadable: {type(e).__name__}") from e
        if not obs.records:
            raise ProviderUnavailable("replay file has no records")
        if self.shift:
            dt = datetime.now(timezone.utc) - max(r.t for r in obs.records)
            for r in obs.records:
                r.t = r.t + dt
            obs.meta["shifted_by_hours"] = round(dt.total_seconds() / 3600, 2)
        end = max(r.t for r in obs.records)
        obs.records = [r for r in obs.records if r.t >= end - timedelta(hours=hours)]
        return obs


class StaticProvider(SpaceWeatherProvider):
    """Wraps an in-memory Observations object (tests, and raw observations posted to the API)."""

    def __init__(self, obs: Observations, name: str = "static"):
        self.obs, self.name = obs, name

    def fetch(self, hours: int = 72) -> Observations:
        return self.obs


# ---------------- cache: no hammering, clean failure ----------------
class CachedProvider(SpaceWeatherProvider):
    """Caches successes for `ttl_s` and failures for `fail_ttl_s`; one fetch at a time (single flight)."""

    def __init__(self, inner: SpaceWeatherProvider, ttl_s: float = 300.0, fail_ttl_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.inner, self.ttl, self.fail_ttl, self.clock = inner, ttl_s, fail_ttl_s, clock
        self.name = inner.name
        self._lock = threading.Lock()
        self._cache: dict[int, tuple[float, Observations | None, str | None]] = {}

    def fetch(self, hours: int = 72) -> Observations:
        with self._lock:
            hit = self._cache.get(hours)
            now = self.clock()
            if hit is not None:
                t, obs, err = hit
                if obs is not None and now - t < self.ttl:
                    return obs
                if obs is None and now - t < self.fail_ttl:
                    raise ProviderUnavailable(err or "provider unavailable (cached failure)")
            try:
                obs = self.inner.fetch(hours)
            except ProviderUnavailable as e:
                self._cache[hours] = (now, None, str(e))
                raise
            except Exception as e:                      # adapter bug or unexpected payload: fail closed, cached
                self._cache[hours] = (now, None, f"provider error: {type(e).__name__}")
                raise ProviderUnavailable(f"provider error: {type(e).__name__}") from e
            self._cache[hours] = (now, obs, None)
            return obs


# ---------------- registry (server-side allow-list) ----------------
_REGISTRY: dict[str, SpaceWeatherProvider] = {}
_REG_LOCK = threading.Lock()


def register(name: str, provider: SpaceWeatherProvider, cache: bool = True) -> None:
    with _REG_LOCK:
        _REGISTRY[name] = CachedProvider(provider) if cache else provider


def available() -> list[str]:
    _defaults()
    return sorted(_REGISTRY)


def get_provider(name: str) -> SpaceWeatherProvider:
    _defaults()
    p = _REGISTRY.get(name)
    if p is None:
        raise ProviderUnavailable(f"unknown or unconfigured provider: {name}")
    return p


def _defaults() -> None:
    with _REG_LOCK:
        if "noaa_swpc" not in _REGISTRY and os.environ.get("EVADE_SPACEWX_DISABLE_NETWORK") != "1":
            _REGISTRY["noaa_swpc"] = CachedProvider(NoaaSwpcProvider())
        f = os.environ.get("EVADE_SPACEWX_REPLAY_FILE")
        if f and "replay" not in _REGISTRY:
            _REGISTRY["replay"] = CachedProvider(FileProvider(f), ttl_s=30.0)


def reset_registry() -> None:
    with _REG_LOCK:
        _REGISTRY.clear()


__all__ = ["SpaceWeatherProvider", "NoaaSwpcProvider", "FileProvider", "StaticProvider", "CachedProvider",
           "ProviderUnavailable", "register", "get_provider", "available", "reset_registry", "http_get_json", "utc"]
