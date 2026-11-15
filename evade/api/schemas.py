"""Pydantic request models for the EVAde API (lit review/planning/api-schema.md). Responses are plain dicts in the envelope.

Every field is bounded: strings by length and pattern (or an allow-list), lists by length, numbers by range. user_id
is a lowercase canonical UUID (the form POST /v1/users returns), device_id a short token. The whole body is also
capped at 1 MB by the app.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Modality = Literal["ecg", "ppg", "eeg", "vitals", "sleep", "behavioral", "radiation", "bone"]
Source = Literal["device", "manual", "simulated", "replay"]
MAX_SAMPLES = 60 * 1000 * 32     # 60 s at up to 1000 Hz for up to 32 channels

UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
DEVICE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,63}$"
METRIC_PATTERN = r"^[a-z][a-z0-9_]{0,39}$"
UNIT_PATTERN = r"^[A-Za-z0-9%/._ -]{1,16}$"
CHANNEL_PATTERN = r"^[A-Za-z0-9._+ -]{1,16}$"
TEXT_PATTERN = r"^[A-Za-z0-9._:/+ -]{0,40}$"

UserId = Annotated[str, Field(min_length=36, max_length=36, pattern=UUID_PATTERN,
                              description="pseudonymous user ID as returned by POST /v1/users (lowercase UUID)")]
DeviceId = Annotated[str, Field(min_length=3, max_length=64, pattern=DEVICE_PATTERN)]
Metric = Annotated[str, Field(min_length=1, max_length=40, pattern=METRIC_PATTERN)]
Unit = Annotated[str, Field(min_length=1, max_length=16, pattern=UNIT_PATTERN)]
Channel = Annotated[str, Field(min_length=1, max_length=16, pattern=CHANNEL_PATTERN)]
ShortText = Annotated[str, Field(max_length=40, pattern=TEXT_PATTERN)]
Phase = Literal["pre_mission", "early_flight", "steady_state", "post_event", "demo"]


def _not_future(t: datetime) -> datetime:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    if t > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("future_timestamp")
    return t


class Window(BaseModel):
    modality: Literal["ecg", "eeg", "ppg"]
    t0_utc: datetime
    fs_hz: float = Field(ge=25, le=1000)
    unit: Unit = "mv"
    channels: list[Channel] = Field(min_length=1, max_length=32)
    samples: list[Annotated[list[float], Field(max_length=60_000)]] = Field(
        min_length=1, max_length=32, description="channels x samples")
    source: Source = "device"
    device_id: DeviceId | None = None

    @field_validator("t0_utc")
    @classmethod
    def check_time(cls, v):
        return _not_future(v)

    @model_validator(mode="after")
    def _shape(self):
        if len(self.samples) != len(self.channels):
            raise ValueError("bad_shape: samples must have one row per channel")
        n = {len(r) for r in self.samples}
        if len(n) != 1:
            raise ValueError("bad_shape: channels have different lengths")
        n = n.pop()
        dur = n / self.fs_hz
        if dur < 2.0 or dur > 60.0:
            raise ValueError("bad_shape: window must be 2-60 s long")
        if n * len(self.channels) > MAX_SAMPLES:
            raise ValueError("payload_too_large")
        if any(not math.isfinite(v) for r in self.samples for v in r):
            raise ValueError("nan_in_signal")
        return self


class Reading(BaseModel):
    metric: Metric
    value: float = Field(ge=-1e6, le=1e6)
    unit: Unit
    t_utc: datetime
    quality: float = Field(1.0, ge=0, le=1)
    modality: Modality = "vitals"

    @field_validator("t_utc")
    @classmethod
    def check_time(cls, v):
        return _not_future(v)

    @field_validator("value", mode="before")
    @classmethod
    def _finite(cls, v):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("nan_in_signal")
        return v


class UserCreate(BaseModel):
    consent: bool
    cohort: ShortText | None = Field(None, max_length=32)
    age_band: Literal["18-29", "30-39", "40-49", "50-59", "60+"] | None = None
    sex: Literal["f", "m", "x", "unspecified"] | None = None

    @field_validator("consent")
    @classmethod
    def _consent(cls, v):
        if not v:
            raise ValueError("consent must be true")
        return v


class BaselineSubmit(BaseModel):
    user_id: UserId
    windows: list[Window] = Field(default_factory=list, max_length=60)
    readings: list[Reading] = Field(default_factory=list, max_length=500)
    phase: Phase = "demo"
    mode: Literal["standard", "demo"] = "demo"


class UserRef(BaseModel):
    user_id: UserId


class BaselineReset(UserRef):
    reason: Literal["new_phase", "device_change", "illness", "user_request"] = "user_request"
    phase: Phase = "demo"


class CompareRequest(BaseModel):
    user_id: UserId
    windows: list[Window] = Field(default_factory=list, max_length=20)
    readings: list[Reading] = Field(default_factory=list, max_length=500)


class EcgRequest(BaseModel):
    user_id: UserId
    window: Window


class EegRequest(BaseModel):
    user_id: UserId
    window: Window


class RadiationRequest(BaseModel):
    user_id: UserId
    t_utc: datetime
    cumulative_dose_msv: float = Field(ge=0, le=5000)
    dose_equivalent_rate_msv_per_day: float = Field(ge=0, le=1000)
    rate_ratio_to_7d_median: float = Field(1.0, ge=0, le=1000)
    spe_active: bool = False
    spe_dose_msv: float = Field(0.0, ge=0, le=5000)
    spe_projected_dose_msv_24h: float | None = Field(None, ge=0, le=5000)
    seu_count_last_hour: int = Field(0, ge=0, le=1_000_000)

    @field_validator("t_utc")
    @classmethod
    def check_time(cls, v):
        return _not_future(v)


SPACEWX_MAX_AGE_DAYS = 7      # posted observations must be recent (the model looks back 48 h)
SPACEWX_MAX_SPAN_H = 96       # and span at most 4 days, so time span cannot blow up the hourly grid


class SpaceWeatherObs(BaseModel):
    """One normalized observation (evade/spaceweather/schema.py FIELDS); every value is optional."""
    t_utc: datetime
    proton_10mev_pfu: float | None = Field(None, ge=0, le=1e7)
    proton_50mev_pfu: float | None = Field(None, ge=0, le=1e7)
    proton_100mev_pfu: float | None = Field(None, ge=0, le=1e7)
    xray_long_wm2: float | None = Field(None, ge=0, le=0.1)
    xray_short_wm2: float | None = Field(None, ge=0, le=0.1)
    kp: float | None = Field(None, ge=0, le=9)
    sw_speed_kms: float | None = Field(None, ge=0, le=5000)
    sw_density_cm3: float | None = Field(None, ge=0, le=1000)
    bz_gsm_nt: float | None = Field(None, ge=-500, le=500)
    dose_rate_usv_h: float | None = Field(None, ge=0, le=1e6)
    proton_satellite: int | None = Field(None, ge=1, le=99, description="GOES number of the proton values")

    @field_validator("t_utc")
    @classmethod
    def check_time(cls, v):
        return _recent(v)


def _recent(v: datetime) -> datetime:
    v = _not_future(v)
    if v < datetime.now(timezone.utc) - timedelta(days=SPACEWX_MAX_AGE_DAYS):
        raise ValueError(f"t_utc older than {SPACEWX_MAX_AGE_DAYS} days")
    return v


class CmeIn(BaseModel):
    """One CME (as DONKI gives it): time it was observed, speed and angular half-width."""
    t_utc: datetime
    speed_kms: float = Field(ge=0, le=5000)
    half_angle_deg: float | None = Field(None, ge=0, le=180)
    earth_directed: bool | None = None

    @field_validator("t_utc")
    @classmethod
    def check_time(cls, v):
        return _recent(v)


class SpaceWeatherRequest(BaseModel):
    user_id: str
    provider: Literal["noaa_swpc", "replay"] | None = Field(None, description="server-side provider by name; or send observations")
    observations: list[SpaceWeatherObs] | None = Field(None, max_length=5000)
    location: Literal["deep_space", "lunar_surface", "mars_surface", "leo_iss"] = "deep_space"
    shielding: Literal["eva_suit", "spacecraft", "storm_shelter"] = "spacecraft"
    cumulative_dose_msv: float | None = Field(None, ge=0, le=5000)
    measured_dose_rate_usv_h: float | None = Field(None, ge=0, le=1e6)
    seu_count_last_hour: int = Field(0, ge=0)
    include_donki: bool = Field(True, description="with provider: also fetch NASA DONKI events (CMEs feed the forecast)")
    proton_source: Literal["swpc_operational", "ncei_archive"] = Field(
        "swpc_operational", description="with observations: which product the proton values are (selects the "
                                        "background floor for proton_satellite)")
    cmes: list[CmeIn] | None = Field(None, max_length=200, description="with observations: recent CMEs (optional)")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.provider is None) == (self.observations is None):
            raise ValueError("give exactly one of provider or observations")
        if self.observations is not None:
            if len(self.observations) == 0:
                raise ValueError("observations must not be empty")
            ts = [o.t_utc for o in self.observations]
            if max(ts) - min(ts) > timedelta(hours=SPACEWX_MAX_SPAN_H):
                raise ValueError(f"observations span more than {SPACEWX_MAX_SPAN_H} h")
        return self


class DomainIn(BaseModel):
    s: float = Field(ge=0, le=1)
    q: float = Field(1.0, ge=0, le=1)
    s_validated: float | None = Field(None, ge=0, le=1)


class RiskRequest(BaseModel):
    user_id: UserId
    domain_inputs: dict[Literal["cardiac", "behavioral", "neuro", "radiation", "bone"], DomainIn] | None = None
    overrides: list[Literal["O1", "O2", "O3", "O4", "O5"]] = Field(default_factory=list, max_length=5)
    logistics_floor_days: float | None = Field(None, ge=0, le=400)
    r_7d_ago: float | None = Field(None, gt=0, lt=1)
    cumulative_dose_msv: float | None = Field(None, ge=0, le=5000)
    dose_rate_msv_day: float | None = Field(None, ge=0, le=1000)
    bone_loss_pct: float | None = Field(None, ge=0, le=100)
    bone_slope_pct_day: float | None = Field(None, ge=0, le=5)


class SelfReport(BaseModel):
    user_id: UserId
    t_utc: datetime
    sleep_quality: int = Field(ge=1, le=5)
    stress: int = Field(ge=1, le=5)
    fatigue: int = Field(ge=1, le=5)
    mood: int = Field(ge=1, le=5)

    @field_validator("t_utc")
    @classmethod
    def check_time(cls, v):
        return _not_future(v)


class Instrument(BaseModel):
    user_id: UserId
    instrument: Literal["phq9", "gad7"]
    t_utc: datetime
    answers: list[Annotated[int, Field(ge=0, le=3)]] = Field(min_length=7, max_length=9)

    @model_validator(mode="after")
    def _len(self):
        n = 9 if self.instrument == "phq9" else 7
        if len(self.answers) != n:
            raise ValueError(f"{self.instrument} needs {n} answers, each 0-3")
        return self


class DeviceBind(BaseModel):
    user_id: UserId
    device_id: DeviceId
    kind: Literal["hr_strap", "ecg_patch", "eeg_headband", "ppg_wrist", "phone", "simulated"]
    vendor: ShortText | None = None
    model: ShortText | None = None
    firmware: ShortText | None = None


class IngestBatch(BaseModel):
    user_id: UserId
    device_id: DeviceId
    seq: int = Field(ge=0, le=2**53)
    windows: list[Window] = Field(default_factory=list, max_length=20)
    readings: list[Reading] = Field(default_factory=list, max_length=500)
    battery_pct: float | None = Field(None, ge=0, le=100)
    clock_offset_ms: float | None = Field(None, ge=-86_400_000, le=86_400_000)
