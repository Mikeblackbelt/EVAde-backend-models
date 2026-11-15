"""In-process backends. NullBackend is the default (EVADE_STORAGE=memory): the app's in-memory state is the only copy,
as in the beta, so it keeps no second copy of readings, flags, risk results or blobs. MemoryBackend keeps everything
(capped) and is the reference implementation the tests run against."""
from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import defaultdict, deque

from .base import Storage

KEEP = 10_000            # per-user cap on input, flag and risk rows kept in memory


class MemoryBackend(Storage):
    name = "memory"
    persistent = False

    def __init__(self, store_raw_windows: bool = False):
        super().__init__(store_raw_windows)
        self._lock = threading.Lock()
        self.users: dict[str, dict] = {}
        self.baselines: dict[tuple, dict] = {}
        self.archive: list[dict] = []
        self.state: dict[str, dict] = defaultdict(dict)
        self.inputs: dict[str, deque] = defaultdict(lambda: deque(maxlen=KEEP))
        self.flags: dict[str, deque] = defaultdict(lambda: deque(maxlen=KEEP))
        self.risk: dict[str, deque] = defaultdict(lambda: deque(maxlen=KEEP))
        self.devices: dict[str, dict] = {}
        self.blobs: dict[str, bytes] = {}

    def has_blobs(self) -> bool:
        return True

    def _delete_user(self, user_id):
        with self._lock:
            n = {"users": int(self.users.pop(user_id, None) is not None)}
            keys = [k for k in self.baselines if k[0] == user_id]
            for k in keys:
                del self.baselines[k]
            n["baselines"] = len(keys)
            before = len(self.archive)
            self.archive = [b for b in self.archive if b["user_id"] != user_id]
            n["baseline_archive"] = before - len(self.archive)
            for name, d in (("user_state", self.state), ("inputs", self.inputs), ("flags", self.flags),
                            ("risk_results", self.risk)):
                v = d.pop(user_id, None)
                n[name] = len(v) if v else 0
            devs = [k for k, v in self.devices.items() if v["user_id"] == user_id]
            for k in devs:
                del self.devices[k]
            n["devices"] = len(devs)
            return n

    def _put_user(self, user_id, attrs, if_absent=False):
        with self._lock:
            if if_absent and user_id in self.users:
                return
            self.users[user_id] = copy.deepcopy(attrs)

    def _get_user(self, user_id):
        with self._lock:
            u = self.users.get(user_id)
            return copy.deepcopy(u) if u is not None else None

    def _put_baselines(self, records):
        with self._lock:
            for r in records:
                self.baselines[(r["user_id"], r["modality"], r["metric"])] = copy.deepcopy(r)

    def _get_baselines(self, user_id):
        with self._lock:
            return [copy.deepcopy(r) for k, r in self.baselines.items() if k[0] == user_id]

    def _archive_baselines(self, user_id, modality, records):
        """Archive exactly the given baselines (by baseline_id), or every stored one of the modality if records is
        None: a baseline created after the caller's snapshot is left alone."""
        with self._lock:
            now = time.time()
            done = {b["baseline_id"] for b in self.archive if b["user_id"] == user_id}
            stored = {k: r for k, r in self.baselines.items() if k[0] == user_id and k[1] == modality}
            if records is None:
                records = list(stored.values())
            mem = {r["baseline_id"]: r for r in records if r.get("modality", modality) == modality}
            for k in [k for k, r in stored.items() if r["baseline_id"] in mem]:
                del self.baselines[k]
            for bid, r in mem.items():
                if bid not in done:
                    self.archive.append({**copy.deepcopy(r), "user_id": user_id, "status": "archived",
                                         "archived_at": now})

    def _archived_ids(self, user_id, ids):
        with self._lock:
            return {b["baseline_id"] for b in self.archive if b["user_id"] == user_id and b["baseline_id"] in ids}

    def _put_state(self, user_id, key, value):
        with self._lock:
            self.state[user_id][key] = copy.deepcopy(value)

    def _get_state(self, user_id):
        with self._lock:
            return copy.deepcopy(dict(self.state.get(user_id, {})))

    def _add_inputs(self, user_id, rows):
        with self._lock:
            self.inputs[user_id].extend({"input_id": uuid.uuid4().hex, **copy.deepcopy(r)} for r in rows)

    def _list_inputs(self, user_id):
        with self._lock:
            return list(self.inputs.get(user_id, []))

    def _add_flags(self, user_id, flags):
        with self._lock:
            self.flags[user_id].extend(copy.deepcopy(flags))

    def _list_flags(self, user_id):
        with self._lock:
            return list(self.flags.get(user_id, []))

    def _add_risk(self, user_id, t, r_mdtr, band, summary):
        with self._lock:
            self.risk[user_id].append({"t": t, "r_mdtr": r_mdtr, "band": band, "summary": copy.deepcopy(summary)})

    def _risk_history(self, user_id):
        with self._lock:
            return [(r["t"], r["r_mdtr"]) for r in self.risk.get(user_id, [])]

    def _put_device(self, device_id, rec):
        with self._lock:
            self.devices[device_id] = copy.deepcopy(rec)

    def _update_device(self, device_id, rec):
        with self._lock:
            cur = self.devices.get(device_id)
            if cur is not None and cur["user_id"] == rec["user_id"] \
                    and (cur.get("last_seq") if cur.get("last_seq") is not None else -1) < rec.get("last_seq", -1):
                self.devices[device_id] = copy.deepcopy(rec)

    def _get_device(self, device_id):
        with self._lock:
            d = self.devices.get(device_id)
            return copy.deepcopy(d) if d is not None else None

    def _delete_device(self, device_id, user_id=None):
        with self._lock:
            cur = self.devices.get(device_id)
            if cur is not None and (user_id is None or cur["user_id"] == user_id):
                del self.devices[device_id]

    def _devices_for_user(self, user_id):
        with self._lock:
            return {k: copy.deepcopy(v) for k, v in self.devices.items() if v["user_id"] == user_id}

    def _put_blob(self, key, data, content_type):
        with self._lock:
            self.blobs[key] = bytes(data)

    def _get_blob(self, key):
        with self._lock:
            return self.blobs.get(key)

    def _list_blobs(self, prefix):
        with self._lock:
            return sorted(k for k in self.blobs if k.startswith(prefix))

    def _delete_prefix(self, prefix):
        with self._lock:
            ks = [k for k in self.blobs if k.startswith(prefix)]
            for k in ks:
                del self.blobs[k]
            return len(ks)

    def _delete_keys(self, keys):
        with self._lock:
            for k in keys:
                self.blobs.pop(k, None)


class NullBackend(MemoryBackend):
    """Default backend: users, baselines, state and devices as MemoryBackend (small, bounded by what the app holds);
    input, flag and risk rows and blobs are dropped, because nothing reads them back without persistence."""

    def has_blobs(self) -> bool:
        return False

    def _add_inputs(self, user_id, rows):
        return None

    def _add_flags(self, user_id, flags):
        return None

    def _add_risk(self, user_id, t, r_mdtr, band, summary):
        return None

    def _put_blob(self, key, data, content_type):
        return None

    def _get_blob(self, key):
        return None

    def _list_blobs(self, prefix):
        return []
