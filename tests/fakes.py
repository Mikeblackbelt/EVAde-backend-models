"""Test doubles for Cloudflare: a D1 REST API backed by in-memory SQLite (via httpx.MockTransport) and an S3 client."""
from __future__ import annotations

import io
import json
import sqlite3
import threading

from evade.storage.cloudflare import SCHEMA_PATH

TOKEN = "cf-test-token-SECRET-123"


class FakeD1:
    """Executes the real SQL (schema.sql applied) so syntax and parameter binding are checked too."""

    def __init__(self, token: str = TOKEN):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.token, self.lock = token, threading.Lock()
        self.fail_with: list = []          # queued failures: a status code, (status, headers), or an Exception
        self.requests: list[dict] = []
        self.max_params_seen = 0
        self.calls = 0                     # every HTTP call that reached the fake, failed or not
        self.on_request = None             # optional hook(body: dict), run before a request is executed

    def _run(self, sql: str, params: list) -> list[dict]:
        assert sql.count("?") <= 100 and len(params) <= 100, "D1 allows at most 100 bound parameters"
        assert all(p is None or isinstance(p, str) for p in params), "D1 REST params are strings or null"
        self.max_params_seen = max(self.max_params_seen, len(params))
        cur = self.db.execute(sql, params)
        return [dict(r) for r in cur.fetchall()] if cur.description else []

    def handler(self, request):
        import httpx
        self.calls += 1
        if self.fail_with:
            f = self.fail_with.pop(0)
            if isinstance(f, Exception):
                raise f
            status, headers = f if isinstance(f, tuple) else (f, {})
            return httpx.Response(status, headers=headers,
                                  json={"success": False, "errors": [{"code": status, "message": "injected"}]})
        assert request.headers["authorization"] == f"Bearer {self.token}"
        assert request.url.path.endswith("/query")
        body = json.loads(request.content)
        if self.on_request is not None:
            self.on_request(body)
        self.requests.append(body)
        stmts = body["batch"] if "batch" in body else [body]
        out = []
        with self.lock:
            try:
                for s in stmts:
                    out.append({"results": self._run(s["sql"], s.get("params", [])), "success": True, "meta": {}})
                self.db.commit()
            except sqlite3.Error as e:
                self.db.rollback()
                return httpx.Response(400, json={"success": False, "errors": [{"code": 7500, "message": str(e)}],
                                                 "result": []})
        return httpx.Response(200, json={"success": True, "errors": [], "messages": [], "result": out})

    def transport(self):
        import httpx
        return httpx.MockTransport(self.handler)

    def count(self, table: str, user_id: str) -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", [user_id]).fetchone()[0]


class _NoSuchKey(Exception):
    def __init__(self):
        super().__init__("NoSuchKey")
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    """The subset of the boto3 S3 client that R2Blobs uses; pages of 2 keys to exercise pagination."""

    PAGE = 2

    def __init__(self):
        self.objects: dict[tuple, bytes] = {}
        self.fail_delete = False           # True: delete_objects raises, as during an R2 outage

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.PAGE]
        more = start + self.PAGE < len(keys)
        return {"Contents": [{"Key": k} for k in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + self.PAGE)} if more else {})}

    def delete_objects(self, Bucket, Delete):
        if self.fail_delete:
            raise ConnectionError("injected R2 outage")
        for o in Delete["Objects"]:
            self.objects.pop((Bucket, o["Key"]), None)
        return {"Deleted": Delete["Objects"]}

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for _, k in self.objects if k.startswith(prefix))
