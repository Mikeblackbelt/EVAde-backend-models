"""Every test runs with a valid API key (32+ characters). It is set at import as well as in the fixture, because test
modules import the app at collection time and the app refuses to start without a key."""
import os

import pytest

TEST_API_KEY = "evade-test-key-0123456789abcdef-0123456789"
KEY_HEADER = {"X-EVAde-Key": TEST_API_KEY}

os.environ["EVADE_API_KEY"] = TEST_API_KEY
for _v in ("EVADE_API_TOKEN", "EVADE_REQUIRE_HTTPS", "EVADE_ENABLE_DOCS", "EVADE_STORAGE"):
    os.environ.pop(_v, None)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("EVADE_API_KEY", TEST_API_KEY)
    monkeypatch.delenv("EVADE_REQUIRE_HTTPS", raising=False)
    return TEST_API_KEY
