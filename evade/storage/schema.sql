-- EVAde D1 schema, migration 1. Idempotent: safe to apply again.
-- Apply with:  python -m evade.storage.migrate     (uses CF_ACCOUNT_ID, CF_API_TOKEN, CF_D1_DATABASE_ID)
--        or:   npx wrangler d1 execute <database-name> --remote --file evade/storage/schema.sql
-- Only pseudonymous user_ids (UUIDs). No names, emails or raw waveforms in any table.

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  attrs TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS baselines (
  user_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  metric TEXT NOT NULL,
  baseline_id TEXT NOT NULL,
  mode TEXT,
  phase TEXT,
  status TEXT NOT NULL,
  vals TEXT NOT NULL DEFAULT '[]',
  center REAL,
  spread REAL,
  created_at REAL,
  updated_at REAL,
  PRIMARY KEY (user_id, modality, metric)
);

CREATE TABLE IF NOT EXISTS baseline_archive (
  baseline_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  metric TEXT NOT NULL,
  mode TEXT,
  phase TEXT,
  vals TEXT NOT NULL DEFAULT '[]',
  center REAL,
  spread REAL,
  created_at REAL,
  archived_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_baseline_archive_user ON baseline_archive (user_id);

CREATE TABLE IF NOT EXISTS user_state (
  user_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS inputs (
  input_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  t_utc TEXT,
  meta TEXT NOT NULL DEFAULT '{}',
  blob_key TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inputs_user ON inputs (user_id, created_at);

CREATE TABLE IF NOT EXISTS flags (
  flag_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  t REAL NOT NULL,
  domain TEXT,
  metric TEXT,
  modality TEXT,
  level TEXT,
  z REAL,
  component_score REAL,
  confidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_flags_user ON flags (user_id, t);

CREATE TABLE IF NOT EXISTS risk_results (
  risk_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  t REAL NOT NULL,
  r_mdtr REAL,
  band TEXT,
  summary TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_risk_user ON risk_results (user_id, t);

CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  rec TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_devices_user ON devices (user_id);

-- sha256 of a deleted user_id: blocks re-creation without keeping the ID itself
CREATE TABLE IF NOT EXISTS tombstones (
  user_hash TEXT PRIMARY KEY,
  deleted_at REAL NOT NULL
);

INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (1, CAST(strftime('%s', 'now') AS INTEGER));
