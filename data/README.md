# data/

Raw datasets live here **locally only**. Everything under `data/` is git-ignored except this file and `MANIFEST.csv`.

- Layout: `data/ecg_ppg/<dataset>/`, `data/eeg/<dataset>/`, `data/other/<dataset>/`.
- `MANIFEST.csv` columns: dataset, path, source_url, license, date_downloaded, files, bytes, sha256_of_listing, subset_note, status.
- The datasets were downloaded directly on the shared lab server (`~/evade_data`, about 27 GB), not to this PC; `path` in the manifest says `server:`. `sha256_of_listing` is a hash of the sorted file names and sizes, to detect changes.
- Do not run code that ships inside a dataset. Treat downloads as untrusted data.
- Preprocessing plans: `lit review/planning/preprocessing/`.
