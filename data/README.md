# data/

Raw datasets live here **locally only**. Everything under `data/` is git-ignored except this file and `MANIFEST.csv`.

- Layout: `data/ecg_ppg/<dataset>/`, `data/eeg/<dataset>/`, `data/other/<dataset>/`.
- `MANIFEST.csv` columns: dataset, path, source_url, license, date_downloaded, files, bytes, sha256_of_listing, subset_note, status.
- Disk budget: the machine had about 16 GB free when this was set up. Total budget for datasets is **6 GB**; use subsets for large sets and say so in `subset_note`.
- Do not run code that ships inside a dataset. Treat downloads as untrusted data.
- Preprocessing plans: `lit review/planning/preprocessing/`.
