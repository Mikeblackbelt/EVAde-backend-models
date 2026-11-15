# EVAde GPU training on CUNY HPCC (arrow, SLURM)

These are the job scripts for [training-plan-v2.md](../../../lit%20review/planning/training-plan-v2.md). The plan has the reasons, the evaluation protocol, the go/no-go criteria and the export contract. This file covers the run order and the commands.

**Status: nothing here has been submitted.**
- The deep training modules the scripts call (`evade.training.deep.*`) are planned in section 7 of the plan and are **not written yet**. Until they exist, every job stops with exit code 4.
- The cluster facts are unknown until `discover.sh` runs, so every GPU script holds `CHANGE_ME_GPU_TYPE`.
  - `sbatch` is *expected* to reject an unknown gres type, but this has not been tested on arrow. An impossible request might instead sit in the queue.
  - The real gate is `grep -n CHANGE_ME *.sbatch`, where only commented lines may remain, followed by `sbatch --test-only <file>`.
- `<hpc-user>` is the cluster account named in the cuny-hpc skill. It is kept out of this public repo; the scripts use `$USER`. `<skills-dir>` is the local Claude skills folder.

## Rules these scripts follow

These rules come from the cuny-hpc and tusz-server skills.

- **All cluster access goes through the broker:** `python <skills-dir>/tusz-server/scripts/srv.py --host arrow run|put|get`. That means:
  - one session;
  - commands batched into few calls;
  - no polling loops;
  - never stopping the broker, because other sessions share it.

  If the broker reports `AUTH_LOCKED`, stop and tell the user.
- **One GPU per job, one node, 8 CPUs.** No job arrays, no multi-node runs. Folds and seeds run in sequence inside one job. At most one of our jobs runs and one waits.
- **Partition:** `partnsf` with `--qos=qosnsf`. The GPU type is pinned to V100 or A100; never K20m. Each job also checks the GPU and exits with code 3 on an unsupported one.
- **Output:**
  - Every job sets `--output` and `--error` under `/scratch/$USER/evade/slurm/`. That folder must exist before `sbatch`.
  - Program output goes to `.log` files in the run folder.
- **Run folders** are `runs/<session>-<commit>`, so resubmitting the same script resumes and skips finished folds. `SIGUSR1` arrives 10 minutes before the wall time, and the code checkpoints and exits with code 99.
- **Scratch is purged** after about 2 weeks and is not backed up. Copy `export/` off the cluster the same week. No touch jobs.
- **Secrets:** none in any file. `ARROW_PW` lives only in the user's environment. R2 keys never go on arrow.

## Layout on arrow

```
/scratch/<hpc-user>/evade/
  venv/      setup_env.sh (torch 2.7.1 + cu126, onnx, onnxruntime); REQUIREMENTS.lock
  code/      git archive of a committed revision, unpacked with tar -xmf; COMMIT holds the hash
  data/      <dataset>/ + VERIFIED marker after the transfer check; *-prep/ written by S2a and S3a
  runs/      <session>-<commit>[-smoke]/ {job_info.txt, run.log, ckpt..., metrics.json, export/<task>/, export/SHA256SUMS}
  slurm/     %x-%j.out / .err
```

The older `~/tusz_v2`, `~/TUSZ-GPT*` and `~/Tuh-Preprocess` folders belong to other work. Leave them alone.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | done | Read `run.log` and `job_info.txt` once. |
| 3 | unsupported or missing GPU | Fix the `--gres` type or `--constraint`. |
| 4 | not ready: missing venv, code, module, data `VERIFIED` marker, prep output, or a split file that changed | The message is in `job_info.txt`. |
| 5 | S3a window counts differ from the beta | Investigate before running S3. |
| 99 | checkpointed near wall time | Resubmit the same script; it resumes. |

## Run order

Each step's go/no-go criterion is in plan section 8. GPU-hours are *estimates* on a V100 basis; after S0, replace them with (smoke seconds per epoch x epochs).

| Step | What | Where | Estimate |
|---|---|---|---|
| 0a | The user sets the password once with `! setx ARROW_PW "<password>"`. Then run `srv.py --host arrow health`, upload and run `discover.sh` (cuny-hpc skill section 4), and record the results in `references/cluster-facts.md`. | login | - |
| 0a | In every `*.sbatch`, replace `CHANGE_ME_GPU_TYPE` with the V100 or A100 gres type. If needed, uncomment the `--account` or `--constraint` line. Then run `grep -n CHANGE_ME *.sbatch` (only commented lines may remain) and `sbatch --test-only <file>` for each file. | login | - |
| 0b | Write and test `evade.training.deep.*` and the split files on the lab server (`~/evade_venv/bin/python -m pytest -q`), then commit. | lab server | - |
| 0b | Upload the code (below), then run `setup_env.sh`. | login | - |
| 0b | Send the hpchelp notice (plan section 4). Pack PTB-XL on the lab server. Download the full CHB-MIT, Siena and Sleep-EDFx SC sets to the lab server and add MANIFEST rows. | lab server | - |
| 0c | `sbatch s0_smoke.sbatch` | 1 GPU | 0.3 GPU-h (`--time` 00:30) |
| 1 | Transfer and verify `ptbxl100`. Put `data/baseline_preds/ecg_eva6_fold10.parquet`, re-scored on the lab server. Then run the smoke, `sbatch --export=ALL,SMOKE=1 --time=00:15:00 --signal=B:USR1@60 s1_ecg_ptbxl.sbatch`, then `sbatch s1_ecg_ptbxl.sbatch`. Every smoke run needs `--signal=B:USR1@60`; otherwise the 600 s warning ends a 15-minute job at minute 5. | 1 GPU | 2.5 GPU-h (05:00) |
| 2a | Transfer and verify `sleep-edfx`, then `sbatch s2a_sleep_prep_cpu.sbatch` (a smoke run first with `SMOKE=1`). | CPU | 8 CPU-h (02:00) |
| 2 | `sbatch s2_sleep_edfx.sbatch` (smoke first) | 1 GPU | 4 GPU-h (08:00) |
| 3a | Transfer and verify `chbmit` and `siena-scalp-eeg`, then `sbatch s3a_seizure_prep_cpu.sbatch`. | CPU | 40 CPU-h (08:00) |
| 3 | `sbatch s3_seizure_chbmit.sbatch` (smoke first) | 1 GPU | 7 GPU-h (14:00) |
| 4 | `sbatch s4_hrv_ppg_optional.sbatch`, only if S1-S3 are done | 1 GPU | 1 GPU-h (03:00) |

**Total:** about 15 GPU-hours and 50 CPU-hours (*estimate*). Submit the next job only after reading the previous one's result.

## Commands

All commands go through the broker. `SRV` means `python <skills-dir>/tusz-server/scripts/srv.py --host arrow`.

**Upload code.** Only a committed revision is uploaded, as one tar; it is unpacked with fresh timestamps.

```
git archive --format=tar -o <scratchpad>/evade_code.tar HEAD
SRV put <scratchpad>/evade_code.tar evade/evade_code.tar
SRV run "cd /scratch/<hpc-user>/evade && rm -rf code && mkdir -p code slurm && tar -xmf evade_code.tar -C code && echo <full commit hash> > code/COMMIT && rm evade_code.tar"
```

**Environment:**

```
SRV run "bash -l /scratch/<hpc-user>/evade/code/evade/training/hpc/setup_env.sh" --timeout 1200
```

**Submit, then check once:**

```
SRV run "cd /scratch/<hpc-user>/evade/code/evade/training/hpc && sbatch --test-only s1_ecg_ptbxl.sbatch && sbatch s1_ecg_ptbxl.sbatch"
SRV run "squeue -u \$USER; sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode"
SRV run "tail -n 40 /scratch/<hpc-user>/evade/runs/<run>/job_info.txt /scratch/<hpc-user>/evade/runs/<run>/run.log"
```

After submitting, report the job ID, partition, GPU type, CPUs, memory, time requested, expected runtime and log paths.

**Data transfer.** The user runs these in tmux on the lab server, because each hop asks for a password. Data goes directly to arrow through chizen: never through this PC, never parked on chizen.

```
rsync -rlp --partial --size-only --bwlimit=20000 --info=progress2 \
  -e "ssh -J <hpc-user>@chizen.csi.cuny.edu" \
  ~/evade_data/eeg/chbmit/ <hpc-user>@arrow.csi.cuny.edu:/scratch/<hpc-user>/evade/data/chbmit/
```

- Use the same pattern for `eeg/siena-scalp-eeg/` to `data/siena-scalp-eeg/`, for `eeg/sleep-edfx/sleep-cassette/` to `data/sleep-edfx/`, and for the packed `ptbxl100/` folder.
- `-rlp` without `-t` gives fresh arrival times.
- Never send the 87k-file PTB-XL tree.

**Verify** (login node, cheap). Compare the file count and bytes with the lab-server listing; for `ptbxl100`, use `sha256sum -c`. Then mark the dataset as verified:

```
SRV run "cd /scratch/<hpc-user>/evade/data/chbmit && find . -type f | wc -l && du -sb . && date -u > VERIFIED"
```

The marker is written once, as a new file. It is not a touch job; no existing file is refreshed.

**Fetch results.** The user runs this on the lab server, the same week as the run, then checks the sums:

```
rsync -rltv -e "ssh -J <hpc-user>@chizen.csi.cuny.edu" \
  <hpc-user>@arrow.csi.cuny.edu:/scratch/<hpc-user>/evade/runs/<run>/export/ ~/evade_models/<run>/
cd ~/evade_models/<run> && sha256sum -c SHA256SUMS
```

From there, the folder goes to the API host's model mount, or to Cloudflare R2 from the machine that holds the R2 keys. Check the sha256 again after each hop. Register each model in `MODEL_REGISTRY.md` (plan section 10) before it is served.

## Files

| File | Purpose |
|---|---|
| `lib.sh` | Shared helpers: stable run folder, job header, GPU guard, readiness checks, a Python runner that forwards `SIGUSR1`, export checksums |
| `setup_env.sh` | Builds or repairs the venv on scratch (idempotent, x86_64) |
| `s0_smoke.sbatch` | Unit tests, plus a synthetic train, evaluate and export run for each task |
| `s1_ecg_ptbxl.sbatch` | ECG PTB-XL superclass, xresnet1d101 (plan 5.1) |
| `s2a_sleep_prep_cpu.sbatch`, `s2_sleep_edfx.sbatch` | Sleep-EDFx prep and baselines (CPU), then the CNN-LSTM (plan 5.2) |
| `s3a_seizure_prep_cpu.sbatch`, `s3_seizure_chbmit.sbatch` | CHB-MIT and Siena prep and baselines (CPU), then the seizure CNN, tracks A, B, C and Final (plan 5.3) |
| `s4_hrv_ppg_optional.sbatch` | Optional MIT-BIH beats and BIDMC PPG (plan 5.4) |

Check syntax locally with `for f in *.sh *.sbatch; do bash -n "$f"; done`. These files must keep LF line endings (`.gitattributes`), because CRLF breaks `sbatch`.
