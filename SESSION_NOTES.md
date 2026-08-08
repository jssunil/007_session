# Session Notes

Running log of working sessions on this repo. Newest entry first.

---

## 2026-08-08 (later) — GitHub publish, interim README, trainer email

- Repo pushed to GitHub as **public**: https://github.com/jssunil/007_session
  (account jssunil, branch main). .gitignore excludes data binaries (1.2 GB)
  and checkpoints; the trained tokenizer IS committed for exact repro.
- Interim README written and pushed: thesis, measured-only numbers (param
  counts at 32K/131K, data stats, zero-collision audit at pos_dim 32/48/64),
  arithmetic tables labelled as such, portability guide (clone → dataset
  rebuild → tests → run/resume on Ubuntu/AWS; checkpoints are cross-OS
  portable, copy them manually to continue a run elsewhere).
- Ubuntu fallback plan captured in working_memory.md: probe arm A on Windows
  first (TdrDelay=60 now set); migrate to Linux only if it crashes again.
- Gmail draft to session trainer created (draft id r-5323464324718909841,
  recipient left blank — user must fill in address and send). Asks: does the
  discrete-byte-prediction reframe count as solving Problem 5; collision
  ceiling on a binding vocab?; small scale OK?; highlight tied arm?; deadline/
  format/negative-result acceptability.

### Where to pick up next session

1. (If not done) reboot → TdrDelay takes effect.
2. Resume smoke test: run any _smoke config twice (second run must print
   "resuming from ... at step N") — the resume-LOAD path is still unverified.
3. Full arms in order: A, C (go/no-go on comparable-NLL), then B, D, E.
   `python experiments/run.py experiments/configs/arm_a_dense_dense.yaml`
4. Then M8 anchors (64K vocab: prepare_data.py --vocab-size 65536 --tag _64k
   --dtype uint32) and the final README rewrite (M9) from measured results.
5. Commit + push results as they land.

## 2026-08-08 — GPU driver crash diagnosis + crash-resilience fixes

### What happened to the full-arm runs

The first full training launch (arms run sequentially) died to a Windows/NVIDIA
driver failure, not a code bug:

- **~17:28** — `arm_a_dense_dense` (12,000 steps) started on CUDA. Healthy:
  ~226K tokens/sec, ppl 90 → 19, argmax acc 45% by step 4500.
- **17:35:06–17:35:45** — NVIDIA driver TDR crash. Windows Error Reporting:
  fault bucket `LKD_0x141_Tdr` (VIDEO_ENGINE_TIMEOUT) in `nvlddmkm.sys`,
  tagged "Blackwell" (RTX 5070 Ti), plus two black-screen live kernel dumps.
  `python.exe` died at 17:35:11 (BEX64 in ucrtbase.dll) — arm A killed at
  step ~4500–5000.
- **17:35:16** — `arm_c_kron_bytegrid` launched next in sequence, but CUDA was
  gone post-crash, so `device: auto` silently fell back to **CPU**. It wrote
  one log line and went nowhere.
- **~17:36–17:47** — machine black-screened and hard-rebooted
  (Kernel-Power Event 41). That killed arm C too.

Root cause class: TDR — on Windows/WDDM the same GPU drives the display and
compute; Windows resets the driver if display work stalls >2s under 100%
compute load. The failed recovery (black screen + reboot) suggests some driver
instability on top (driver 596.49 at the time).

### Fixes applied this session

1. **`experiments/run.py` — resume from checkpoint.**
   - Checkpoints now save full state: model, optimizer, step, tokens_seen,
     train-batcher RNG state, torch/CUDA RNG states.
   - Saved atomically (`.pt.tmp` + rename) so a crash mid-save can't corrupt.
   - On launch, an existing full-state `results/checkpoints/<name>.pt` is
     resumed from its step; `--fresh` ignores it. Old model-only checkpoints
     are ignored with a notice (can't resume faithfully without optimizer
     state) — so arm A's pre-crash step-4000 checkpoint will NOT be resumed;
     arm A restarts from step 0.
   - JSONL start line logs `"event": "resume"` and `start_step` when resuming;
     final metrics blob records `resumed_from_step`.
2. **`run.py` — fail-loud device check.** `device: auto` + no CUDA now raises
   immediately instead of silently training on CPU (~200x slower). Explicit
   `device: cpu` in a config still works (tests/debugging unaffected).
3. **Registry (verified set, needs the reboot to take effect):**
   `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` →
   `TdrDelay=60`, `TdrDdiDelay=60` (DWORD). Gives kernels 60s instead of 2s
   before Windows kills the driver.

### Verification status

- `run.py` compiles; checkpoint-**save** path exercised once via a 20-step
  throwaway run (artifacts cleaned out of `results/`). The **resume-load**
  path has NOT been executed yet — user deferred testing until after reboot.
  Two-phase test configs still sit in the session scratchpad if needed, or
  just rerun any smoke arm twice with a bumped `max_steps`.

### State of results as of session end

- `results/logs/arm_a_dense_dense.jsonl` — evals through step 4500 (pre-crash,
  partial run; superseded once arm A reruns).
- `results/logs/arm_c_kron_bytegrid.jsonl` — start line only (the CPU launch).
- `results/checkpoints/arm_a_dense_dense.pt` — old model-only format, step
  4000; will be ignored by resume and overwritten on rerun.
- Smoke runs for arms a/c/d/e completed earlier same day (logs + metrics
  present). `sweep_vocab` figure + metrics written 17:32.
- No full arm has completed. None running.

### Next steps after reboot

1. Relaunch arms sequentially:
   `python experiments/run.py experiments/configs/arm_a_dense_dense.yaml`
   then b, c, d, e. If a run dies, rerun the same command — it resumes from
   the last 4,000-step checkpoint.
2. Reduce TDR recurrence risk: close GPU-heavy apps during runs (LM Studio,
   hardware-accelerated browsers, overlays). If it TDRs again despite
   TdrDelay=60 and a quiet desktop, suspect the driver: check for a newer
   (Studio) driver than 596.49 and clean-install.
3. Optionally verify the resume path with a smoke arm before committing hours
   to full arms.
