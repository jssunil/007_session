1. SESSION_NOTES.md (project root) — the full record: crash timeline with the event-log evidence, the exact run.py changes (resume, atomic checkpoints, fail-loud device check), the registry keys, what's verified vs. not (the resume-load path hasn't run yet), the current state of results/, and the post-reboot steps. It's structured as a running log, so future sessions can prepend new entries.
2. My persistent memory for this project — a compact pointer with the key facts (TDR crash cause, resume support added but load-path unverified, TdrDelay=60 pending reboot), so a future session recalls this automatically even if you don't mention it.

Safe to reboot whenever you're ready. After that: relaunch arm A with python experiments/run.py experiments/configs/arm_a_dense_dense.yaml — and ideally run one smoke arm twice first to confirm resume works.

---

## Ubuntu dual-boot as fallback (assessed 2026-08-08)

The crash was a TDR — a Windows/WDDM-only mechanism (watchdog kills the display
driver when compute starves display work). Linux has no equivalent, so moving
there eliminates that failure class entirely, and WDDM overhead disappears
(typically ~5–15% faster). Code needs zero changes (pure PyTorch, relative
paths); checkpoints are portable, so a partial arm resumes on either OS.

What Linux does NOT fix: if the black-screen + hard reboot came from power/
thermal/VRAM problems under sustained load, that follows the hardware. A Linux
run doubles as a diagnostic — if it also dies there, suspect hardware, not OS.

Migration cost ~30 min if needed:
- NVIDIA driver 570+ (open kernel modules) for Blackwell; verify nvidia-smi.
- venv with torch cu128 (sm_120), tokenizers, numpy, matplotlib, pyyaml.
- Copy project to the ext4 SSD — don't memmap train.bin off NTFS via ntfs3.
- Registry TDR keys irrelevant on Linux; resume + fail-loud changes carry over.

Decision rule: remaining GPU work is only ~2–3 hours (~15 min per arm, AR arm
30–40 min). After reboot, run the resume smoke test, then arm A (~15 min) as a
probe on Windows with TdrDelay=60 and a quiet desktop. If it completes, finish
on Windows. If it crashes again, move to Ubuntu immediately — don't keep
fighting the Windows driver.
