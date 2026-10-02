# Production Plan — Sign2Speak (status: Oct 2026)

## 0. Goal

One production sign-language recognizer: **video-first multi-model comparison** to pick a winner, then production-hardening. No `videos/` gloss ships until it passes offline + live gates. Kiosk is a future plan; current focus is webcam-live accuracy >80% + minimalist UI (prediction + EN/HI audio).

## 1. Scope decisions

- **Primary data: video 15-gloss set** (`processed/metadata.csv`). Now **809 unique clips** (train 516/val 158/test 135) after the `videos/` expansion (was 531).
- **Framework: PyTorch** for the comparison matrix. TF `cnn_lstm_default` stays as one-click alternative in the UI (17-gloss webcam vocab).
- **Depths:** per-class + confusion, latency + size, live probe (`/probe` page with true-label logging + JSON export; `scripts/live_smoke.py` superseded by it).

## 2. Inputs — DONE (with lessons)

- Unified loader `ml/training/datasets/video_keypoints.py`: 1629-d `[pose99,lh63,rh63,face1404]`, samplers `{last,uniform}`, `--mask-face` → 225-d, metadata dedupe (127 repeat rows) → 336/109/86 originally.
- `keypoint_service.py` is source of truth: `webcam_to_video_frame()` (exact), `adapt_for_model()` (passthrough/conversion/declared mask, else 400). Live path migrated to Holistic (`MediaService`) with tiered fallback.
- Lesson: eval must mirror training sampling (uniform model scored 49% under last-30 eval vs true 65% — now sampler-aware via registry hyperparams).

## 3. Model matrix — BUILT (registry: 20 models)

TCN / Transformer / CNN-LSTM-torch / LSTM / GRU via shared `arch_factory.py` + `train_torch.py` + `torch_common.py`. ST-GCN scoped in `pretrained-plan.md`, not yet built.

## 4. Training discipline — LIVE

Split-before-augment (`--augment-copies`, train-only, seeded), fixed seeds, early-stop on val acc. Every run: `outputs/training/<family>/<run_id>/` (checkpoint, history.json, plots, reports) + manifest (env provenance) + registry entry (arch/hyperparams/metrics). `train_cnn_lstm --dataset-mode auto` still falls back to legacy when `processed/datasets/cnn_lstm/` is absent.

## 5. Results (test, n=135 unless noted)

| Model | Acc | Key lever |
|---|---|---|
| tcn_uniform_noface_v2 | **83.7%** | uniform + noface + aug×3 (n=86 round: 69.8%) |
| tcn_uniform_noface_v3 | 83.0% | aug×6 — diminishing returns |
| transformer_uniform_noface_v1 | 68.6% | combo |
| tcn_uniform_v1 | 65.1% | uniform sampling (+23–29 pts alone) |

Levers ranked: uniform sampling > face-mask (+14–26) > data expansion (+4.4 head-to-head) > augmentation (neutral alone, composes). RNNs unfit at this scale.
**Active model: `tcn_uniform_noface_v3`** (per request; v2 scores +0.7).
**Gate verdict**: overall ≥0.80 PASS; worst-gloss ≥0.70 FAIL (good 0.33, go 0.40, hello 0.60). OOV signs (`brother`, `yes`) can never classify on 15-gloss models — UI now scopes vocabulary with chips.

## 6. Production-readiness checklist

**Architecture:**
- [x] `keypoint_service.py` + dim guard (`adapt_for_model`, 400 on mismatch); live path on Holistic.
- [x] Quality gate recalibrated on adapted features; per-hand L/R log stats.
- [x] UI scoping (gloss chips, model switcher, OOV note) + `/probe` handedness page; `hands`/`model` in predict response.
- [ ] Async TTS (still synchronous; graceful null-audio offline).
- [ ] Rate limiting; per-gloss calibration + `unknown` handling.
- [ ] Dead code: `capture_service`, `model_repository`, `audio_repository` still unwired.

**Reliability/observability:**
- [x] Per-prediction `{model, feat_dim, confidence, hands L/R}` in logs.
- [ ] Metrics endpoints; TTS/translation failure counters; `/ready` load gate.

**Testing:** `tests/` still empty; Flask test-client smoke + `/probe` manual protocol are the current verification.

**Data/privacy/ops:** webcam personalization set (kiosk-camera or laptop) still to collect; no container yet; versions unpinned (torch 2.14+cpu, TF 2.12, mediapipe 0.10.10 in dev env).

## 7. Phased execution

1. ~~Unify loader + keypoint service~~ ✅
2. ~~Model files + unified trainer~~ ✅ (5 arches)
3. ~~Train matrix~~ ✅ → 4. ~~Offline compare~~ ✅ → 5. Live probe ✅ (page built; **user probe run pending** — decides mirror-swap).
6. **Next:** mirror verdict → `good/go/hello` targeted data or scope trim → auto-stop capture (Start-only UX) → distance normalization (needs retrain, bundle conditionally) → pretrained phases per `pretrained-plan.md`.
7. Ship when per-gloss gate passes; rollback = registry switch (tested).

## 8. Key commands

```bash
python app/main.py   # PYENV_VERSION=3.10.18
python -m ml.training.train_torch --arch tcn --model-name X --augment-copies 3 --sampler uniform --mask-face
python -m ml.evaluation.compare_models --models a b c --split test
python3 scripts/build_video_annotations.py   # videos/ annotation sweep
```

## 9. Risks (updated)

- `good/go/hello` fail worst-gloss gate on expanded data too — representation or label-noise problem, not scarcity alone (`go` 6→11 train, recall flat 0.40).
- `videos/` labels are weaker (multi-sign compilations); quarantine via `filename` column if ablations demand it.
- Mirror-swap hypothesis still open — probe data pending.
- `.gitignore` hides `videos/ mp_data/ test_data/ outputs/ audio/ processed/{keypoints,train,val,test,datasets}/`; `metadata.csv` + annotations stay tracked.
