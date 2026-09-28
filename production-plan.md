# Production Plan — Sign2Speak

## 0. Goal

One production sign-language recognizer: **video-first multi-model comparison** to pick a winner, then production-hardening, then optional single-user (webcam) personalization. No `videos/` gloss ships until it passes offline + live gates.

## 1. Scope decisions

- **Primary data: video 15-gloss set** (`processed/metadata.csv`, `selected_videos/` lineage). Webcam `mp_data/`/`test_data/` is single-signer → biased → excluded from comparison, reused only for later personalization.
- **Framework: PyTorch only** for the comparison matrix (TF `cnn_lstm_default` stays as legacy reference, not a contender).
- **Depths:** per-class + confusion, latency + size, live smoke test. (Threshold sweep deferred — derivable later from saved prediction CSVs.)

## 2. Pre-condition: unify inputs (blocking)

- Problem: video 1629-d (no pose visibility) vs webcam 1662-d; `evaluate_model._normalize_for_model()` silently truncates/pads.
- Fix: new `ml/training/datasets/video_keypoints.py` — single loader (`.pt` → float32 → 30-frame pad/truncate → fixed column order, 1629-d). All comparison models consume identical tensors. Document layout once.
- Related debt: no `keypoint_service.py`; `media_service` (holistic 1662-d) vs `MediaProcessor` (pose+hands, face=zeros) diverge. Unify into one service on the 1629-d video layout for the comparison; keep webcam layout only in the personalization stage.

## 3. Model matrix (all PyTorch, same splits/seeds)

| # | Model | File | Role |
|---|---|---|---|
| 1 | BiLSTM baseline | `model_lstm.py` (exists) | reference |
| 2 | GRU | new `model_gru.py` | cheap cell-swap, often beats LSTM on small keypoint data |
| 3 | CNN+LSTM (torch port) | new `model_cnn_lstm_torch.py` | hybrid, made comparable |
| 4 | Transformer | `model_transformer.py` (retrain as `transformer_v2` on unified loader; old `.pth` unevaluated) | attention baseline |
| 5 | **TCN (added)** | new `model_tcn.py` (dilated causal Conv1D) | suggested best accuracy/latency trade-off for keypoint sequences |
| 6–7 | Ablations (pick 2) | config-only | e.g. Transformer layers 3→2; LSTM hidden 256→128 |

Future (not this round): ST-GCN, quantization/ONNX.

## 4. Training discipline

- Identical stratified splits from `processed/metadata.csv`, split-before-augment (`ml/preprocessing/augment.py`, train-only), fixed seeds, same early-stop on `val_acc`.
- Each run: `outputs/training/<family>/<run_id>/` via `ml/training/common.py` + registry entry (`lstm_baseline_v1`, `gru_v1`, `cnn_lstm_torch_v1`, `transformer_v2`, `tcn_v1`; `framework:pytorch`, `input_features:1629`, same 15 labels).
- Note: `processed/datasets/cnn_lstm/` was never built; `train_cnn_lstm --dataset-mode auto` falls back to legacy. Build it or train directly from `processed/` — do not mix lineages.

## 5. Comparison protocol

- **Offline:** `python -m ml.evaluation.compare_models --all-models --split test` → leaderboard (accuracy, F1-macro) + per-class P/R/F1 + confusion matrices + `predictions.csv` (already emitted; add per-class table into `comparison_summary.json`).
- **Latency + size (new columns):** `predict_ms_p50/p95` (CPU, batch=1, post-warmup timed loop), `params`, `artifact_MB` in leaderboard CSV.
- **Live smoke (new `scripts/live_smoke.py`):** replay fixed keypoint clips per gloss through `POST /inference/predict` (`generate_audio:false`); record `{gloss, confidence, accepted@0.7, latency_ms}` → `live_smoke_<model>.json`. Same scripted input for every model.
- **Gates:** overall test acc ≥ 0.80 AND worst retained gloss recall ≥ 0.70 AND p95 predict ≤ budget (suggest 150 ms CPU) AND live smoke accept-rate reported.

## 6. Production-readiness checklist

**Architecture:**
- [ ] Create `keypoint_service.py`; delete `MediaProcessor` fork; assert `input_features` vs registry on load.
- [ ] Wire or delete dead code: `capture_service`, `media_service`, `model_repository`, `audio_repository` (all implemented, unwired).
- [ ] Async TTS: move `generate_bilingual_audio` + translation off `POST /predict` (queue/worker + `(text,lang,voice)` cache); endpoint returns prediction immediately, audio via `GET /inference/audio/<file>` or job id.
- [ ] Reject option: keep `422` quality gate + `prediction_threshold` (0.7 default); add per-gloss calibration + `unknown` handling.
- [ ] Validation/security: payload size cap (already `MAX_CONTENT_LENGTH`), keypoint shape validation (exists), add rate limiting; no user-controlled filenames.

**Reliability/observability:**
- [ ] Structured logs already exist; add per-prediction `{model_name, version, confidence, latency_ms}` + TTS/translation failure counters.
- [ ] Metrics: inference p50/p95, TTS latency, translation fallback rate, `422` rate.
- [ ] Health: `/health`/`/ready` registry-aware (done); add `/ready` gate on active-model load success.

**Testing (currently 0 — `tests/` are empty `.gitkeep`):**
- [ ] Unit: `normalize_keypoints`, loader shapes, registry switch, quality gate.
- [ ] Integration: `POST /predict` golden clips per model; `/media/frame` decode; registry list/switch.
- [ ] E2E/live smoke (§5) as release gate.

**Data/privacy/ops:**
- [ ] Signer-independent eval (leave-one-signer-out or grouped split) — webcam 86% log is optimistic without it.
- [ ] Consent + retention policy for webcam clips; no raw video logging by default.
- [ ] Device target decision (server CPU/GPU vs edge/mobile — rules out TF `.h5` for mobile; PyTorch winner exportable to ONNX/TorchScript later).
- [ ] Containerize (API image; optional worker image), Gunicorn/Uvicorn workers, reverse proxy; pin TF/torch/mediapipe versions.
- [ ] Rollout: registry `active_model` switch + rollback; persist predictions/feedback for active-learning retraining.

## 7. Phased execution

1. **Unify loader + keypoint service** (exit: identical tensor shapes/counts across models).
2. **Add GRU/CNN-LSTM-torch/TCN + trainers** (exit: all importable, one dry-run batch each).
3. **Train matrix on video-15** (exit: 5+ registry entries with manifests).
4. **Offline compare + latency/size** (exit: leaderboard + per-class tables).
5. **Live smoke** (exit: per-model live tables).
6. **Harden** (§6 arch + tests + async TTS) (exit: gates green, TTS non-blocking).
7. **Ship** (exit: `active_model` set, rollback tested) → **then** optional webcam fine-tune.

## 8. Key commands

```bash
python -m ml.preprocessing.build_dataset --augment-copies 1 --sequence-length 30
python -m ml.training.train_tcn --model-name tcn_v1 [--set-active]
python -m ml.evaluation.evaluate_model --model-name tcn_v1 --split test
python -m ml.evaluation.compare_models --all-models --split test
python scripts/live_smoke.py --model tcn_v1
```

## 9. Risks

- `go` (6 train samples) and weak glosses (`good`, `nice`) cap overall score; judge on per-class table, not just top-line accuracy.
- Old transformer artifact untrusted — `transformer_v2` retrain required.
- Single-signer webcam data stays out of comparison by design; final personalization needs its own held-out test.
