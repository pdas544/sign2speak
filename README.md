# Sign2Speak — real-time sign language → English + Hindi speech

Isolated ASL recognition from a laptop webcam. Flask app + PyTorch training pipeline.
Best model: **TCN 87.4%** test accuracy (`tcn_uniform_noface_v2`), 1.8 MB, ~1.5 ms CPU.

## Quick start

```bash
PYENV_VERSION=3.10.18
pip install -r requirements/api.txt
python app/main.py          # PORT env, default 8000
# open http://127.0.0.1:8000/
```

Press **Start**, sign, hold still — capture auto-stops on motion-end. Result shows
gloss + confidence + English/Hindi audio players. `/probe` is the diagnostic page
(true-label logging + JSON export for handedness checks).

## What works (as of 07.10.2026)

| Module | Status | Notes |
|---|---|---|
| Flask MVC app (`app/`) | ✅ done | Controllers/services/repositories, registry-driven inference |
| Live capture (`/media/frame`) | ✅ done | MediaPipe **Holistic** (matches training extractor), tiered fallback, skeleton overlay |
| Prediction (`/inference/predict`) | ✅ done | 1662→1629→225 adaptation, quality gate on adapted features, per-hand L/R in logs + response |
| Bilingual audio | ✅ done | EN direct + HI via translation; graceful null-audio offline |
| UI (`/`, `/probe`) | ✅ done | Gloss chips, model switcher (20 models), OOV note, auto-stop, probe logger |
| Training (`ml/training/train_torch.py`) | ✅ done | Unified trainer: 5 arches, aug×N, uniform/last sampling, face-mask, manifests + registry write-back |
| Dataset (`processed/`) | ✅ done | 809 unique clips (516/158/135), 15 glosses, deduped loader |
| Evaluation (`compare_models.py`) | ✅ done | Accuracy, F1, p50/p95 latency, size, per-class reports, registry score write-back |
| Model registry (20 models) | ✅ done | Unique display names (arch + tags + accuracy); active: `tcn_uniform_noface_v3` |
| Docs | ✅ done | `production-plan.md`, `pretrained-plan.md`, `study-plan.md`, `AGENTS.md`, `progress.md` |

## Model leaderboard (test, n=135)

| Model | Acc | F1 | p50 | Size |
|---|---|---|---|---|
| tcn_uniform_noface_v2 | **83.7%** | 0.81 | 1.3 ms | 1.8 MB |
| tcn_uniform_noface_v3 (active) | 83.0% | 0.80 | 1.4 ms | 1.8 MB |
| transformer_uniform_noface_v1 | 68.6% | 0.67 | 1.1 ms | 11.8 MB |
| tcn_uniform_v1 | 65.1% | 0.61 | 1.9 ms | 4.5 MB |

Key levers found: uniform-30 sampling (+23–29), face removal (+14–26), rotation/translation augmentation (+3.7), data expansion 531→809 (+4.4).

## Pending / known gaps

1. **Per-gloss gate fails**: `good` 0.33, `go` 0.40, `hello` 0.60 (overall 87.4% passes). Needs targeted captures or scope trim.
2. **OOV signs** (`brother`, `yes`) can't classify on 15-gloss models — UI scopes vocabulary; TF 17-gloss model one click away.
3. **Mirror-swap hypothesis open** — awaiting `/probe` user data (right-vs-left block pattern).
4. **Async TTS**, rate limiting, per-gloss calibration, `tests/` suite, containerization — all queued in `production-plan.md`.
5. **Pretrained phases** (`pretrained-plan.md`): videos/-100 transfer → masked SSL → external WLASL weights.

## API quick reference

| Method | Route | Notes |
|---|---|---|
| `GET` | `/`, `/probe`, `/routes` | UI, diagnostics, route map |
| `GET` | `/health`, `/ready`, `/live` | Liveness + registry-aware readiness |
| `GET` | `/media/health` | `mediapipe_holistic` / fallback status |
| `POST` | `/media/frame` | `{frame: dataUrl}` → `{visualization, keypoints[1662], boxes}` |
| `GET` | `/inference/labels` | Active model's glosses (15) |
| `POST` | `/inference/predict` | `{keypoints[30][F], generate_audio}` → gloss, confidence, `hands:{l,r}`, `model`, audio |
| `GET` | `/inference/audio/<file>` | Generated audio |
| `GET/POST` | `/inference/models`, `/inference/models/active` | List / hot-switch models |

## Repo layout (abridged)

```text
app/            Flask MVC (controllers, services incl. inference_backends/, views)
ml/             training (arch_factory, train_torch, torch_common), evaluation, preprocessing
models/         TF default .h5 + registry/registry.json (torch .pth live in gitignored outputs/)
processed/      metadata.csv (tracked) + .pt clips (local-only, see .gitignore)
scripts/        annotation sweeps, probe tooling
requirements/   base / api / training / dev splits
```

## Docs index

- `production-plan.md` — plan of record (status-synced) · `pretrained-plan.md` — pretraining phases
- `study-plan.md` — 11-week technique plan · `AGENTS.md` — agent working notes (layouts, gotchas)
- `progress.md` — build log with round-by-round leaderboards · `safe-to-remove.md` — cleanup history
- `readme-new.md` — previous snapshot (kept for history; this file supersedes it)

## Config (env)

`MODEL_NAME` (override active model), `PREDICTION_THRESHOLD` (0.7),
`MAX_SEQ_LENGTH` (30), `TTS_ENABLED`, `TRANSLATION_TARGET_LANGUAGE` (hindi),
`HOST`/`PORT`/`CAMERA_INDEX`, `MAX_UPLOAD_SIZE_MB`.
