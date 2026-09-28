# Sign2Speak (Current State — Sept 2026)

Real-time sign-language → English + Hindi speech. Flask MVC app + `ml/` training pipeline.
Smoked-tested: `GET /health ok`, `GET /inference/labels 17 labels`,
`POST /inference/predict 200/422`, `GET /media/health ok`.

## Quick Start

```bash
pip install -r requirements/api.txt      # app runtime
# optional: training / dev extras
pip install -r requirements/training.txt
pip install -r requirements/dev.txt

python app/main.py                       # PORT env, default 8000
# open http://127.0.0.1:8000/  (Start/Stop recognition)
```

Health / introspection:

```bash
curl localhost:8000/live
curl localhost:8000/health
curl localhost:8000/media/health
curl localhost:8000/inference/labels
curl localhost:8000/routes
```

Predict (30 frames × 1662 features, audio off for smoke test):

```bash
curl -X POST localhost:8000/inference/predict \
  -H 'Content-Type: application/json' \
  -d '{"keypoints": [], "generate_audio": false}'
# empty → 400; zeros → 422 low-signal; motion → 200 {predicted_gloss, confidence, audio}
```

## Folder Structure

```text
sign2speak_data/
├── app/                                # Flask MVC runtime
│   ├── main.py                         # create_app(), GET / + /routes
│   ├── config/
│   │   ├── settings.py                 # env-driven Settings (MODEL_NAME, THRESHOLD, TTS_*, DIRS)
│   │   └── logging.py
│   ├── controllers/                    # thin HTTP layer
│   │   ├── __init__.py                 # register_controllers()
│   │   ├── health_controller.py        # /health /ready /live (registry-aware)
│   │   ├── inference_controller.py     # /inference/predict|labels|audio|models (+active)
│   │   └── media_controller.py         # /media/frame|health|video_feed (lazy cv2/mediapipe)
│   ├── services/                       # business logic
│   │   ├── inference_service.py        # registry + backend dispatch, quality gate input
│   │   ├── model_registry_service.py   # models/registry/registry.json CRUD + build_backend()
│   │   ├── inference_backends/
│   │   │   ├── base.py                 # InferenceBackend ABC
│   │   │   ├── tf_backend.py           # .h5 via tensorflow (lazy, time_major shim)
│   │   │   └── torch_backend.py        # generic .pth via arch_factory (lazy torch)
│   │   ├── keypoint_service.py         # VIDEO_1629 vs WEBCAM_1662 source of truth + assert_compatible
│   │   ├── tts_service.py              # bilingual mp3 via TTSHelper (lazy, degrades to null)
│   │   ├── translation_service.py      # deep_translator + cache (lazy, falls back to EN)
│   │   ├── media_service.py            # holistic 1662-d extraction (lazy; currently unused by controller)
│   │   └── capture_service.py          # webcam ring-buffer (implemented, unwired)
│   ├── repositories/
│   │   ├── model_repository.py         # torch-only loader (currently unused — registry does loading)
│   │   └── audio_repository.py         # audio file helpers (currently unused — tts_service writes direct)
│   ├── models/
│   │   ├── request_models.py
│   │   └── response_models.py
│   └── views/
│       ├── templates/index.html        # main UI (webcam + prediction + EN/HI audio + history)
│       ├── templates/result.html
│       └── static/{css/shared|index|result.css, js/app.js}
├── ml/                                 # training pipeline
│   ├── preprocessing/
│   │   ├── extract_keypoints.py        # wrapper → root extract-keypoints-full.py
│   │   ├── build_dataset.py            # mp_data → processed arrays + metadata.csv (split-before-augment)
│   │   └── augment.py
│   ├── training/
│   │   ├── train_cnn_lstm.py           # TF hybrid, --dataset-mode auto|processed|legacy
│   │   ├── train_torch.py              # unified PyTorch trainer --arch {lstm,gru,cnn_lstm,tcn,transformer}
│   │   ├── arch_factory.py             # shared model construction (trainer + serving)
│   │   ├── torch_common.py             # shared loop: early-stop, reports, history.json
│   │   ├── train_lstm.py               # wrapper → root train_lstm.py
│   │   ├── train_transformer.py        # wrapper → root model_transformer.py
│   │   ├── common.py                   # run dirs, manifests (+env provenance), registry (arch/hyperparams/metrics)
│   │   └── datasets/video_keypoints.py # unified 1629-d loader (dedupes metadata → 336/109/86)
│   └── evaluation/
│       ├── evaluate_model.py           # per-model eval + latency/size + registry write-back
│       └── compare_models.py           # leaderboard (acc, F1, p50/p95, MB)
├── models/
│   ├── action_model_cnn_lstm_new.h5    # active TF default (webcam lineage)
│   ├── registry/registry.json          # 7 models: TF default + transformer_v1 + 5 torch v2
│   └── (outputs/models/*.pth)          # gitignored serving artifacts (tcn_v1, gru_v1, …)
├── data/{raw,processed,keypoints}/     # canonical dirs (placeholders, .gitkeep)
├── outputs/{models,training/}          # run artifacts, checkpoints, plots, reports
├── audio/  logs/                       # runtime-generated (auto-created)
├── tests/{unit,integration,e2e}/        # EMPTY — .gitkeep only, 0 tests
├── scripts/                            # EMPTY — .gitkeep only
├── requirements/
│   ├── base.txt                        # numpy/opencv/mediapipe/sklearn/pyttsx3/deep-translator
│   ├── api.txt                         # + flask/fastapi/uvicorn
│   ├── training.txt                    # + tensorflow/torch/TTS/soundfile
│   └── dev.txt                         # + pytest/black/ruff
├── src/                                # legacy package (superseded, unimported)
├── mp_data/ processed/ test_data/      # legacy datasets (kept, large)
├── annotations/ videos*/ selected_videos/  # legacy media/annotations (kept, large)
└── *.md                                # README.md (legacy) / readme-new.md (this) / codebase-summary.md /
                                        # progress.md / safe-to-remove.md (cleanup log)
```

## Dataset Summary

Three data lineages exist. Only **(A)** trains the comparison matrix; **(B)** trains
the legacy serving model; **(C)** is an untouched reserve.

### (A) Video keypoints — comparison dataset (WL-ASL lineage, 1629-d)

- **Source**: `selected_videos/` (518 videos) + `videos/` reserve (278 matched, 14 dupes).
  **531 → 809 unique clips**: train **516** / val **158** / test **135**.
  Best model `tcn_uniform_noface_v2`: test **83.7%** / F1 0.81 (overall gate PASS;
  worst-gloss gate FAIL on good/go/hello — see progress.md Round 4).
- **Keypoints**: `processed/keypoints/{train,val,test}/*.pt` via `extract-keypoints-full.py`
  (MediaPipe Holistic, layout `[pose99 xyz, lh63, rh63, face1404]` — pose **present**,
  visibility channel absent).
- **Index**: `processed/metadata.csv` — 695 rows but only **531 unique clips**
  (127 `skipped (already processed)` duplicates). Always load via
  `ml.training.datasets.video_keypoints.read_split()`, never raw csv.
- **Unique-clip split (train/val/test)**:

| Gloss | Train | Val | Test | Videos dl |
|---|---|---|---|---|
| beautiful | 15 | 7 | 5 | 27 |
| big | 11 | 7 | 6 | 23 |
| boy | 23 | 9 | 4 | 36 |
| friend | 29 | 6 | 3 | 35 |
| go | 6 | 4 | 5 | 13 |
| good | 19 | 8 | 5 | 32 |
| happy | 19 | 11 | 8 | 38 |
| hello | 18 | 7 | 16 | 40 |
| like | 28 | 8 | 6 | 42 |
| nice | 30 | 9 | 5 | 43 |
| no | 26 | 9 | 4 | 38 |
| sister | 27 | 5 | 4 | 35 |
| teacher | 31 | 6 | 9 | 43 |
| what | 27 | 5 | 3 | 35 |
| white | 27 | 8 | 3 | 38 |
| **Total** | **336** | **109** | **86** | **518** |

- **Quality** (measured over all 531 clips): pose signal 100% (present everywhere);
  face 99% (near-constant → mostly noise, yet 1404/1629 features); right hand 75%
  (3% of clips near-zero); left hand 49% (**31% of clips <5% signal** — one-handed
  signs + detection misses); mean clip length **87 frames**, currently truncated to
  the most-recent 30 (uniform sampling recommended, see §Known Gaps).
- **Verdict**: thin (`go`=6 train) and imbalanced; explains the 38–48% test accuracies.
  Needs augmentation + more samples per gloss before any ship decision.

### (B) Webcam keypoints — legacy serving dataset (1662-d, single-signer)

- **Source**: laptop webcam via `test_action_recognition.py:capture_video()`.
- `mp_data/` + `test_data/`: **17 glosses × 30 sequences** (balanced, layout
  `[pose132 xyzv, face1404, lh63, rh63]` — pose *with* visibility).
- Trains TF `cnn_lstm_default` (offline acc 0.86). Single-signer → optimistic;
  excluded from the comparison; reserved for post-decision personalization.

### (C) Reserve (unused in any training)

- `videos/`: **100 gloss folders, 1356 videos** (6–44 per gloss, mean 13.6) — never used.
- `videos_top_10_gloss/`: 10 folders, **empty**. `videos_all/`: 385 flat mp4s.

## Root Files (all load-bearing)

| File | Used by |
|---|---|
| `action_recognition_model.py` | `ml/training/train_cnn_lstm.py` (TF hybrid) |
| `test_action_recognition.py` | webcam collection + `train_cnn_lstm` legacy loader |
| `model_transformer.py` | generic `TorchBackend`, `train_torch --arch transformer` |
| `model_lstm.py` / `model_gru.py` | `train_torch --arch lstm\|gru` via `arch_factory` |
| `model_cnn_lstm_torch.py` / `model_tcn.py` | `train_torch --arch cnn_lstm\|tcn` via `arch_factory` |
| `tts_helper.py` | `tts_service`, `inference_service` (both lazy) |
| `train_lstm.py` → `dataset_lstm.py`, `model_lstm.py` | `ml/training/train_lstm.py` chain |
| `extract-keypoints-full.py` | `ml/preprocessing/extract_keypoints.py` |

Removed 24 legacy root files — see `safe-to-remove.md` (recoverable via git).

## API Reference

| Method | Route | Notes |
|---|---|---|
| `GET` | `/` | UI page |
| `GET` | `/routes` | route map |
| `GET` | `/health`, `/ready`, `/live` | registry-aware model check |
| `GET` | `/media/health` | `ok` (mediapipe) or `degraded` (fallback zeros) |
| `POST` | `/media/frame` | `{frame: dataUrl}` → `{visualization, keypoints[1662], boxes}` |
| `GET` | `/media/video_feed` | MJPEG server-side webcam (needs server camera) |
| `GET` | `/inference/labels` | 17 glosses for active model |
| `POST` | `/inference/predict` | `{keypoints[30][1662], generate_audio}` → `{predicted_gloss, confidence, accepted, audio:{en,hi}}`; zeros → `422` |
| `GET` | `/inference/audio/<file>` | serve generated mp3 |
| `GET/POST` | `/inference/models`, `/inference/models/active` | list / switch active model at runtime |

## Configuration (env vars)

`MODEL_NAME` (override registry `active_model`), `MODEL_REGISTRY_PATH`,
`PREDICTION_THRESHOLD` (0.7), `MAX_SEQ_LENGTH` (30), `TTS_ENABLED`,
`TRANSLATION_TARGET_LANGUAGE` (hindi), `AUDIO_OUTPUT_DIR`, `HOST/PORT`,
`CAMERA_INDEX`, `MAX_UPLOAD_SIZE_MB`, `TEMPLATES_DIR/STATIC_DIR`.

## Known Gaps (next work)

1. `tests/`, `scripts/`, `data/` empty — no automated coverage.
2. Accuracy: best video model 47.7% (transformer_v1) vs 80% gate. Levers, in order:
   train augmentation (`augment_sequence` exists but unused in torch loop), small-RNN
   ablations (`--hidden-size/--num-layers` flags ready), uniform 30-frame sampling
   (clips average 87 frames; last-30 truncation discards 2/3), face-mask experiment
   (1404/1629 near-constant features), per-feature standardization.
3. `capture_service`, `media_service`, `model_repository`, `audio_repository` implemented but unwired.
4. TTS/translation synchronous in `POST /predict` (returns null-audio gracefully offline, but blocks thread when enabled) — move to queue + cache.
5. Serving still on WEBCAM_1662; torch v2 models need the Phase-7 cutover to VIDEO_1629
   (`assert_compatible` guards `predict()` with a 400 until then).
6. `requirements.txt` (legacy all-in-one) overlaps `requirements/` splits; `README.md` still documents deleted `realtime_prediction.py` flow.
7. Language fit: training signs are ASL (WL-ASL) but audio output is Hindi — evaluate
   ISL-aligned data (INCLUDE/FDMSE-ISL) before production (see Dataset Summary).
