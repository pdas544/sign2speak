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
│   │   │   ├── tf_backend.py           # .h5 via tensorflow (lazy)
│   │   │   └── torch_backend.py        # .pth via SignLanguageTransformer (lazy torch)
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
│   │   ├── train_cnn_lstm.py           # --dataset-mode auto|processed|legacy, --set-active
│   │   ├── train_lstm.py               # wrapper → root train_lstm.py
│   │   ├── train_transformer.py        # wrapper → root model_transformer.py
│   │   ├── common.py                   # create_run_dir / register_model / write_manifest
│   │   └── datasets/.gitkeep           # placeholder
│   └── evaluation/
│       ├── evaluate_model.py           # single registered model over metadata.csv
│       └── compare_models.py           # leaderboard across registry
├── models/
│   ├── action_model_cnn_lstm_new.h5    # active TF default (+ .keras variants)
│   └── registry/registry.json          # {active_model, models{cnn_lstm_default(tf), transformer_v1(pytorch)}}
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

## Root Files (only 8 `.py` — all load-bearing)

| File | Used by |
|---|---|
| `action_recognition_model.py` | `ml/training/train_cnn_lstm.py` |
| `test_action_recognition.py` | `train_cnn_lstm`, `action_recognition_model` |
| `model_transformer.py` | `torch_backend`, `ml/training/train_transformer.py` |
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
2. No `keypoint_service.py` — `media_service` (holistic 1662-d) vs controller `MediaProcessor` (pose+hands, face=zeros) diverge; unify.
3. `capture_service`, `media_service`, `model_repository`, `audio_repository` implemented but unwired.
4. TTS/translation synchronous in `POST /predict` (returns null-audio gracefully offline, but blocks thread when enabled) — move to queue + cache.
5. Registry dim mismatch: `cnn_lstm 1662` vs `transformer 1629` features, differing label sets — validate on switch.
6. `requirements.txt` (legacy all-in-one) overlaps `requirements/` splits; `README.md` still documents deleted `realtime_prediction.py` flow.
