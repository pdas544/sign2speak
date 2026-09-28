# Refactor Progress

## Completed
- [x] Create app config package
- [x] Create settings module ([app/config/settings.py](app/config/settings.py))
- [x] Create logging module ([app/config/logging.py](app/config/logging.py))
- [x] Create media controller ([app/controllers/media_controller.py](app/controllers/media_controller.py))
- [x] Create health controller ([app/controllers/health_controller.py](app/controllers/health_controller.py))
- [x] Create inference controller ([app/controllers/inference_controller.py](app/controllers/inference_controller.py))
- [x] Create controllers init/registry ([app/controllers/__init__.py](app/controllers/__init__.py))
- [x] Create Flask app entrypoint ([app/main.py](app/main.py))

## Services Directory
- [x] Create services directory ([app/services](app/services))
- [x] Create media service ([app/services/media_service.py](app/services/media_service.py))
- [x] Create inference service ([app/services/inference_service.py](app/services/inference_service.py))
- [x] Create translation service ([app/services/translation_service.py](app/services/translation_service.py))
- [x] Create tts service ([app/services/tts_service.py](app/services/tts_service.py))
- [x] Create capture service ([app/services/capture_service.py](app/services/capture_service.py))

## Repositories Directory
- [x] Create repositories directory ([app/repositories](app/repositories))
- [x] Create model repository ([app/repositories/model_repository.py](app/repositories/model_repository.py))
- [x] Create audio repository ([app/repositories/audio_repository.py](app/repositories/audio_repository.py))

## Models Directory
- [x] Create models directory ([app/models](app/models))
- [x] Create request models ([app/models/request_models.py](app/models/request_models.py))
- [x] Create response models ([app/models/response_models.py](app/models/response_models.py))

## Views Directory
- [x] Extract shared styles ([app/views/static/css/shared.css](app/views/static/css/shared.css))
- [x] Extract index page styles ([app/views/static/css/index.css](app/views/static/css/index.css))
- [x] Extract result page styles ([app/views/static/css/result.css](app/views/static/css/result.css))
- [x] Extract recognition page JS ([app/views/static/js/app.js](app/views/static/js/app.js))
- [x] Wire Flask static/template folders ([app/main.py](app/main.py))

## ML Directory
- [x] Create ml root directory ([ml](ml))
- [x] Create training directory ([ml/training](ml/training))
- [x] Create shared training utilities ([ml/training/common.py](ml/training/common.py))
- [x] Create CNN-LSTM training entry script ([ml/training/train_cnn_lstm.py](ml/training/train_cnn_lstm.py))
- [x] Create LSTM training entry script ([ml/training/train_lstm.py](ml/training/train_lstm.py))
- [x] Create transformer training entry script ([ml/training/train_transformer.py](ml/training/train_transformer.py))
- [x] Create datasets placeholder ([ml/training/datasets/.gitkeep](ml/training/datasets/.gitkeep))
- [x] Create evaluation directory ([ml/evaluation](ml/evaluation))
- [x] Create single-model evaluation script ([ml/evaluation/evaluate_model.py](ml/evaluation/evaluate_model.py))
- [x] Create multi-model comparison script ([ml/evaluation/compare_models.py](ml/evaluation/compare_models.py))
- [x] Create evaluation package init ([ml/evaluation/__init__.py](ml/evaluation/__init__.py))
- [x] Create preprocessing directory ([ml/preprocessing](ml/preprocessing))
- [x] Create preprocessing package init ([ml/preprocessing/__init__.py](ml/preprocessing/__init__.py))
- [x] Create augmentation utilities ([ml/preprocessing/augment.py](ml/preprocessing/augment.py))
- [x] Create keypoint extraction wrapper ([ml/preprocessing/extract_keypoints.py](ml/preprocessing/extract_keypoints.py))
- [x] Create dataset builder script ([ml/preprocessing/build_dataset.py](ml/preprocessing/build_dataset.py))


## Project Scaffold (post-ML)
- [x] Create canonical data directories ([data/raw](data/raw), [data/processed](data/processed), [data/keypoints](data/keypoints))
- [x] Create test structure ([tests/unit](tests/unit), [tests/integration](tests/integration), [tests/e2e](tests/e2e))
- [x] Create scripts directory ([scripts](scripts))
- [x] Create split requirements directory ([requirements](requirements))
- [x] Add requirements manifests ([requirements/base.txt](requirements/base.txt), [requirements/api.txt](requirements/api.txt), [requirements/training.txt](requirements/training.txt), [requirements/dev.txt](requirements/dev.txt))
- [x] Create cleanup audit file ([safe-to-remove.md](safe-to-remove.md))

## Multi-model Backend
- [x] Create backend abstraction base ([app/services/inference_backends/base.py](app/services/inference_backends/base.py))
- [x] Create backends package init ([app/services/inference_backends/__init__.py](app/services/inference_backends/__init__.py))
- [x] Create TensorFlow backend ([app/services/inference_backends/tf_backend.py](app/services/inference_backends/tf_backend.py))
- [x] Create PyTorch backend ([app/services/inference_backends/torch_backend.py](app/services/inference_backends/torch_backend.py))
- [x] Create model registry seed file ([models/registry/registry.json](models/registry/registry.json))
- [x] Create model registry service ([app/services/model_registry_service.py](app/services/model_registry_service.py))

## Modified Files (multi-model refactor)
- [x] Integrate CNN-LSTM training with preprocessing datasets (`--dataset-mode auto|processed|legacy`) ([ml/training/train_cnn_lstm.py](ml/training/train_cnn_lstm.py))
- [x] Add `model_name` + `model_registry_path` to Settings ([app/config/settings.py](app/config/settings.py))
- [x] Refactor InferenceService to use registry + backends ([app/services/inference_service.py](app/services/inference_service.py))
- [x] Refactor InferenceController; add `/inference/models` + `/inference/models/active` ([app/controllers/inference_controller.py](app/controllers/inference_controller.py))
- [x] Update root `/` to render `index.html`; move old route map to `/routes` ([app/main.py](app/main.py))

## Notes
- Progress file must be updated whenever a new directory or file is created.
- Active model is set in `models/registry/registry.json` (key: `active_model`) or overridden via `MODEL_NAME` env var.
- To train and register a new CNN-LSTM model: `python -m ml.training.train_cnn_lstm --model-name my_model --set-active`

## PyTorch Comparison Matrix (video lineage, 15 glosses, 336/109/86 clips)

Unified trainer: `python -m ml.training.train_torch --arch {lstm,gru,cnn_lstm,tcn,transformer} --model-name X`
(factory: [ml/training/arch_factory.py](ml/training/arch_factory.py), loop: [ml/training/torch_common.py](ml/training/torch_common.py)).

### Round 1 — no augmentation, last-30 sampler (test acc, n=86)
| Model | Acc | F1 | p50 | Size |
|---|---|---|---|---|
| transformer_v1 (legacy artifact) | 47.7% | 0.43 | 1.6ms | 13.2MB |
| transformer_v2 | 44.2% | 0.38 | 1.3ms | 13.2MB |
| tcn_v1 | 41.9% | 0.37 | 2.2ms | 4.5MB |
| cnn_lstm_torch_v1 | 38.4% | 0.33 | 0.8ms | 4.9MB |
| lstm_baseline_v1 | 12.8% | 0.04 | 5.3ms | 27MB |
| gru_v1 | 5.8% | 0.01 | 7.5ms | 20MB |

### Round 2 — `--augment-copies 3` + sampling/mask variants (test acc, n=86)
| Model | Acc | F1 | Variant |
|---|---|---|---|
| tcn_uniform_v1 | **65.1%** | 0.61 | uniform-30 sampler |
| tcn_noface_v1 | 52.3% | 0.52 | face-masked (225-d), explicit serving adapter |
| transformer_v3 | 46.5% | 0.41 | aug only |
| lstm_noface_v1 | 38.4% | 0.33 | masked (was 12.8% unmasked) |
| cnn_lstm_torch_v2 | 37.2% | 0.33 | aug only |
| tcn_v2 | 36.1% | 0.32 | aug only, last-30 |
| gru_128x2_v1 | 26.7% | 0.25 | small (was 5.8%) |
| lstm_128x2_v1 / lstm_128x1_v1 | 14% / 12% | — | small still collapses |

### Round 3 — uniform + noface combo (test acc, n=86)
| Model | Acc | F1 | p50 | Size |
|---|---|---|---|---|
| tcn_uniform_noface_v1 | **69.8%** | 0.70 | 1.3ms | 1.8MB |
| transformer_uniform_noface_v1 | 68.6% | 0.67 | 1.1ms | 11.8MB |

### Round 3 — uniform + noface combo (test acc, n=86)
| Model | Acc | F1 | p50 | Size |
|---|---|---|---|---|
| tcn_uniform_noface_v1 | **69.8%** | 0.70 | 1.3ms | 1.8MB |
| transformer_uniform_noface_v1 | 68.6% | 0.67 | 1.1ms | 11.8MB |

Stacking works: uniform sampling × face removal × augmentation compose
(47.7% → 65.1% → 69.8%). Still 10 pts short of the 80% gate.

### Round 4 — videos/ expansion (278 new clips, 531 → 809 unique)
- Step 0 (per-class analysis of 69.8% model): 6/15 glosses perfect; weak =
  good 0.20, like 0.33, go 0.40, hello 0.50 (high-conf confusions → hello/no,
  like/happy), happy 0.62. Mixed scarcity + representation errors.
- Steps 1–2: [scripts/build_video_annotations.py](scripts/build_video_annotations.py)
  matched `videos/` files to WLASL YouTube ids (deduped 14 overlap, stratified
  splits for unlisted crawls) → [annotations/videos_15gloss.json](annotations/videos_15gloss.json)
  (278 rows); `extract-keypoints-full.py` parameterized (`--annotations/--videos-dir/--output-dir`,
  append-merge, explicit `video_id`) — 278/278 success, 0 failed.
- Dataset now: train 516 / val 158 / test 135 (`go` 6→11, `like` 28→53 train).
- Step 3a (TCN uniform+noface, aug ×3): `tcn_uniform_noface_v2` → **83.7%** / F1 0.81.
- Step 3b (same, aug ×6): `tcn_uniform_noface_v3` → 83.0% — diminishing returns, ×3 is the sweet spot.
- Head-to-head on new test (n=135): v2 83.7% > v3 83.0% > v1 79.3% (old model rescored).
- **Gate verdict**: overall ≥0.80 PASS, worst-gloss ≥0.70 FAIL —
  good 0.33, go 0.40, hello 0.60 (`like` fixed 0.33→1.00 by new data).
  Ship-blocked for good/go/hello pending targeted collection or scope trim.
Eval harness is sampler-aware (`evaluate_model` reads registry hyperparams; incident:
uniform model scored 49% before the fix vs true 65%).

### Serving status
- Generic `TorchBackend` (arch dispatch) + `serving_check` dim guard in `predict()`
  (mismatch → HTTP 400, verified). Masked models serve via declared pose+hands mask.
- Flask-verified: `tcn_v1` real clip → 200; masked smoke model → 200.
- **API/interface phase: active model is `tcn_uniform_noface_v3` (83.0%, 225-d masked).**
  `InferenceService.predict` adapts inputs via `keypoint_service.adapt_for_model`
  (exact passthrough, documented 1662→1629 conversion, declared face mask —
  verified identical predictions on 1629-direct vs 1662-converted inputs).
  UI `/` unchanged (Start/Stop → prediction + EN/HI audio players).
- Registry: 18 models. Previous default was `cnn_lstm_default` (webcam TF).
