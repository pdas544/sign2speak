# AGENTS.md — sign2speak_data

## Doc hierarchy (trust order)

- `production-plan.md` = plan of record (video-first, PyTorch-only matrix). Follow it.
- `readme-new.md` = current architecture/API reference. `README.md` is **stale** (references deleted `realtime_prediction.py`) — do not follow it.
- `progress.md` checkboxes mean "file created", **not** wired/tested. Verify before claiming done.
- `safe-to-remove.md` = cleanup log of 24 deleted root files + keep-table. Root `.py` files that remain are load-bearing — do not delete.

## Boot & smoke test

```bash
python app/main.py                       # PORT env, default 8000
python3 -c "
from app.main import create_app
app = create_app(); c = app.test_client()
print(c.get('/live').status_code)                 # 200
print(c.get('/health').get_json()['status'])      # ok
print(c.get('/inference/labels').get_json()['count'])  # 17 (TF default)
"
```

- `POST /inference/predict` with zeros → `422` (quality gate, correct). Random motion → `200`. `generate_audio:true` without `deep-translator`/`pyttsx3` → `200` with `audio:{en:null,hi:null}` (graceful, not a bug).
- `tests/` is empty (`.gitkeep` only), no pytest config. The Flask test-client snippet above **is** the verification step.

## Hard rules (learned the hard way)

1. **Never add top-level imports of heavy deps** (`torch`, `mediapipe`, `cv2`, `pyttsx3`, `deep_translator`). `import app.main` must succeed without them. Use the lazy `_require_*()` pattern in `torch_backend.py`, `media_controller.py`, `translation_service.py`, `tts_service.py`, `tts_helper.py`.
2. **Two keypoint layouts — never mix.** `VIDEO_1629` = `[pose99, lh63, rh63, face1404]` (no visibility) vs `WEBCAM_1662` = `[pose132, face1404, lh63, rh63]` (visibility at index 3). `app/services/keypoint_service.py` is the source of truth; both media paths delegate to it. Serving stays on 1662 until a v2 torch model wins.
3. **`processed/metadata.csv` has 127 duplicate `file_path` rows** ("skipped (already processed)" repeats). Always load via `ml.training.datasets.video_keypoints.read_split()` (dedupes → 336/109/86), never raw csv.
4. **`evaluate_model._normalize_for_model()` silently truncates/pads feature dims.** A 1629-vs-1662 comparison through it is meaningless — check dims first (`keypoint_service.assert_compatible`).
5. **Registry over settings.** Active model = `models/registry/registry.json:active_model`, overridable via `MODEL_NAME` env. `trained_at:null` = pre-registry artifact, untrusted. `Settings.model_path` default (transformer `.pth`) disagrees with registry default (TF `.h5`) — registry wins at runtime.
6. **`train_cnn_lstm --dataset-mode auto` falls back to legacy** (`test_data/`) when `processed/datasets/cnn_lstm/` is absent — which is the current state. Don't mistake a legacy run for a video-data run; check the logged `Dataset mode:` line.
7. **`.gitignore` hides `videos/`, `mp_data/`, `test_data/`, `outputs/`, `audio/`, `processed/keypoints/`.** Data and artifacts never commit. `git status` showing only code changes is expected.

## Environment

- Python 3.10 (pyenv), no repo venv. `torch 2.14+cpu` and `flask` were pip-installed into the environment during this session; `requirements/` splits are aspirational (api pulls both flask+fastapi, training pulls both tf+torch — install only what you need).
- TF `.h5` loads have a `time_major` compat shim in `tf_backend.py` — don't remove it.
- `videos_top_10_gloss/` exists but is **empty**; `videos/` (101 glosses) was never used in training; the 15-gloss video set (`selected_videos/` lineage) is the comparison dataset.
