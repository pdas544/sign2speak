# Safe To Remove — Cleanup Log

> Updated after root cleanup (Sept 2026). All entries below were deleted from
> the repo root; recoverable via `git log` / `git restore` if needed.
>
> Second wave (Oct 2026): legacy dirs/files below, same recoverability.
> `videos_all/` explicitly kept for future use. Gitignored data dirs
> (`videos/`, `selected_videos/`, `test_data/`, `mp_data/`, `outputs/`,
> `processed/{keypoints,train,val,test}/`) are load-bearing and untouched —
> deleting those would be permanent (not in git).

## Removed — Legacy Directories/Files (Oct 2026, zero references verified)

- `src/` (dataload, deploy, evaluate, model, training — superseded, unimported)
- `annotations/MSASL_{train,val,test}.json` (only stale README/registry-string refs)
- `class_distribution.png`, `wsasl_gloss_video_counts.png` (unreferenced artifacts)
- `videos_top_10_gloss/` (was empty; plain removal, nothing lost)

## Removed — Low Risk (legacy one-off utilities / scratch)

These were not imported by `app/*` or `ml/*`:

- `analyze.py`
- `analyze_dataset.py`
- `analyze_filtered_json.py`
- `analyze_msasl.py`
- `analyze_wlasl.py`
- `download-videos.py`
- `move_videos.py`
- `renamedir.py`
- `updated-filtered-annotations.py`
- `testfile.py`
- `tts_test.py`
- `test_translation.py`
- `sign2speech_pipeline_tcn.py`

## Removed — Medium Risk (legacy entrypoints, superseded by `app/main.py`)

Verified via boot + smoke test on port 8001 (`/health ok`,
`/inference/labels 17 labels`, `POST /predict 200/422`) before removal:

- `api_server.py`
- `api-test.py`

## Removed — Extra Legacy (verified unimported, superseded)

- `realtime_prediction.py` — legacy desktop OpenCV loop (replaced by `app` + browser UI)
- `model_baseline.py` — unused model variant, nothing imports it
- `model_cnn_lstm.py` — unused variant (canonical training lives in `ml/training/train_cnn_lstm.py` + `action_recognition_model.py`)
- `model.py` — unused stub, nothing imports it
- `extract_keypoints_lstm.py` — legacy capture script (canonical path is `ml/preprocessing/extract_keypoints.py` → `extract-keypoints-full.py`)
- `requirements_tts.txt`, `requirements_upload.txt` — superseded by `requirements/{base,api,training,dev}.txt`
- `temp.md`, `test-prediction.md` — scratch notes

## Kept In Root (actively used — do not remove)

| File | Used by |
|---|---|
| `action_recognition_model.py` | `ml/training/train_cnn_lstm.py` (`ActionRecognitionModel`) |
| `test_action_recognition.py` | `ml/training/train_cnn_lstm.py` (`SignLanguageDetector`), `action_recognition_model.py` |
| `model_transformer.py` | `app/services/inference_backends/torch_backend.py`, `ml/training/train_transformer.py` |
| `tts_helper.py` | `app/services/tts_service.py`, `app/services/inference_service.py` (lazy) |
| `train_lstm.py` | `ml/training/train_lstm.py` wrapper (`runpy`) |
| `dataset_lstm.py` | root `train_lstm.py` (`create_data_loaders`) |
| `model_lstm.py` | root `train_lstm.py` (`ASLKeypointLSTM`) |
| `extract-keypoints-full.py` | `ml/preprocessing/extract_keypoints.py` wrapper |
| `requirements.txt` | legacy all-in-one install pointer (canonical splits live in `requirements/`) |

## Restore If Needed

```bash
# list deleted files
git log --diff-filter=D --name-only --oneline | head -n 40
# restore a single file, e.g.:
git restore --source=HEAD~1 -- api_server.py
```
