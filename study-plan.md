# Study Plan — Sign2Speak techniques (~11 weeks, 3–5 h/week)

Goal: cover ~80% of the techniques this codebase uses. Each module maps to repo
files and ends with a hands-on exercise **in this codebase**.
Weekly shape (~4h): 90 min video/reading → 90 min code-along → 60 min repo exercise.

## 1. PyTorch fluency
Tensors, `nn.Module`, DataLoader, autograd, AdamW, schedulers.
- [PyTorch tutorials (60-min Blitz + Deep Learning with PyTorch)](https://docs.pytorch.org/tutorials/)
- Exercise: rewrite `model_gru.py`'s forward pass from memory.

## 2. How training actually works
CE loss, AdamW, BatchNorm, LR schedules, early stopping, reading train/val curves.
- [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
  (makemore + WaveNet + GPT track our exact loop)
- Exercise: explain `ml/training/torch_common.py` + narrate one `history.json` curve.

## 3. RNN / LSTM / GRU (and why ours collapsed)
Gates, vanishing gradients, bidirectionality; 5–7M params vs 336 clips.
- [PyTorch Sequence Models tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- Exercise: diff `model_lstm.py` vs `model_gru.py`; explain the 5.8% GRU run.

## 4. CNNs for sequences: TCN
Causal dilated convolutions, receptive-field arithmetic.
- Bai et al. 2018 — [arXiv:1803.01271](https://arxiv.org/abs/1803.01271), plus Karpathy's WaveNet video
- Exercise: compute `model_tcn.py`'s receptive field; explain why it won (83.7%).

## 5. Transformers
Self-attention, multi-head attention, positional encoding, encoder stacks.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face NLP course (transformer chapters)](https://huggingface.co/learn/nlp-course)
- Exercise: map each `model_transformer.py` block to a paper section.

## 6. Evaluation methodology
Stratified splits, leakage (augment-after-split), per-class P/R/F1, confusion
matrices, offline accuracy vs production gates.
- [scikit-learn model evaluation guides](https://scikit-learn.org/stable/model_evaluation.html)
- Exercise: reproduce one leaderboard row via `compare_models.py`; explain the
  sampler-mismatch incident (49% vs true 65%) in `progress.md`.

## 7. Pose, keypoints + CV plumbing
MediaPipe Holistic graph, landmark topology, coordinate frames, OpenCV capture.
- [MediaPipe Solutions guide](https://ai.google.dev/edge/mediapipe/solutions/guide)
- Exercise: draw both layouts in `keypoint_service.py` from memory (dims + order).

## 8. Serving ML with Flask
REST design, blueprints, request validation, lazy heavy imports, tiered fallback.
- [Flask docs (quickstart + API patterns)](https://flask.palletsprojects.com/)
- Exercise: trace `POST /inference/predict` from `app.js` through adaptation to backend.

## 9. MLOps-lite
Registries, manifests, seeds/reproducibility, experiment discipline.
- Internal: `model_registry_service.py`, `ml/training/common.py`, `AGENTS.md`.
- Exercise: register a retrained model and roll it back via the registry.

## 10. Graph nets preview (supports pretrained-plan.md)
Skeleton graphs, spatial-temporal convolution, masked self-supervised pretraining.
- [ST-GCN](https://arxiv.org/abs/1801.07455), [SignBERT+](https://arxiv.org/abs/2305.04868)
- Exercise: sketch how a 225-d masked frame becomes graph nodes/edges.

## Paid resources
Only one conditional pick: **Coursera Deep Learning Specialization, Course 5
(Sequence Models)** — most content auditable free; pay only if graded
assignments + certificate motivate you. (Fast.ai, free, is a fine substitute
for modules 1–2 if you prefer top-down teaching.)

## Capstone (week 11)
Add one documented experiment to the matrix (e.g. scheduled-sampling variant),
compare it, and write it up in `progress.md`.
