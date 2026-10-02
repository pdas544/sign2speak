# Pretrained-Model Plan — Sign2Speak

## Goal
Close the live-accuracy gap (best offline 83.7%, gate 80% passed overall but
good/go/hello fail per-gloss) via pretraining. Ranking principle: in-domain
data you own first, external checkpoints only if needed.

## Locked decisions
- **Pretraining source: own `videos/` first** (1356 videos / ~100 glosses, same
  extractor, no licenses, CPU-feasible). External WLASL weights second.
- **ST-GCN: yes, one compact variant** (2–3 blocks, masked 225-d input,
  p50 budget <5ms CPU). Graph weight-sharing suits skeleton + tiny data far
  better than the dense 1629→256 projections that memorized the 336-clip set.

## Why this fits
The gap is generalization across train/serve skew (dictionary videos → live
webcam, holistic → pose+hands history, mirror risk) — not capacity. Pretraining
learns signer/view-invariant motion features from ~3× more clips (85 non-target
glosses = free data); a small head then fits the 15 targets. Precedents:
INCLUDE recipe (pretrained encoder + trained decoder = 94.5% on INCLUDE-50),
FDMSE-ISL pretrain → 97.79% after fine-tune.

## Phase 1 — Supervised transfer on videos/-100 (first)
- Annotation sweep for all ~100 `videos/` glosses (extend
  `scripts/build_video_annotations.py`; WLASL `split` fields give train/val).
- Extract keypoints (parameterized extractor already supports this).
- Train TCN encoder + 100-way head (`tcn_pretrain100`), then fine-tune on 15:
  compare `--freeze-encoder` vs full fine-tune (freezing usually wins here).
- Infra: `train_torch --pretrained <run> [--freeze-encoder]`; registry
  `pretrained_from` field. Eval gates unchanged + live probe set.

## Phase 2 — Masked self-supervised pretraining (only if Phase 1 misses gate)
- BERT-style: mask random joints/frames/clips in videos/-100 keypoints, train
  encoder-decoder reconstruction (SignBERT+ strategy minus MANO complexity).
  Uses even unlabelable clips.
- New `ml/training/pretrain_masked.py` reusing `video_keypoints` loader +
  `torch_common` loop with MSE objective. Fine-tune as Phase 1.

## Phase 3 — External WLASL checkpoints (only if 1–2 stall)
Ranked: **DSTA-SLR** (SOTA, WLASL2000 weights published, <⅓ compute, >2×
speed — realtime-friendly) > **dxli94 Pose-TGCN** (weights + splits + keypoints
published) > SignBERT+ (best paper fit, no official code — avoid).
Cost: keypoint-convention adapter (their joints ≠ our 1629/225), WLASL C-UDA
agreement signature. Budget ~1 day, mostly adapter + verification.

## Non-goals (pretraining won't fix these)
- OOV signs (`brother`/`yes`) — vocabulary/scope work, not representation.
- Confirmed mirror swap — probe verdict + toggle.
- `good/go/hello` failures if label noise — probe replay set decides.

## Order
Probe verdict → Phase 1 → Phase 2 on evidence → Phase 3 if stalled.
