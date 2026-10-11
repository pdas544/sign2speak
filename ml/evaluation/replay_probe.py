"""Replay probe exports (with captured keypoints) as a live regression set.

Usage:
    python3 -m ml.evaluation.replay_probe probe_xxx.json [--model NAME]

Each row's stored WEBCAM_1662 window is re-run through InferenceService
(which adapts 1662 -> model dim exactly like the live path), so old probe
sessions become repeatable accuracy numbers for every new model.
Rows without keypoints (pre-capture exports) are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_json")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from app.services.inference_service import InferenceService

    rows = json.load(open(args.probe_json))
    svc = InferenceService()
    if args.model:
        svc.switch_model(args.model)

    labels = svc.labels()
    n_total = n_replay = n_correct = 0
    print(f"model={svc.active_model_info().get('name')} labels={len(labels)}")
    for r in rows:
        kp = r.get("keypoints")
        if not kp:
            continue
        n_replay += 1
        gloss, conf = svc.predict(np.asarray(kp, dtype=np.float32))
        ok = gloss == r.get("true_gloss")
        n_correct += ok
        flag = "OK " if ok else "MISS"
        print(f"[{flag}] true={r.get('true_gloss')} used_hand={r.get('used_hand')} "
              f"-> {gloss}@{conf:.2f} (was {r.get('predicted_gloss')}@{r.get('confidence') or 0:.2f})")
    n_total = len(rows)
    print(f"replayed {n_replay}/{n_total} rows, live-replay acc={n_correct/max(n_replay,1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
