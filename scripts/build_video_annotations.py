"""
scripts/build_video_annotations.py
──────────────────────────────────
Build an extraction annotation file for the videos/ reserve directory.

Matches videos/<our-label>/*.mp4 against wlasl_dataset.json instances by
YouTube video_id (filename convention: {video_id}_{title}.mp4 — the same
convention extract-keypoints-full.find_video_file() looks up).

- Label assigned = our folder name (WLASL gloss names may differ, e.g.
  folder 'beautiful' vs WLASL gloss 'pretty').
- Skips video_ids already present in processed/metadata.csv (dedupes the
  selected_videos/ overlap).
- Reuses each instance's WLASL split for consistency with existing rows.

Output: annotations/videos_15gloss.json  (rows: text, split, url,
video_id, signer_id) + stdout coverage report.

Usage:
    python3 scripts/build_video_annotations.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.datasets.video_keypoints import VIDEO_LABELS  # noqa: E402


WLASL_JSON = PROJECT_ROOT / "wlasl_dataset.json"
VIDEOS_DIR = PROJECT_ROOT / "videos"
METADATA_CSV = PROJECT_ROOT / "processed" / "metadata.csv"
OUTPUT_JSON = PROJECT_ROOT / "annotations" / "videos_15gloss.json"


def _youtube_id_from_url(url: str) -> str | None:
    """Same convention as extract-keypoints-full.extract_video_id_from_url."""
    for pattern in (r"(?:v=|/)([0-9A-Za-z_-]{11}).*", r"youtu.be/([0-9A-Za-z_-]{11})"):
        match = re.search(pattern, url or "")
        if match:
            return match.group(1)
    return None


def _youtube_id_from_filename(name: str) -> str | None:
    """Filenames are {youtube_id}_{title}.mp4; ids are exactly 11 chars."""
    candidate = name[:11]
    if len(candidate) == 11 and re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
        return candidate
    return None


def _load_existing_video_ids() -> set[str]:
    """Video IDs already extracted (selected_videos/ lineage)."""
    ids: set[str] = set()
    if not METADATA_CSV.exists():
        return ids
    with METADATA_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("video_id"):
                ids.add(row["video_id"].strip())
    return ids


def _existing_split_ratios() -> dict[str, dict[str, float]]:
    """Per-gloss split fractions in processed/metadata.csv (deduped)."""
    ratios: dict[str, dict[str, float]] = {}
    if not METADATA_CSV.exists():
        return ratios
    seen: set[str] = set()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with METADATA_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row.get("file_path") or row.get("video_id")
            if not key or key in seen:
                continue
            seen.add(key)
            counts[row["label"]][row["split"]] += 1
    for label, dist in counts.items():
        total = sum(dist.values())
        ratios[label] = {s: c / total for s, c in dist.items()}
    return ratios


def _draw_split(rng: random.Random, ratios: dict[str, float]) -> str:
    """Stratified draw mirroring the gloss's existing split distribution."""
    if not ratios:
        return "train"
    roll = rng.random()
    cumulative = 0.0
    for split in ("train", "val", "test"):
        cumulative += ratios.get(split, 0.0)
        if roll < cumulative:
            return split
    return "train"


def _index_wlasl() -> dict[str, dict]:
    """Map YouTube video_id -> instance dict across all 2000 WLASL glosses.

    NOTE: WLASL's own `video_id` field is an internal integer id, NOT the
    YouTube id used in local filenames — index by the URL instead.
    """
    data = json.loads(WLASL_JSON.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for entry in data:
        for inst in entry.get("instances", []):
            yid = _youtube_id_from_url(str(inst.get("url", "")))
            if yid and yid not in index:
                index[yid] = inst
    return index


def main() -> None:
    wlasl = _index_wlasl()
    print(f"[Annotations] WLASL index: {len(wlasl)} video ids")
    existing = _load_existing_video_ids()
    print(f"[Annotations] Already extracted: {len(existing)} video ids")
    ratios = _existing_split_ratios()
    rng = random.Random(42)

    rows: list[dict] = []
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    signers: dict[str, set] = defaultdict(set)

    for label in VIDEO_LABELS:
        gloss_dir = VIDEOS_DIR / label
        if not gloss_dir.is_dir():
            print(f"[Annotations] WARNING: missing dir videos/{label}")
            continue
        files = sorted(
            f for f in gloss_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        for path in files:
            yid = _youtube_id_from_filename(path.name)
            if yid is None:
                stats[label]["unmatched_file"] += 1
                continue
            inst = wlasl.get(yid)
            if inst is None:
                # Not in WLASL metadata (different crawl): keep the file with
                # a stratified split draw. Recorded with url="" and the
                # extractor matches it via the explicit video_id field.
                split = _draw_split(rng, ratios.get(label, {}))
                rows.append({
                    "text": label,
                    "split": split,
                    "url": "",
                    "video_id": yid,
                    "signer_id": None,
                })
                stats[label]["matched_unlisted"] += 1
                continue
            if yid in existing:
                stats[label]["duplicate_of_processed"] += 1
                continue
            rows.append({
                "text": label,
                "split": str(inst.get("split", "train")).strip().lower(),
                "url": str(inst.get("url", "")),
                "video_id": yid,
                "signer_id": inst.get("signer_id"),
            })
            stats[label]["matched"] += 1
            if inst.get("signer_id") is not None:
                signers[label].add(inst.get("signer_id"))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n[Annotations] Wrote {len(rows)} rows -> {OUTPUT_JSON}")
    print(f"\n{'gloss':10s} {'matched':>7s} {'dup':>5s} {'unmatch':>7s} signers")
    for label in VIDEO_LABELS:
        s = stats[label]
        print(f"{label:10s} {s.get('matched',0):>7d} "
              f"{s.get('duplicate_of_processed',0):>5d} "
              f"{s.get('unmatched_file',0):>7d} {len(signers[label])}")


if __name__ == "__main__":
    main()
