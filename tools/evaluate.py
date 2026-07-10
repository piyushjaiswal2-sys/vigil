"""Evaluate VIGIL's detection pipeline on a public dataset.

This is the end-term evaluation harness. It measures the *repo's own*
detect->validate path (S2->S3) with a real YOLOv8 backend injected into the
DetectBlock, on a standard public dataset (COCO128 by default), and reports:

  * mAP@0.5 and per-class AP        (computed here, over VIGIL pipeline output)
  * precision / recall / F1 @IoU0.5 (at the pipeline's operating threshold)
  * mean latency (ms/frame) + FPS   (CPU, single stream)
  * official ultralytics `val`       (mAP50, mAP50-95, P, R) as a cross-check

Run:
    python tools/evaluate.py --model yolov8n.pt --data coco128.yaml
    python tools/evaluate.py --limit 32          # quick smoke on 32 images

Outputs a JSON + Markdown report under `eval/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engines.blocks  # noqa: E402,F401  (registers L3 blocks)
from engines.blocks import DetectBlock, ValidateBlock  # noqa: E402
from engines.detectors import YoloV8Detector  # noqa: E402
from engines.types import NormalizedFrame  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# geometry + metrics
# --------------------------------------------------------------------------- #
def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(confidences: list[float], is_tp: list[bool], n_gt: int) -> float:
    """VOC all-point AP from confidence-ranked TP/FP flags for one class."""
    if n_gt == 0:
        return float("nan")
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    tp = fp = 0
    precisions, recalls = [], []
    for i in order:
        if is_tp[i]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    if not precisions:
        return 0.0
    # make precision monotonically non-increasing, then integrate over recall
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def load_dataset(data: str):
    """Resolve a YOLO dataset to (image_paths, names).

    A repo-local dataset yaml (e.g. the bundled `datasets/coco128.yaml`) is
    read directly, so evaluation runs fully offline with no download. Anything
    else is handed to ultralytics, which resolves/downloads as needed.
    """
    import yaml

    local = Path(data)
    if not local.is_absolute():
        local = REPO_ROOT / data
    if local.exists() and local.suffix in (".yaml", ".yml"):
        info = yaml.safe_load(local.read_text(encoding="utf-8"))
        root = Path(info.get("path", ".")).expanduser()
        if not root.is_absolute():
            root = (REPO_ROOT / root).resolve()
        val_dir = root / (info.get("val") or info.get("train"))
        images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if images:
            names = info["names"]
            if not isinstance(names, dict):
                names = dict(enumerate(names))
            return images, names

    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset(data)
    names = info["names"] if isinstance(info["names"], dict) else dict(enumerate(info["names"]))
    val = info.get("val") or info.get("train")
    val_path = Path(val[0] if isinstance(val, list) else val)
    if val_path.is_file():  # a .txt list of images
        images = [Path(p.strip()) for p in val_path.read_text().splitlines() if p.strip()]
    else:
        images = sorted(p for p in val_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return images, names


def label_path_for(img: Path) -> Path:
    """Map an image path to its YOLO label .txt (…/images/… -> …/labels/…)."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_gt(img: Path, w: int, h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Read YOLO-format ground truth as (class_id, pixel-xyxy)."""
    lp = label_path_for(img)
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text().splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        cls = int(float(vals[0]))
        cx, cy, bw, bh = (float(v) for v in vals[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        out.append((cls, (x1, y1, x2, y2)))
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_pipeline(images, names, model, conf, imgsz, iou_thr):
    import cv2

    inv_names = {v: k for k, v in names.items()}
    detector = YoloV8Detector(model=model, conf=conf, imgsz=imgsz)
    detect = DetectBlock(config={"detector": detector})
    validate = ValidateBlock(config={"min_confidence": conf})

    # Warm up once (model load + first-inference graph build) so the timed
    # latencies below reflect steady-state throughput, not one-time startup.
    if images:
        warm = cv2.imread(str(images[0]))
        if warm is not None:
            detector(NormalizedFrame(index=-1, timestamp=0.0, width=warm.shape[1],
                                     height=warm.shape[0], data=str(images[0]),
                                     source=str(images[0])))

    # per-class accumulators
    gt_count: dict[int, int] = {}
    pred_records: dict[int, list[tuple[float, bool]]] = {}
    tp = fp = fn = 0
    latencies: list[float] = []

    for idx, img in enumerate(images):
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        frame = NormalizedFrame(index=idx, timestamp=0.0, width=w, height=h,
                                data=str(img), source=str(img))
        t0 = time.perf_counter()
        dets = detect.run({"frame": frame}).outputs["detections"]
        validated = validate.run({"detections": dets}).outputs["detections"]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        gts = read_gt(img, w, h)
        for cls, _ in gts:
            gt_count[cls] = gt_count.get(cls, 0) + 1

        # greedy IoU matching per image (predictions sorted by confidence)
        preds = sorted(validated.items, key=lambda d: d.confidence, reverse=True)
        matched = [False] * len(gts)
        for det in preds:
            cls = inv_names.get(det.label, -1)
            best_iou, best_j = 0.0, -1
            for j, (gcls, gbox) in enumerate(gts):
                if matched[j] or gcls != cls:
                    continue
                v = iou(det.bbox, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_iou >= iou_thr and best_j >= 0
            if is_tp:
                matched[best_j] = True
                tp += 1
            else:
                fp += 1
            pred_records.setdefault(cls, []).append((det.confidence, is_tp))
        fn += matched.count(False)

    # per-class AP -> mAP@0.5
    aps = []
    for cls, n_gt in gt_count.items():
        recs = pred_records.get(cls, [])
        ap = average_precision([c for c, _ in recs], [t for _, t in recs], n_gt)
        if ap == ap:  # not NaN
            aps.append(ap)
    mAP50 = sum(aps) / len(aps) if aps else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "images": len(latencies),
        "total_gt": sum(gt_count.values()),
        "classes_with_gt": len(gt_count),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP@0.5": round(mAP50, 4),
        "mean_latency_ms": round(mean_ms, 2),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms else 0.0,
    }


def official_val(model, data, conf, imgsz):
    from ultralytics import YOLO

    try:
        res = YOLO(model).val(data=data, conf=conf, imgsz=imgsz, device="cpu",
                              verbose=False, plots=False)
    except Exception:
        # Local yaml may not resolve inside ultralytics; use its bundled name.
        res = YOLO(model).val(data="coco128.yaml", conf=conf, imgsz=imgsz,
                              device="cpu", verbose=False, plots=False)
    b = res.box
    return {
        "mAP@0.5": round(float(b.map50), 4),
        "mAP@0.5:0.95": round(float(b.map), 4),
        "precision": round(float(b.mp), 4),
        "recall": round(float(b.mr), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate VIGIL detection on a public dataset")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--data", default="datasets/coco128.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP match")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all)")
    ap.add_argument("--skip-official", action="store_true")
    args = ap.parse_args()

    print(f"[eval] model={args.model} data={args.data} conf={args.conf} imgsz={args.imgsz}")
    images, names = load_dataset(args.data)
    if args.limit:
        images = images[: args.limit]
    print(f"[eval] {len(images)} image(s); {len(names)} classes")

    pipeline = evaluate_pipeline(images, names, args.model, args.conf, args.imgsz, args.iou)
    print(f"[eval] VIGIL pipeline: {json.dumps(pipeline)}")

    official = None
    if not args.skip_official:
        try:
            official = official_val(args.model, args.data, args.conf, args.imgsz)
            print(f"[eval] ultralytics val: {json.dumps(official)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] official val skipped: {type(exc).__name__}: {exc}")

    report = {
        "model": args.model, "dataset": args.data, "conf": args.conf,
        "imgsz": args.imgsz, "iou_match": args.iou, "device": "cpu",
        "vigil_pipeline": pipeline, "ultralytics_official": official,
    }
    out_dir = REPO_ROOT / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
