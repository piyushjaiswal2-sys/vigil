"""Evaluate VIGIL's reasoning layer (S4) against a live LLM endpoint.

The reasoning core routes through an OpenAI-compatible `/v1` endpoint
(freellmapi, or any compatible provider) configured via the environment:

    FREELLMAPI_BASE_URL   e.g. https://api.groq.com/openai
    FREELLMAPI_API_KEY    bearer token
    FREELLMAPI_MODEL      model id

This harness reports two things:

  A. Live integration on real detections — run YOLOv8 over a sample of COCO128
     frames, feed the validated detections through the Adjudicator + live LLM,
     and report success rate (LLM used vs heuristic fallback), latency, and
     sample RiskEvents.
  B. Risk-scoring accuracy — a small labelled safe/risky scenario set; check
     whether the model's risk score lands on the correct side of 0.5.

Run (loads .env automatically):
    python tools/eval_reasoning.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency); does not overwrite existing vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


load_dotenv(REPO_ROOT / ".env")

import engines.blocks  # noqa: E402,F401
from agent import Adjudicator, FreeLlmApiProvider  # noqa: E402
from engines.blocks import DetectBlock, ValidateBlock  # noqa: E402
from engines.detectors import YoloV8Detector  # noqa: E402
from engines.types import Detection, NormalizedFrame, ValidatedDetections  # noqa: E402
from tools.evaluate import load_dataset  # noqa: E402

THRESHOLD = 0.5           # risk >= THRESHOLD is predicted "high"
CALL_DELAY_S = 0.4        # gentle pacing for free-tier rate limits


# --------------------------------------------------------------------------- #
# B. labelled safe/risky scenarios
# --------------------------------------------------------------------------- #
def _d(label, conf=0.85):
    return Detection(label=label, confidence=conf, bbox=(0.1, 0.1, 0.5, 0.6))


SCENARIOS = [
    # risky (expect high)
    ("armed_intruder", "a person pointing a knife at a cashier in a store",
     [_d("person"), _d("person"), _d("knife")], "high"),
    ("perimeter_breach", "a person climbing over a restricted-area fence at night",
     [_d("person")], "high"),
    ("crowd_surge", "a dense crowd pushing against a barrier at a stadium exit",
     [_d("person") for _ in range(8)], "high"),
    ("runway_incursion", "a truck driving across an active airport runway",
     [_d("truck"), _d("airplane")], "high"),
    ("weapon_in_school", "a knife visible in a school hallway near students",
     [_d("knife"), _d("person"), _d("person")], "high"),
    # safe (expect low)
    ("empty_office", "an empty office at midday, no people present",
     [_d("chair"), _d("laptop"), _d("keyboard")], "low"),
    ("pet_at_home", "a cat resting on a sofa in a living room",
     [_d("cat"), _d("couch")], "low"),
    ("desk_work", "a person calmly working at an office desk",
     [_d("person"), _d("laptop"), _d("cup")], "low"),
    ("quiet_park", "a bird perched on a bench in a quiet park",
     [_d("bird"), _d("bench")], "low"),
    ("clear_hallway", "a monitored hallway with no people or objects of concern",
     [], "low"),
]


def eval_scenarios(adj: Adjudicator) -> dict:
    rows, correct, llm_used = [], 0, 0
    for name, scene, dets, expected in SCENARIOS:
        vd = ValidatedDetections(frame_index=0, items=dets, dropped=0)
        t0 = time.perf_counter()
        decision = adj.decide(vd, context={"scene": scene})
        dt = (time.perf_counter() - t0) * 1000.0
        ev = decision.event
        predicted = "high" if ev.risk >= THRESHOLD else "low"
        ok = predicted == expected
        correct += ok
        provider = ev.meta.get("provider", "?")
        llm_used += provider == "freellmapi"
        rows.append({"scenario": name, "expected": expected, "risk": round(ev.risk, 3),
                     "predicted": predicted, "correct": ok, "label": ev.label,
                     "provider": provider, "latency_ms": round(dt, 1)})
        time.sleep(CALL_DELAY_S)
    return {
        "n": len(SCENARIOS),
        "accuracy": round(correct / len(SCENARIOS), 4),
        "llm_used": llm_used,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# A. live integration on real COCO128 detections
# --------------------------------------------------------------------------- #
def eval_live(adj: Adjudicator, data: str, model: str, sample: int) -> dict:
    import cv2

    images, _ = load_dataset(data)
    detector = YoloV8Detector(model=model, conf=0.25)
    detect = DetectBlock(config={"detector": detector})
    validate = ValidateBlock(config={"min_confidence": 0.25})

    used, latencies, samples, n = 0, [], [], 0
    for idx, img in enumerate(images):
        if n >= sample:
            break
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        frame = NormalizedFrame(index=idx, timestamp=0.0, width=w, height=h,
                                data=str(img), source=str(img))
        dets = detect.run({"frame": frame}).outputs["detections"]
        validated = validate.run({"detections": dets}).outputs["detections"]
        if not validated.items:
            continue
        n += 1
        t0 = time.perf_counter()
        decision = adj.decide(validated, context={"scene": "a monitored security camera feed"})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ev = decision.event
        if ev.meta.get("provider") == "freellmapi":
            used += 1
        if len(samples) < 6:
            labels = sorted({d.label for d in validated.items})
            samples.append({"image": img.name, "detections": labels,
                            "risk": round(ev.risk, 3), "label": ev.label,
                            "summary": ev.summary, "provider": ev.meta.get("provider")})
        time.sleep(CALL_DELAY_S)

    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "frames_scored": n,
        "llm_success_rate": round(used / n, 4) if n else 0.0,
        "mean_latency_ms": round(mean_ms, 1),
        "samples": samples,
    }


def main() -> int:
    base = os.getenv("FREELLMAPI_BASE_URL", "")
    model = os.getenv("FREELLMAPI_MODEL", "auto")
    if not base:
        print("FREELLMAPI_BASE_URL not set (put it in .env). Aborting.")
        return 1
    print(f"[reason] endpoint={base} model={model}")

    provider = FreeLlmApiProvider()
    adj = Adjudicator(provider=provider)

    print("[reason] A: live integration on COCO128 detections ...")
    live = eval_live(adj, "datasets/coco128.yaml", "yolov8n.pt", sample=12)
    print(f"[reason]   {json.dumps({k: v for k, v in live.items() if k != 'samples'})}")

    print("[reason] B: risk-scoring on labelled scenarios ...")
    scen = eval_scenarios(adj)
    print(f"[reason]   accuracy={scen['accuracy']} (llm_used {scen['llm_used']}/{scen['n']})")

    report = {"endpoint": base, "model": model, "threshold": THRESHOLD,
              "live_integration": live, "risk_scoring": scen}
    out = REPO_ROOT / "eval" / "reasoning_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[reason] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
