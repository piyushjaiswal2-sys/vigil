# VIGIL

Visual Intelligence Graph & Inference Layer: a block-graph, real-time
computer-vision platform with a single, swappable free-tier LLM reasoning core
([freellmapi](https://github.com/tashfeenahmed/freellmapi)).

VIGIL runs a camera feed through a validated block-graph
(`capture → clean → detect → validate → reason → report`) and emits a bounded,
evidence-carrying `RiskEvent` at the end of every cycle. The design goal is a
pipeline of small, inspectable, swappable stages rather than one opaque model.
Perception is fast and local; reasoning is a single remote call routed through
one OpenAI-compatible `/v1` endpoint.

License: Apache-2.0. Python 3.10–3.12.

## Contents

- [Reference lineage](#reference-lineage)
- [Layers](#layers)
- [Pipeline (S0–S5)](#pipeline-s0s5)
- [Reasoning core](#reasoning-core)
- [Data contracts](#data-contracts)
- [Quickstart](#quickstart)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)
- [Safety model](#safety-model)
- [Status](#status)
- [License and attribution](#license-and-attribution)

## Reference lineage

VIGIL re-implements one idea from each of four open-source projects for learning
purposes. It vendors none of their code, weights, or models.

| Project | Idea borrowed |
|---|---|
| [roboflow/inference](https://github.com/roboflow/inference) | Visual-workflow editor, model-chaining blocks, and the live-video `InferencePipeline` abstraction; CV is composed, not coded. |
| [pysource-com/VisoNode](https://github.com/pysource-com/VisoNode) | No-code node-graph UX: wire camera → detector → output visually. |
| [SharpAI/DeepCamera](https://github.com/SharpAI/DeepCamera) | Agentic camera reasoning and edge alerting (delegated to freellmapi here, not a local model). |
| [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) | The detector ↔ reasoning-LLM split inside a low-latency processor pipeline. |

## Layers

The codebase is organized as five layers, each mapping to a directory.

| Layer | Domain | Responsibility | Location |
|---|---|---|---|
| L1 | Computational core | Blocks, registry, state primitives | `core/` |
| L2 | Visual programming | Ports, connections, DAG validation, execution order | `core/graph/` |
| L3 | Computer vision | Frames, preprocessing, detection, validation | `engines/` |
| L4 | Full-stack | FastAPI, WebSocket, dashboard, storage, metrics | `server/`, `frontend/` |
| L5 | Agentic AI | freellmapi client, adjudication, RAG, safety | `agent/` |

The graph is validated before execution: every edge is a typed port contract,
so a malformed pipeline fails at build time rather than mid-stream. There is
exactly one LLM node in the graph: the freellmapi endpoint.

## Pipeline (S0–S5)

One pass over a single frame, end to end.

| Stage | Block | Guarantee |
|---|---|---|
| S0 | `CaptureBlock` | Monotonic frame index; deterministic stub when no camera. |
| S1 | `CleanBlock` | Normalized geometry and color space for inference. |
| S2 | `DetectBlock` | Runs an injected detector; empty-but-valid output when none is present. |
| S3 | `ValidateBlock` | Drops low-confidence / malformed boxes; reports a `dropped` count. |
| S4 | `Adjudicator` | Reasoning via freellmapi; deterministic heuristic fallback when offline. |
| S5 | Dispatch | Bounded `RiskEvent`: `0.0 ≤ risk ≤ 1.0`, summary ≤ 280 chars. |

## Reasoning core

Every LLM call routes through a single OpenAI-compatible `/v1` endpoint.
VIGIL loads no model weights and hardcodes no provider name; provider routing,
failover, and selection are handled by freellmapi behind that endpoint.

```yaml
# config/llm.yaml (illustrative)
base_url: http://freellmapi:8080/v1
token: ${VIGIL_LLM_TOKEN}
routing: priority-healthy-in-budget
sticky_session_minutes: 30
# no model name is hardcoded; freellmapi decides at call time
```

When `base_url` is unset, the adjudicator degrades to a deterministic heuristic
risk score so the pipeline never hard-fails on the LLM hop.

## Data contracts

Every block speaks in typed dataclasses, so the graph is verifiable end to end.

```python
@dataclass
class ValidatedDetections:
    frame_index: int
    items: list[Detection]    # class, bbox, confidence
    dropped: int              # boxes removed at S3

@dataclass
class RiskEvent:
    frame_index: int
    risk: float               # clamped 0.0 .. 1.0
    label: str
    summary: str              # bounded (<= 280 chars)
    detections: list[Detection]
    meta: dict
```

## Quickstart

Docker:

```bash
cp .env.example .env          # set VIGIL_LLM_TOKEN=freellmapi-...
docker compose up -d          # api :8000, web :5173, freellmapi :8080, grafana :3000
python tools/validate.py      # validate every block + the example DAG
```

Local (no Docker):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pytest pyyaml ultralytics               # test + detector deps
pytest -q                                           # run the test suite
python tools/validate.py                            # validate blocks + DAG
uvicorn server.app:create_app --factory --reload    # serve the API
```

Headless analyze call:

```bash
curl -X POST localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"workflow":"perimeter_safety","source":"rtsp://cam-1/stream"}'
```

## Evaluation

The detection stage was evaluated by wiring a real YOLOv8 backend into
`DetectBlock` and running the `DetectBlock → ValidateBlock` path over a public
dataset; these numbers are reproducible offline with no API keys (raw output in
[`eval/results.json`](./eval/results.json)). The reasoning stage was then
evaluated against a live LLM endpoint (see [Reasoning layer](#reasoning-layer-s4)).

### Setup

| | |
|---|---|
| Hardware | Intel Core i5-1155G7 @ 2.50 GHz, CPU only (no GPU) |
| Runtime | Python 3.13.5, torch 2.13.0+cpu, ultralytics 8.4.91 |
| Detector | YOLOv8n (3.15 M params, 8.7 GFLOPs) |
| Dataset | COCO128 (128 images, 929 instances, 71 of 80 classes present) |
| Operating point | conf = 0.25, imgsz = 640, TP match at IoU ≥ 0.5 |

### Detection results

Metrics were computed over VIGIL's own pipeline output and cross-checked against
ultralytics' official `val`. The two agree closely, which validates the harness;
the mAP gap reflects the AP-interpolation convention (VOC all-point over present
classes here, versus COCO 101-point over all 80 classes in ultralytics).

| Metric | VIGIL pipeline (S2→S3) | ultralytics `val` |
|---|:---:|:---:|
| Precision | 0.724 | 0.678 |
| Recall | 0.500 | 0.500 |
| F1 | 0.592 | — |
| mAP@0.5 | 0.488 | 0.468 |
| mAP@0.5:0.95 | — | 0.364 |

Counts over 128 images: TP 465, FP 177, FN 464 (929 ground-truth instances).
Throughput on CPU (single stream, steady-state): 80.05 ms/frame, 12.49 FPS.

### Test suite

On a clean clone the repository shipped with 4 of 14 tests failing. Two bugs were
fixed: `Block.__init__` discarded the `config` dict (frame size and the injected
detector were ignored), and the adjudicator's heuristic fallback constructed a
`RiskEvent` without the required `frame_index` (raising `TypeError` on every
no-LLM decision). The missing `tools/validate.py` (referenced by the quickstart)
and a `config/pipeline.yaml` loader were also added.

| Check | Before | After |
|---|:---:|:---:|
| `pytest` | 10 / 14 | 14 / 14 |
| `python tools/validate.py` | missing | passes |

### Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; else: source .venv/bin/activate
pip install pytest pyyaml ultralytics               # ultralytics pulls torch + opencv (CPU)
pytest -q
python tools/validate.py
python tools/evaluate.py --model yolov8n.pt --data coco128.yaml   # downloads COCO128
# quick smoke: python tools/evaluate.py --limit 32 --skip-official

# reasoning layer (needs a key in .env: FREELLMAPI_BASE_URL / _API_KEY / _MODEL)
python tools/eval_reasoning.py
```

### Reasoning layer (S4)

The reasoning layer was evaluated against a live OpenAI-compatible endpoint —
Groq `llama-3.3-70b-versatile` used directly as the provider. A self-hosted
freellmapi gateway exposes the identical `/v1` contract, so this is a
config-only difference (`FREELLMAPI_BASE_URL` / `FREELLMAPI_API_KEY` /
`FREELLMAPI_MODEL`). Raw output: [`eval/reasoning_results.json`](./eval/reasoning_results.json).

Live integration: YOLOv8 detections from 12 COCO128 frames, adjudicated by the LLM.

| Metric | Value |
|---|:---:|
| LLM success rate | 12 / 12 (100%) |
| Mean reasoning latency | 848 ms / call |
| Heuristic fallbacks | 0 |

Risk-scoring: 10 labelled scenarios (5 clearly risky, 5 clearly safe), scored
against a 0.5 risk threshold.

| Metric | Value |
|---|:---:|
| Accuracy | 10 / 10 (100%) |
| Risky scenarios | risk 0.80–0.90 |
| Safe scenarios | risk 0.00 |

Sample outputs: person holding a knife in a store → risk 0.90, label "Threat";
truck on an active runway → risk 0.90, label "Incursion"; cat on a sofa → risk
0.00, label "Pet"; person + umbrella on a street → risk 0.45, label "Person".

Making this path work required two fixes in `agent/freellmapi_client.py`, both
found during evaluation:

- The client sent no `User-Agent`, so Cloudflare-fronted gateways (Groq, and
  freellmapi.co itself) rejected every call with a 1010 block.
- `_parse` ran `json.loads` on the raw reply, so any model that wrapped its JSON
  in a Markdown code fence (most instruct models do) silently fell back to the
  heuristic. Parsing now strips fences and extracts the JSON object.

## Repository layout

```text
vigil/
├─ core/        # L1 core: blocks, registry, state
│  └─ graph/    # L2 DAG wiring, port contracts, executor
├─ engines/     # L3 vision: capture, clean, detect, validate, detectors, types
├─ agent/       # L5 freellmapi client, adjudicator, safety, provider
├─ server/      # L4 FastAPI app, routes, schemas
├─ frontend/    # L4 dashboard
├─ config/      # settings + pipeline.yaml (S0–S3 DAG)
├─ tools/       # validate.py, evaluate.py, eval_reasoning.py
├─ tests/       # pytest suites
├─ eval/        # evaluation output (results.json, reasoning_results.json)
└─ .github/     # CI: pytest + ruff
```

## Safety model

The agent tool layer is treated as untrusted input in, bounded contract out:

- Every event carries evidence, timestamp, source, and confidence.
- High-stakes actions require explicit human approval.
- Free text reaching the reasoning core passes through `agent.safety.sanitize_text`.
- LLM output passes `enforce_output`: risk clamped to `[0, 1]`, summary bounded, label defaulted.
- An append-only audit log guards the tool layer.
- Reasoning is centralized in freellmapi, so there is one outbound AI boundary to secure.

## Status

Concept reference repository. The five-layer architecture, the S0–S3 vision
front-half, and the agent/safety core are implemented and tested. The detector
backend and provider keys are user-supplied; every block has a deterministic
stub path so the graph stays importable and testable without a GPU.

| Area | State |
|---|:---:|
| L1 core / L2 graph | Done |
| L3 engines (S0–S3) | Done |
| L5 agent (freellmapi + safety) | Done |
| L4 server / frontend | Done |
| Config / CI | Done |
| Tests | 14 / 14 |
| Detection eval (YOLOv8 on COCO128) | Done, see [Evaluation](#evaluation) |
| Reasoning eval (live LLM) | Done, see [Evaluation](#evaluation) |

## License and attribution

Apache-2.0. An educational synthesis crediting four upstream projects
([roboflow/inference](https://github.com/roboflow/inference),
[pysource-com/VisoNode](https://github.com/pysource-com/VisoNode),
[SharpAI/DeepCamera](https://github.com/SharpAI/DeepCamera),
[GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents)), with
reasoning powered by [freellmapi](https://github.com/tashfeenahmed/freellmapi).
All trademarks and code belong to their respective owners.
