<div align="center">

# ⚡ VIGIL

### Visual Intelligence Graph &amp; Inference Layer

**A block-graph, real-time computer-vision platform with a single, swappable free-tier LLM reasoning core — [freellmapi](https://github.com/tashfeenahmed/freellmapi).**

_Turn any camera stream into risk-scored, human-readable, auditable events — through a pipeline of small, inspectable decisions instead of one opaque model._

<br/>

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](./server)
[![CI](https://img.shields.io/badge/CI-pytest_matrix_%2B_ruff-2088FF?logo=githubactions&logoColor=white)](./.github/workflows/ci.yml)
[![Reasoning](https://img.shields.io/badge/LLM_core-freellmapi_only-8A2BE2?logo=openai&logoColor=white)](https://github.com/tashfeenahmed/freellmapi)
[![Status](https://img.shields.io/badge/status-concept_reference-orange.svg)](#-project-status)
[![Layers](https://img.shields.io/badge/architecture-5_layers_(L1%E2%86%92L5)-teal.svg)](#-the-five-layers-univision-learning-map)

</div>

---

## 📍 Table of Contents

- [What is VIGIL?](#-what-is-vigil)
- [Reference Lineage](#-reference-lineage-standing-on-four-shoulders)
- [The Five Layers](#-the-five-layers-univision-learning-map)
- [System Architecture](#-system-architecture)
- [The Pipeline (S0–S5)](#-the-pipeline-s0s5)
- [The AI Core — freellmapi only](#-the-ai-core--freellmapi-only)
- [Data Contracts](#-data-contracts)
- [Quickstart](#-quickstart)
- [Repository Layout](#-repository-layout)
- [Safety Model](#-safety-model-non-negotiable)
- [Project Status](#-project-status)
- [License & Attribution](#-license--attribution)

---

## 🧭 What is VIGIL?

VIGIL runs a live camera feed through a **validated block-graph** — `capture → clean → detect → validate → reason → report → store` — emitting a bounded, evidence-carrying `RiskEvent` at the end of every cycle.

The design philosophy is deliberate:

> **Not one big model. A pipeline of small, inspectable, swappable decisions.**

Each stage is a typed **Block** with declared input/output **ports**, wired into a **directed acyclic graph (DAG)** that is validated *before* it ever runs. The reasoning layer is intentionally centralized: **every LLM call in VIGIL goes to one place — the [freellmapi](https://github.com/tashfeenahmed/freellmapi) endpoint.** VIGIL never loads model weights, never runs a local model, and never hardcodes a provider. It speaks a single OpenAI-compatible `/v1` contract, and freellmapi handles routing, failover and provider selection behind that one door.

```mermaid
mindmap
  root((VIGIL))
    L1 Core
      Blocks
      Registry
      State
    L2 Graph
      Ports
      DAG validation
      Executor
    L3 Vision
      Capture
      Clean
      Detect / OCR
      Validate
    L4 Full-stack
      FastAPI
      WebSocket
      Dashboard
      Audit log
    L5 Agent
      freellmapi client
      ReAct adjudicator
      RAG / FAISS
      Safety
```

---

## 🧬 Reference Lineage: Standing on Four Shoulders

VIGIL is a **synthesis** — it borrows one hard-won idea from each of four production-grade open-source projects and fuses them into a single coherent stack. The concept cards below link to each upstream repository.

<table>
  <tr>
    <td width="50%" valign="top" align="center">
      <a href="https://github.com/roboflow/inference">
        <img src="https://opengraph.githubassets.com/1/roboflow/inference" width="420" alt="roboflow/inference" />
      </a>
      <br/><br/>
      <b>What VIGIL borrows:</b> the <i>Visual Workflow</i> editor, model-chaining <b>blocks</b>, and the live-video <code>InferencePipeline</code> abstraction — the idea that CV should be <i>composed, not coded</i>.
    </td>
    <td width="50%" valign="top" align="center">
      <a href="https://github.com/pysource-com/VisoNode">
        <img src="https://opengraph.githubassets.com/1/pysource-com/VisoNode" width="420" alt="pysource-com/VisoNode" />
      </a>
      <br/><br/>
      <b>What VIGIL borrows:</b> the <i>no-code node graph</i> UX — wire a <b>camera → YOLO → live output</b> visually, with zero boilerplate, so a graph is something you <i>draw</i>.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" align="center">
      <a href="https://github.com/SharpAI/DeepCamera">
        <img src="https://opengraph.githubassets.com/1/SharpAI/DeepCamera" width="420" alt="SharpAI/DeepCamera" />
      </a>
      <br/><br/>
      <b>What VIGIL borrows:</b> <i>agentic camera reasoning</i> + edge alerting — the notion that a camera can <i>reason about a scene</i>, not just detect boxes. (In VIGIL that reasoning is delegated to freellmapi, never a local model.)
    </td>
    <td width="50%" valign="top" align="center">
      <a href="https://github.com/GetStream/Vision-Agents">
        <img src="https://opengraph.githubassets.com/1/GetStream/Vision-Agents" width="420" alt="GetStream/Vision-Agents" />
      </a>
      <br/><br/>
      <b>What VIGIL borrows:</b> the clean <b>detector ↔ reasoning-LLM split</b> inside a low-latency processor pipeline — <i>fast perception, slow deliberation</i>, cleanly separated.
    </td>
  </tr>
</table>

> ℹ️ **Attribution:** These upstream projects are independent works by their respective authors under their own licenses. VIGIL re-implements *concepts and interfaces* for learning purposes; it vendors none of their code, weights, or models.

---

## 📚 The Five Layers (UniVision Learning Map)

VIGIL is structured as five stacked layers. Each maps 1:1 to a directory and to a rung on the UniVision learning ladder — read the codebase bottom-up and you learn the whole stack.

| Layer | Domain | Responsibility | Lives in |
|:-----:|--------|----------------|----------|
| **L1** | Computational core | Variables, logic, state, block/registry primitives, pipelines | `core/` |
| **L2** | Visual programming | Blocks, ports, connections, DAG validation, execution order | `core/graph/` |
| **L3** | Computer vision | Frames, preprocessing, YOLO detection, OCR, tracking, anomaly | `engines/` |
| **L4** | Full-stack | FastAPI, WebSocket streaming, dashboard, queue, storage, metrics | `server/`, `frontend/` |
| **L5** | Agentic AI | freellmapi client, ReAct adjudication, RAG/FAISS, safety & human oversight | `agent/` |

```mermaid
flowchart LR
  L1["L1 · Core<br/>blocks · registry · state"]:::l1 --> L2["L2 · Graph<br/>ports · DAG · executor"]:::l2
  L2 --> L3["L3 · Vision<br/>capture · detect · validate"]:::l3
  L3 --> L4["L4 · Full-stack<br/>API · WebSocket · dashboard"]:::l4
  L4 --> L5["L5 · Agent<br/>freellmapi · ReAct · RAG · safety"]:::l5
  classDef l1 fill:#0d3b66,stroke:#4dabf7,color:#fff;
  classDef l2 fill:#0d5c63,stroke:#3bc9db,color:#fff;
  classDef l3 fill:#0d661f,stroke:#69db7c,color:#fff;
  classDef l4 fill:#665c0d,stroke:#ffd43b,color:#fff;
  classDef l5 fill:#3d0d66,stroke:#b197fc,color:#fff;
```

---

## 🏗️ System Architecture

```mermaid
flowchart TD
  subgraph L3G["L3 · Vision Engines"]
    CAM(["📹 Cameras<br/>RTSP / USB"]) --> S0["S0 Capture<br/>ring buffers"]
    S0 --> S1["S1 Clean<br/>pHash + motion"]
    S1 --> S2["S2 Detect / OCR<br/>YOLOv8 + OCR"]
    S2 --> S3["S3 Validate<br/>rule + stat gate"]
  end
  subgraph L5G["L5 · Agent (reasoning)"]
    S3 --> ADJ["S4 Reason<br/>Adjudicator (ReAct)"]
    ADJ <--> LLM(["freellmapi /v1<br/>the ONLY LLM door"])
    ADJ <--> RAG[("RAG / FAISS<br/>memory")]
    ADJ --> GUARD{"Safety<br/>guardrails"}
    GUARD --> S5["S5 Dispatch<br/>RiskEvent"]
  end
  subgraph L4G["L4 · Presentation"]
    S5 --> WS["WebSocket<br/>stream"]
    S5 --> DB[("Postgres<br/>audit log")]
    WS --> UI(["🖥️ Dashboard"])
  end
  classDef vision fill:#0d3b66,stroke:#4dabf7,color:#fff;
  classDef agent fill:#3d0d66,stroke:#b197fc,color:#fff;
  classDef io fill:#0d661f,stroke:#69db7c,color:#fff;
  class S0,S1,S2,S3 vision;
  class ADJ,GUARD,S5,RAG agent;
  class CAM,LLM,UI,DB,WS io;
```

> The graph is **validated before execution**: every edge is a typed port contract, so a malformed pipeline fails at build time — never mid-stream on a live camera. Note there is exactly **one** LLM node in the entire graph: `freellmapi /v1`.

---

## 🔁 The Pipeline (S0–S5)

One pass over a single frame, end to end. Perception is fast and local; deliberation is a single remote call to freellmapi.

```mermaid
sequenceDiagram
  autonumber
  participant Cam as 📹 Camera
  participant Eng as Vision Engines (L3)
  participant Adj as Adjudicator (L5)
  participant LLM as freellmapi /v1
  participant API as Server (L4)
  participant UI as Dashboard
  Cam->>Eng: raw frame
  Note over Eng: S0 capture → S1 clean<br/>S2 detect → S3 validate
  Eng->>Adj: ValidatedDetections
  Adj->>Adj: sanitize context (injection defense)
  Adj->>LLM: score detections (ReAct prompt)
  LLM-->>Adj: draft risk + rationale
  Adj->>Adj: enforce_output (clamp risk, bound summary)
  Adj->>API: RiskEvent (evidence + confidence)
  API->>UI: stream over WebSocket
  API->>API: append to append-only audit log
```

| Stage | Block | Guarantee |
|:-----:|-------|-----------|
| **S0** | `CaptureBlock` | Monotonic frame index; deterministic stub when no camera |
| **S1** | `CleanBlock` | Normalized geometry + color space for inference |
| **S2** | `DetectBlock` | Empty-but-valid output when no detector backend is present |
| **S3** | `ValidateBlock` | Drops low-confidence / malformed boxes; reports `dropped` count |
| **S4** | `Adjudicator` | Reasoning via **freellmapi only**; deterministic heuristic fallback when offline |
| **S5** | `Dispatch` | Bounded `RiskEvent` — `0.0 ≤ risk ≤ 1.0`, summary ≤ 280 chars |

---

## 🧠 The AI Core — freellmapi only

> 🔒 **Hard rule.** VIGIL has exactly **one** reasoning core: **[freellmapi](https://github.com/tashfeenahmed/freellmapi)**. There are **no local models**, **no bundled open-source weights**, and **no hardcoded provider names** anywhere in the codebase. Every chat, embedding, audio and image call flows through this single OpenAI-compatible `/v1` endpoint.

```mermaid
flowchart LR
  subgraph VIGIL["VIGIL callers"]
    A["Adjudicator (S4)"]
    B["OCR post-reasoning"]
    C["RAG embeddings"]
  end
  A --> D(["freellmapi /v1"])
  B --> D
  C --> D
  D -->|routes to healthy,<br/>in-budget provider| E["18 free-tier<br/>LLM providers"]
  classDef core fill:#3d0d66,stroke:#b197fc,color:#fff;
  classDef caller fill:#0d3b66,stroke:#4dabf7,color:#fff;
  class D,E core;
  class A,B,C caller;
```

**Why a single door?**

- **Base URL:** `http://freellmapi:8080/v1` — one endpoint for chat, embeddings, audio and images.
- **Smart routing:** freellmapi picks the highest-priority *healthy, in-budget* provider; sticky sessions for 30 min.
- **Automatic failover:** transparent `429`/`5xx` fallback across the provider chain; embedding failover is locked to the same vector dimension so FAISS never breaks.
- **Zero coupling:** VIGIL code names no provider and loads no weights — swapping providers is a freellmapi config change, invisible to VIGIL.

```yaml
# config/llm.yaml (illustrative) — the ONLY place a model backend is referenced
base_url: http://freellmapi:8080/v1   # freellmapi is the sole LLM core
token: ${VIGIL_LLM_TOKEN}             # freellmapi-...
routing: priority-healthy-in-budget
sticky_session_minutes: 30
# note: no model name is hardcoded — freellmapi decides at call time
```

---

## 📦 Data Contracts

Every block speaks in typed dataclasses, so the graph is verifiable end to end. The two central contracts:

```python
@dataclass
class ValidatedDetections:
    frame_index: int          # monotonic, from S0
    detections: list[Box]     # class, bbox, confidence
    dropped: int              # boxes removed at S3
    source: str               # camera id / stream url

@dataclass
class RiskEvent:
    risk: float               # clamped 0.0 .. 1.0
    label: str                # e.g. "perimeter_breach"
    summary: str              # <= 280 chars, bounded
    evidence: list[Box]       # boxes that drove the decision
    confidence: float         # 0.0 .. 1.0
    timestamp: str            # ISO-8601, UTC
    source: str               # provenance for audit
```

---

## 🚀 Quickstart

```bash
# 1. Configure
cp .env.example .env          # set VIGIL_LLM_TOKEN=freellmapi-...

# 2. Bring the stack up
docker compose up -d          # api :8000 · web :5173 · freellmapi :8080 · grafana :3000

# 3. Validate every block + example DAG
python tools/validate.py

# 4. Open the dashboard — drag blocks, wire a graph, hit Run
open http://localhost:5173
```

Run a workflow **headless**:

```bash
curl -X POST localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"workflow":"perimeter_safety","source":"rtsp://cam-1/stream"}'
```

Local development (no Docker):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                       # run the test suite
uvicorn server.app:create_app --factory --reload  # serve the API
```

---

## 📂 Repository Layout

```text
vigil/
├─ core/        # L1 computational core: blocks, registry, state
│  └─ graph/    # L2 DAG wiring, port contracts, executor
├─ engines/     # L3 vision: capture · clean · detect · validate + shared types
├─ agent/       # L5 freellmapi client · adjudicator · safety · rag · tools
├─ server/      # L4 FastAPI app · routes · schemas · WebSocket · metrics
├─ frontend/    # L4 dashboard: index.html · styles.css · app.js
├─ config/      # settings + pipeline.yaml (declarative S0–S3 DAG) + llm.yaml
├─ tests/       # pytest suites (test_engines, test_agent, ...)
├─ .github/     # CI: pytest matrix + ruff lint
├─ ARCHITECTURE.md  CONTRIBUTING.md  docker-compose.yml  pyproject.toml  Makefile
└─ README.md
```

---

## 🛡️ Safety Model (Non-Negotiable)

The agent tool layer is treated as an **untrusted boundary in, bounded contract out**:

- **Evidence-first:** every AI event carries evidence, timestamp, source and confidence.
- **Human-in-the-loop:** high-stakes actions require explicit human approval.
- **Injection defense:** free text reaching the reasoning core passes through `agent.safety.sanitize_text`.
- **Output contract:** freellmapi output passes `enforce_output` — risk clamped to `[0,1]`, summary bounded, label defaulted.
- **Auditability:** an append-only audit log guards the tool layer.
- **Single trust surface:** because reasoning is centralized in freellmapi, there is exactly one outbound AI boundary to secure — no local model to sandbox, patch, or supply-chain audit.

---

## 📊 Project Status

> **Concept reference repository.** Interfaces, stubs and the five-layer architecture are complete and validated; provider keys are user-supplied to freellmapi. Every block runs a deterministic stub path so the graph stays importable, testable, and GPU-free out of the box.

| Phase | Layer | State |
|:-----:|-------|:-----:|
| 1 | Skeleton | ✅ |
| 2 | L1 Core | ✅ |
| 3 | L3 Engines | ✅ |
| 4 | L5 Agent (freellmapi) | ✅ |
| 5 | L4 Server | ✅ |
| 6 | L4 Frontend | ✅ |
| 7 | Config / CI | ✅ |
| 8 | Tests | ✅ |

---

## 📜 License & Attribution

VIGIL is released under the **[Apache-2.0](./LICENSE)** license. It is an educational synthesis inspired by — and crediting — four upstream projects: [roboflow/inference](https://github.com/roboflow/inference) · [pysource-com/VisoNode](https://github.com/pysource-com/VisoNode) · [SharpAI/DeepCamera](https://github.com/SharpAI/DeepCamera) · [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) — with **all reasoning powered exclusively by [tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi)**. All trademarks and code belong to their respective owners.

<div align="center">

**Built as a layered learning map · read it bottom-up (L1 → L5) and you learn the whole stack.**

</div>
