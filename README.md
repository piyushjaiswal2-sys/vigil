<div align="center">

# VIGIL
### Visual Intelligence Graph & Inference Layer

**A block-graph, real-time vision platform with a swappable free-tier LLM reasoning core.**

</div>

---

## What is VIGIL?

VIGIL turns a live camera feed into risk-scored, human-readable events by running frames through a visual, validated block-graph: capture, clean, detect, validate, reason, report, store. It is not one big model; it is a pipeline of small, inspectable decisions.

The reasoning layer is provider-agnostic. VIGIL talks to a single OpenAI-compatible /v1 endpoint served by **freellmapi** (https://github.com/tashfeenahmed/freellmapi), which stacks the free tiers of 18 LLM providers behind one bearer token with smart routing and automatic failover. Swap Groq for Gemini for a local vLLM without touching VIGIL code.

## Why it exists (reference lineage)

| Reference | What VIGIL borrows |
|-----------|--------------------|
| roboflow/inference | Visual Workflow editor, model-chaining blocks, live-video InferencePipeline |
| pysource-com/VisoNode | No-code node graph: camera to YOLO to live output |
| SharpAI/DeepCamera | Local VLM analysis + agentic camera reasoning + alerting |
| getstream/Vision-Agents | Detector + reasoning-LLM split in a low-latency processor pipeline |

## The five layers (UniVision learning map)

1. Computational core: variables, logic, state, pipelines.
2. Visual programming: blocks, ports, connections, DAG validation, execution order.
3. Computer vision: frames, preprocessing, YOLO detection, OCR, tracking, anomaly.
4. Full-stack: FastAPI, WebSocket streaming, dashboard, queue, storage, metrics.
5. Agentic AI: LLM tools, ReAct adjudication, RAG/FAISS, safety and human oversight.

## Architecture

<pre>
 CAMERAS -> [S0 Ingest] -> [S1 Preprocess] -> [S2 Detect/OCR] -> [S3 Anomaly]
 RTSP/USB    ring buffers    pHash + motion     YOLOv8 + OCR       rule+stat gate
                                                                        |
         dashboard <- [S5 Dispatch] <- [S4 Reason] <---------------------+
         WebSocket      WS + Postgres    freellmapi /v1
                                         (ReAct adjudicator + RAG/FAISS)
</pre>

## Quickstart

<pre>
cp .env.example .env          # set VIGIL_LLM_TOKEN=freellmapi-...
docker compose up -d          # api :8000, web :5173, freellmapi :8080, grafana :3000
python tools/validate.py      # confirm all components + example DAGs are valid
open http://localhost:5173    # drag blocks, wire a graph, hit Run
</pre>

Run a workflow headless:

<pre>
curl -X POST localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"workflow":"perimeter_safety","source":"rtsp://cam-1/stream"}'
</pre>

## The AI core (freellmapi)

VIGIL never hardcodes a model. See config/llm.yaml:

- base_url: http://freellmapi:8080/v1  (single OpenAI-compatible endpoint)
- one freellmapi-... bearer token for chat, embeddings, audio and images
- router picks the highest-priority healthy, in-budget model; sticky sessions for 30 min
- automatic 429/5xx failover across the fallback chain; embeddings failover locked to same dimension

Adjudicator, OCR post-reasoning, and RAG all call this one endpoint.

## Repository layout

- core/blocks, core/graph  -> layers 1 and 2
- vision/                  -> layer 3
- agent/                   -> layer 5 (freellmapi client, adjudicator, tools, rag, safety)
- server/                  -> layer 4 API + WebSocket + queue + db + metrics
- web/                     -> layer 4 React Flow dashboard
- config/, tests/, tools/, docs/

## Safety (non-negotiable)

Every AI event carries evidence, timestamp, source and confidence. High-stakes actions require human approval. Prompt-injection defenses and an append-only audit log guard the agent tool layer.

## Status

Concept repository. Interfaces and stubs are complete and validated; model weights and provider keys are user-supplied.

## License

Apache-2.0.
