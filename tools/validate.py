"""Validate every registered block and the declarative pipeline DAG.

This is the script referenced by the README quickstart:

    python tools/validate.py

It performs three checks and exits non-zero if any fail:

  1. Registry   — every concrete Block subclass registered and importable.
  2. DAG build  — `config/pipeline.yaml` builds into a valid, acyclic,
                  type-checked graph (port contracts satisfied).
  3. Dry run    — one synchronous tick executes end to end and reaches the
                  declared `output` port, proving the graph is runnable.

The pipeline spec accepts either edge convention:
    - {from: "a.port", to: "b.port"}   (as written in pipeline.yaml)
    - {src: "a.port", dst: "b.port"}   (as consumed by core.graph.build)
so the declarative file and the builder stay in sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Ensure the repo root is importable when run as `python tools/validate.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engines.blocks  # noqa: E402,F401  (import registers the L3 blocks)
from core.blocks import available  # noqa: E402
from core.graph import Executor, build  # noqa: E402

PIPELINE = REPO_ROOT / "config" / "pipeline.yaml"


def load_spec(path: Path) -> dict:
    """Load a pipeline YAML into the spec shape core.graph.build expects."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    edges = []
    for e in raw.get("edges", []):
        src = e.get("src", e.get("from"))
        dst = e.get("dst", e.get("to"))
        if not src or not dst:
            raise ValueError(f"edge missing endpoints: {e!r}")
        edges.append({"src": src, "dst": dst})
    return {"nodes": raw.get("nodes", []), "edges": edges, "output": raw.get("output")}


def main() -> int:
    ok = True

    # 1. Registry -----------------------------------------------------------
    blocks = available()
    print(f"[1/3] registry: {len(blocks)} block(s) registered -> {', '.join(blocks)}")
    for required in ("capture", "clean", "detect", "validate"):
        if required not in blocks:
            print(f"      MISSING required block: {required!r}")
            ok = False

    # 2. DAG build ----------------------------------------------------------
    try:
        spec = load_spec(PIPELINE)
        graph = build(spec)  # raises GraphError on any structural/type fault
        order = graph.topological_order()
        print(f"[2/3] dag: '{PIPELINE.name}' valid; execution order -> {' -> '.join(order)}")
    except Exception as exc:  # noqa: BLE001 (report and fail, don't traceback-spam)
        print(f"[2/3] dag: FAILED to build/validate -> {type(exc).__name__}: {exc}")
        return 1

    # 3. Dry run ------------------------------------------------------------
    try:
        results = Executor(graph).run()
        out = spec.get("output")
        if out:
            node_id, _, port = out.partition(".")
            payload = results[node_id].outputs[port]
            print(f"[3/3] dry-run: reached output '{out}' -> {type(payload).__name__}")
        else:
            print(f"[3/3] dry-run: executed {len(results)} node(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"[3/3] dry-run: FAILED -> {type(exc).__name__}: {exc}")
        ok = False

    print("OK - all checks passed." if ok else "FAILED - see errors above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
