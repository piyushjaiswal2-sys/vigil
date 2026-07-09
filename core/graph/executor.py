"""L2 — Graph executor.

Executes a validated Graph in topological order. Each node's inputs are
gathered from upstream node outputs along wired edges, validated against
the block's declared input ports, then passed to `block.run`.

This reference executor is synchronous and single-pass (one tick over the
DAG). A streaming/back-pressured executor can build on the same contract.
"""
from __future__ import annotations

import logging
from typing import Any

from core.blocks import BlockResult
from core.graph.graph import Graph

logger = logging.getLogger("vigil.graph.executor")


class Executor:
    """Synchronous, single-pass DAG executor."""

    def __init__(self, graph: Graph) -> None:
        graph.validate()
        self.graph = graph
        self.order = graph.topological_order()

    def run(self, seed: dict[str, dict[str, Any]] | None = None) -> dict[str, BlockResult]:
        """Run one pass over the DAG.

        `seed` optionally provides inputs for source nodes keyed by node id,
        e.g. {"cap": {"source": "rtsp://..."}}. Returns each node's result.
        """
        seed = seed or {}
        results: dict[str, BlockResult] = {}

        for node_id in self.order:
            node = self.graph.nodes[node_id]
            inputs = self._gather_inputs(node_id, results)
            inputs.update(seed.get(node_id, {}))
            node.block.validate_inputs(inputs)
            logger.debug("running node %s (%s)", node_id, node.block.name)
            results[node_id] = node.block.run(inputs)

        return results

    def _gather_inputs(
        self, node_id: str, results: dict[str, BlockResult]
    ) -> dict[str, Any]:
        """Collect input values for a node from upstream results."""
        inputs: dict[str, Any] = {}
        for edge in self.graph.predecessors(node_id):
            upstream = results.get(edge.src_node)
            if upstream is None:
                continue
            if edge.src_port not in upstream.outputs:
                raise KeyError(
                    f"{edge.src_node}: missing output {edge.src_port!r}"
                )
            inputs[edge.dst_port] = upstream.outputs[edge.src_port]
        return inputs


def run_graph(
    graph: Graph, seed: dict[str, dict[str, Any]] | None = None
) -> dict[str, BlockResult]:
    """Convenience: build an Executor and run a single pass."""
    return Executor(graph).run(seed)
