"""L2 — Graph engine: spec, build, validation, topological order.

A Graph is a DAG of block instances connected by typed edges. Before
execution the graph is validated: no cycles, every edge connects an
existing output port to an existing, type-compatible input port, and
every required input port is satisfied by exactly one edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.blocks import Block, create


@dataclass(frozen=True)
class Edge:
    """A directed connection: src_node.src_port -> dst_node.dst_port."""

    src_node: str
    src_port: str
    dst_node: str
    dst_port: str


@dataclass
class Node:
    """A graph node: a unique id bound to a block instance."""

    id: str
    block: Block


class GraphError(Exception):
    """Raised when a graph is structurally or type invalid."""


@dataclass
class Graph:
    """A validated DAG of blocks."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node_id: str, block: Block) -> Graph:
        if node_id in self.nodes:
            raise GraphError(f"duplicate node id: {node_id!r}")
        self.nodes[node_id] = Node(id=node_id, block=block)
        return self

    def add_edge(self, src: str, dst: str) -> Graph:
        """Add an edge using 'node.port' endpoints."""
        src_node, src_port = _split(src)
        dst_node, dst_port = _split(dst)
        self.edges.append(Edge(src_node, src_port, dst_node, dst_port))
        return self

    def predecessors(self, node_id: str) -> list[Edge]:
        """Return the incoming edges for a node."""
        return [e for e in self.edges if e.dst_node == node_id]

    def validate(self) -> None:
        """Run all structural and type checks; raise GraphError on failure."""
        self._check_endpoints()
        self._check_types()
        self._check_inputs_satisfied()
        self.topological_order()  # raises on cycle

    def topological_order(self) -> list[str]:
        """Return node ids in dependency order (Kahn's algorithm)."""
        indegree = {nid: 0 for nid in self.nodes}
        adjacency: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            adjacency[e.src_node].append(e.dst_node)
            indegree[e.dst_node] += 1

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for nxt in adjacency[nid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.nodes):
            raise GraphError("graph contains a cycle")
        return order

    def _check_endpoints(self) -> None:
        for e in self.edges:
            if e.src_node not in self.nodes:
                raise GraphError(f"edge references unknown node: {e.src_node!r}")
            if e.dst_node not in self.nodes:
                raise GraphError(f"edge references unknown node: {e.dst_node!r}")
            if not _has_port(self.nodes[e.src_node].block.outputs, e.src_port):
                raise GraphError(f"{e.src_node}: no output port {e.src_port!r}")
            if not _has_port(self.nodes[e.dst_node].block.inputs, e.dst_port):
                raise GraphError(f"{e.dst_node}: no input port {e.dst_port!r}")

    def _check_types(self) -> None:
        for e in self.edges:
            out = _port(self.nodes[e.src_node].block.outputs, e.src_port)
            inp = _port(self.nodes[e.dst_node].block.inputs, e.dst_port)
            if not issubclass(out.type_, inp.type_):
                raise GraphError(
                    f"type mismatch {e.src_node}.{e.src_port} -> "
                    f"{e.dst_node}.{e.dst_port}"
                )

    def _check_inputs_satisfied(self) -> None:
        for nid, node in self.nodes.items():
            wired = {e.dst_port for e in self.predecessors(nid)}
            for port in node.block.inputs:
                if port.required and port.name not in wired:
                    raise GraphError(f"{nid}: required input {port.name!r} unwired")


def _split(endpoint: str) -> tuple[str, str]:
    node, _, port = endpoint.partition(".")
    if not node or not port:
        raise GraphError(f"endpoint must be 'node.port', got {endpoint!r}")
    return node, port


def _has_port(ports: tuple, name: str) -> bool:
    return any(p.name == name for p in ports)


def _port(ports: tuple, name: str):
    for p in ports:
        if p.name == name:
            return p
    raise GraphError(f"unknown port: {name!r}")


def build(spec: dict) -> Graph:
    """Build a Graph from a declarative spec.

    Spec shape:
        {
          "nodes": [{"id": "cap", "block": "capture", "config": {...}}, ...],
          "edges": [{"src": "cap.frame", "dst": "clean.frame"}, ...],
        }
    """
    graph = Graph()
    for n in spec.get("nodes", []):
        block = create(n["block"], **n.get("config", {}))
        graph.add_node(n["id"], block)
    for e in spec.get("edges", []):
        graph.add_edge(e["src"], e["dst"])
    graph.validate()
    return graph
