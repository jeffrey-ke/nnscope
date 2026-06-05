"""nnscope — collapsible HTML trees and hook-execution tracing for torch.nn.Module."""

from nnscope.tree import ModuleNode, build_tree, iter_nodes
from nnscope.html import render_html, write_html
from nnscope.trace import (
    Event,
    PairCheck,
    PairingReport,
    TraceResult,
    assert_gligen_pairing,
    trace_forward,
)

__all__ = [
    "ModuleNode",
    "build_tree",
    "iter_nodes",
    "render_html",
    "write_html",
    "Event",
    "PairCheck",
    "PairingReport",
    "TraceResult",
    "assert_gligen_pairing",
    "trace_forward",
]
