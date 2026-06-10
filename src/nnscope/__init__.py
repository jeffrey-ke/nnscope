"""nnscope — collapsible HTML trees and hook-execution tracing for torch.nn.Module."""

from nnscope.tree import ModuleNode, apply_grad_report, build_tree, iter_nodes
from nnscope.html import render_html, write_html
from nnscope.grad import (
    GradReport,
    GradState,
    ModuleGrad,
    ParamGrad,
    StaticAudit,
    find_modules,
    probe_gradients,
    static_grad_audit,
    sum_floating,
)
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
    "apply_grad_report",
    "build_tree",
    "iter_nodes",
    "render_html",
    "write_html",
    "GradReport",
    "GradState",
    "ModuleGrad",
    "ParamGrad",
    "StaticAudit",
    "find_modules",
    "probe_gradients",
    "static_grad_audit",
    "sum_floating",
    "Event",
    "PairCheck",
    "PairingReport",
    "TraceResult",
    "assert_gligen_pairing",
    "trace_forward",
]
