"""Smoke test for nnscope: tree capture, HTML rendering, execution tracing.

CPU-only, no weights, no GPU. Run from the repo root:

    env -u PYTHONPATH uv run python tests/smoke_nnscope.py
"""

import doctest

import torch
import torch.nn as nn

import nnscope.trace
from nnscope import (
    assert_gligen_pairing,
    build_tree,
    iter_nodes,
    render_html,
    trace_forward,
)


# ---------------------------------------------------------------- tree.py
class ToyHost(nn.Module):
    """Known tree: a frozen 'backbone' (mimics frozen SAM) + trainable inserts."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.backbone.requires_grad_(False)
        # shape-preserving at each hooked layer's input width, like a real
        # GligenBlock preserves its attn block's hidden dim
        self.inserts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(8, 8)])

    def forward(self, x):
        handles = [
            self.backbone[i].register_forward_pre_hook(
                lambda m, args, _g=g: (_g(args[0]),)
            )
            # hook insert 0 onto backbone.0 and insert 1 onto backbone.2
            for i, g in zip((0, 2), self.inserts)
        ]
        try:
            return self.backbone(x)
        finally:
            for h in handles:
                h.remove()


def test_tree():
    model = ToyHost()
    root = build_tree(model)
    by_path = {n.path: n for n in iter_nodes(root)}

    assert root.total_params == sum(p.numel() for p in model.parameters()), \
        f"root rollup {root.total_params} != model param count"
    backbone = by_path["backbone"]
    assert backbone.total_trainable == 0, "frozen subtree must report 0 trainable"
    assert backbone.total_params > 0
    inserts = by_path["inserts"]
    assert inserts.total_trainable == inserts.total_params > 0, \
        "insert subtree must be fully trainable"
    assert backbone.own_params == 0, "containers own no params directly"
    assert [c.name for c in root.children] == [n for n, _ in model.named_children()], \
        "child order must match named_children()"
    assert by_path["backbone.0"].class_name == "Linear"
    assert "in_features=4" in by_path["backbone.0"].extra_repr
    print(f"tree: OK ({len(by_path)} nodes, root={root.total_params} params, "
          f"{root.total_trainable} trainable)")


# ---------------------------------------------------------------- html.py
def test_html():
    model = ToyHost()
    root = build_tree(model)
    target = "backbone.2"
    page = render_html(
        root,
        title="smoke",
        annotations={target: "⚡ inserts[1] pre-hook"},
        highlight=[target, "inserts.1"],
        initial_depth=1,
    )
    assert page.count("<details") == sum(1 for n in iter_nodes(root) if n.children), \
        "every non-leaf renders exactly one <details>"
    assert "⚡ inserts[1] pre-hook" in page, "annotation badge missing"
    # the highlighted node's parent chain must start open so the badge is visible on load
    assert '<details class="node" open' in page, "ancestor of highlight not open"
    assert 'id="filter"' in page and 'id="expand"' in page and 'id="collapse"' in page
    assert "http://" not in page and "https://" not in page, "must be offline-standalone"
    assert "❄" in page and "\U0001f525" in page, "frozen/trainable glyphs missing"
    print(f"html: OK ({len(page)} bytes)")


# ---------------------------------------------------------------- trace.py
def test_trace():
    model = ToyHost()
    watch = {
        "A0": model.backbone[0], "G0": model.inserts[0],
        "A1": model.backbone[2], "G1": model.inserts[1],
    }
    res = trace_forward(watch, lambda: model(torch.randn(2, 4)))

    order = [(e.path, e.kind) for e in res.events]
    assert order == [
        ("A0", "pre"), ("G0", "pre"), ("G0", "post"), ("A0", "post"),
        ("A1", "pre"), ("G1", "pre"), ("G1", "post"), ("A1", "post"),
    ], f"unexpected event order: {order}"
    assert res.events[0].shapes == ((2, 4),), "pre-hook should record input shapes"

    good = assert_gligen_pairing(res, [("G0", "A0"), ("G1", "A1")])
    assert good.ok, f"correct pairing must PASS:\n{good.summary()}"

    # negative: index-swapped pairing must FAIL (this is the bug class we hunt)
    bad = assert_gligen_pairing(res, [("G1", "A0"), ("G0", "A1")])
    assert not bad.ok, "misaligned pairing must FAIL"
    assert all(not c.ok for c in bad.checks)
    print("trace: OK (golden order, PASS on correct pairing, FAIL on swapped)")


def test_doctests():
    results = doctest.testmod(nnscope.trace, verbose=False)
    assert results.attempted > 0 and results.failed == 0, \
        f"doctests: {results.failed} failures / {results.attempted} attempted"
    print(f"doctests: OK ({results.attempted} examples)")


if __name__ == "__main__":
    test_tree()
    test_html()
    test_trace()
    test_doctests()
    print("smoke_nnscope: ALL OK")
