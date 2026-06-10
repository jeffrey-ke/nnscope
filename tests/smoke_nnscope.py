"""Smoke test for nnscope: tree capture, HTML rendering, execution tracing.

CPU-only, no weights, no GPU. Run from the repo root:

    env -u PYTHONPATH uv run python tests/smoke_nnscope.py
"""

import doctest

import torch
import torch.nn as nn

import nnscope.grad
import nnscope.trace
from nnscope import (
    apply_grad_report,
    assert_gligen_pairing,
    build_tree,
    iter_nodes,
    probe_gradients,
    render_html,
    static_grad_audit,
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


# ---------------------------------------------------------------- grad.py
class GradToy(nn.Module):
    """One module per gradient fate, all forced without CUDA:

      frozen        used, requires_grad_(False)        -> no grad (not a leak)
      healthy       used, trainable                    -> "flowed"
      disconnected  trainable but NEVER used           -> "none"
      gate          nn.Parameter(0.) in tanh(gate)*..  -> "flowed" (sech^2(0)=1)
      gated         used but scaled by tanh(0)=0       -> "zero"  (the GLIGEN gate)
    """

    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.healthy = nn.Linear(4, 4)
        self.disconnected = nn.Linear(4, 4)
        self.gate = nn.Parameter(torch.zeros(()))
        self.gated = nn.Linear(4, 4)

    def forward(self, x):
        return self.frozen(x) + self.healthy(x) + torch.tanh(self.gate) * self.gated(x)


def test_grad_probe():
    m = GradToy()
    r = probe_gradients(m, lambda: m(torch.randn(2, 4)), seed=0)
    st = {p.path: p.state for p in r.params}
    assert st["healthy.weight"] == "flowed" and st["healthy.bias"] == "flowed"
    assert st["disconnected.weight"] == "none" and st["disconnected.bias"] == "none"
    assert st["gate"] == "flowed", "the gate param itself must receive signal"
    assert st["gated.weight"] == "zero" and st["gated.bias"] == "zero"
    assert "frozen.weight" not in st and not r.frozen_leaks, "frozen leaked into the report"

    assert r.modules["healthy"].state == "flowed"
    dc = r.modules["disconnected"]
    assert dc.state == "none" and not dc.received_signal
    gated = r.modules["gated"]
    assert gated.state == "zero" and not gated.received_signal, \
        "closed gate => present-but-zero grad, no signal"
    assert r.modules[""].state == "mixed"
    # gate-immune wiring witness: the model as a whole got signal despite the closed gate
    assert r.subtree_received_signal() and r.subtree_received_signal("")

    # negative: a fully-detached output has nothing to probe -> fail loud
    detached = nn.Linear(4, 4)
    try:
        probe_gradients(detached, lambda: detached(torch.randn(2, 4)).detach())
    except ValueError:
        pass
    else:
        raise AssertionError("probe must raise on a detached (no-grad) output")
    print("grad probe: OK (flowed/zero/none/gate + detached-raises)")


def test_static_audit():
    m = GradToy()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    a = static_grad_audit(m, opt)
    assert a.ok and a.has_optimizer
    assert not a.missing_from_optimizer and not a.extra_in_optimizer

    # freeze a module AFTER the optimizer captured it — the classic silent bug
    m.healthy.requires_grad_(False)
    b = static_grad_audit(m, opt)
    assert not b.ok, "freeze-after-construction must FAIL the audit"
    assert any("healthy" in n for n in b.extra_in_optimizer), b.summary()

    # no optimizer => requires_grad partition only; identity not checked, ok True
    c = static_grad_audit(m)
    assert c.ok and not c.has_optimizer
    print("static audit: OK (matches, catches freeze-after-construction, partition-only)")


def test_grad_overlay_html():
    m = GradToy()
    r = probe_gradients(m, lambda: m(torch.randn(2, 4)), seed=0)
    root = build_tree(m)
    apply_grad_report(root, r)
    page = render_html(root, title="grad")
    assert "✅" in page, "flowed glyph missing"
    assert "⚠" in page, "zero (gated) glyph missing"
    assert "\U0001f480" in page, "none (disconnected) glyph missing"
    assert 'class="legend"' in page, "legend missing when states present"

    # back-compat: a fresh (un-stamped) tree with no grad_states renders no overlay
    plain = render_html(build_tree(m))
    assert "✅" not in plain and "\U0001f480" not in plain, "observed glyph leaked into plain render"
    assert 'class="legend"' not in plain, "legend must be omitted without states"
    print(f"grad overlay html: OK ({len(page)} bytes)")


def test_doctests():
    attempted = failed = 0
    for mod in (nnscope.trace, nnscope.grad):
        results = doctest.testmod(mod, verbose=False)
        attempted += results.attempted
        failed += results.failed
    assert attempted > 0 and failed == 0, \
        f"doctests: {failed} failures / {attempted} attempted"
    print(f"doctests: OK ({attempted} examples)")


if __name__ == "__main__":
    test_tree()
    test_html()
    test_trace()
    test_grad_probe()
    test_static_audit()
    test_grad_overlay_html()
    test_doctests()
    print("smoke_nnscope: ALL OK")
