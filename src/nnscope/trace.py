"""Record the execution order of watched modules during one forward pass.

Built for verifying hook-based layer insertion (e.g. GLIGEN-style blocks that
live OUTSIDE the host model's tree and are called from ephemeral
``register_forward_pre_hook``s installed only for the duration of a forward):
structure alone cannot prove such a block runs — only an execution trace can.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Event:
    path: str                              # the label the module was watched under
    kind: Literal["pre", "post"]
    shapes: tuple[tuple[int, ...], ...]    # tensor shapes in args (pre) / output (post)
    seq: int                               # global monotonic order


@dataclass
class TraceResult:
    events: list[Event]

    def seq_of(self, path: str, kind: str) -> list[int]:
        return [e.seq for e in self.events if e.path == path and e.kind == kind]


def _tensor_shapes(obj: Any) -> tuple[tuple[int, ...], ...]:
    shapes: list[tuple[int, ...]] = []

    def walk(o: Any) -> None:
        if isinstance(o, torch.Tensor):
            shapes.append(tuple(o.shape))
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)

    walk(obj)
    return tuple(shapes)


def trace_forward(watch: dict[str, nn.Module], run: Callable[[], Any]) -> TraceResult:
    """Hook every watched module (pre + post), call ``run``, return the ordered log.

    Hooks fire in ``Module.__call__`` no matter who calls the module — including
    a call made from inside another module's pre-hook — so a side module invoked
    by an ephemeral injection hook still produces events. Register the trace
    BEFORE the traced forward: any hook the forward itself installs on a watched
    module is appended after ours, so our pre-event opens the module's window.

    >>> class Toy(nn.Module):
    ...     '''Mimics GligenWrapper: `gligen` is a SEPARATE module, called from an
    ...     ephemeral pre-hook installed on `attn` during forward (like hook_into).'''
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.attn = nn.Linear(4, 4)     # stands in for a SAM attn block
    ...         self.gligen = nn.Linear(4, 4)   # stands in for its GligenBlock
    ...     def forward(self, x):
    ...         h = self.attn.register_forward_pre_hook(lambda m, a: (self.gligen(a[0]),))
    ...         try:
    ...             return self.attn(x)
    ...         finally:
    ...             h.remove()
    >>> toy = Toy()
    >>> res = trace_forward({"A0": toy.attn, "G0": toy.gligen}, lambda: toy(torch.randn(2, 4)))
    >>> [(e.path, e.kind) for e in res.events]
    [('A0', 'pre'), ('G0', 'pre'), ('G0', 'post'), ('A0', 'post')]
    >>> assert_gligen_pairing(res, [("G0", "A0")]).ok
    True
    """
    events: list[Event] = []
    handles = []

    def make(label: str):
        def pre(_mod, args):
            events.append(Event(label, "pre", _tensor_shapes(args), len(events)))

        def post(_mod, _args, output):
            events.append(Event(label, "post", _tensor_shapes(output), len(events)))

        return pre, post

    try:
        for label, mod in watch.items():
            pre, post = make(label)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))
        run()
    finally:
        for h in handles:
            h.remove()
    return TraceResult(events=events)


@dataclass
class PairCheck:
    gligen: str
    attn: str
    ok: bool
    reason: str


@dataclass
class PairingReport:
    ok: bool
    checks: list[PairCheck] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"pairing: {'PASS' if self.ok else 'FAIL'}"]
        for c in self.checks:
            lines.append(f"  [{'ok' if c.ok else 'FAIL'}] {c.gligen} -> {c.attn}: {c.reason}")
        return "\n".join(lines)


def assert_gligen_pairing(result: TraceResult, pairs: list[tuple[str, str]]) -> PairingReport:
    """Check each (gligen, attn) pair fired once with gligen NESTED in attn's window.

    The invariant is nesting, not adjacency — deliberately: nesting tolerates
    other watched modules' events landing in between, where an adjacency check
    would be brittle. Per pair (G, A):

      1. G and A each fired exactly once (one pre + one post),
      2. A_pre < G_pre < G_post < A_post  — G ran inside A's pre-hook window,
         i.e. on the tokens entering A, before A computed,
      3. no OTHER pair's gligen entered A's window — catches index misalignment
         between target layers and their blocks.
    """
    gligen_labels = [g for g, _ in pairs]
    checks: list[PairCheck] = []
    for g, a in pairs:
        g_pre, g_post = result.seq_of(g, "pre"), result.seq_of(g, "post")
        a_pre, a_post = result.seq_of(a, "pre"), result.seq_of(a, "post")
        if not all(len(s) == 1 for s in (g_pre, g_post, a_pre, a_post)):
            checks.append(PairCheck(g, a, False,
                f"fire counts: {g} pre/post={len(g_pre)}/{len(g_post)}, "
                f"{a} pre/post={len(a_pre)}/{len(a_post)} (each must be exactly 1)"))
            continue
        nested = a_pre[0] < g_pre[0] < g_post[0] < a_post[0]
        if not nested:
            checks.append(PairCheck(g, a, False,
                f"not nested: {a}_pre={a_pre[0]} {g}_pre={g_pre[0]} "
                f"{g}_post={g_post[0]} {a}_post={a_post[0]}"))
            continue
        intruders = [
            other for other in gligen_labels
            if other != g
            for s in result.seq_of(other, "pre")
            if a_pre[0] < s < a_post[0]
        ]
        if intruders:
            checks.append(PairCheck(g, a, False, f"other gligen in {a}'s window: {intruders}"))
            continue
        checks.append(PairCheck(g, a, True, "fired once, nested in attn window"))
    return PairingReport(ok=all(c.ok for c in checks), checks=checks)
