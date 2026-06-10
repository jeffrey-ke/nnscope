"""Record the execution order of watched modules during one forward pass.

Built for verifying hook-based layer insertion (e.g. GLIGEN-style blocks that
live OUTSIDE the host model's tree and run from a ``register_forward_pre_hook``
on the host layer): structure alone cannot prove such a block runs — only an
execution trace can. The hook may be EPHEMERAL (installed inside forward for one
pass) or PERSISTENT (installed once at construction); both are handled — see
``assert_gligen_pairing`` for how the two read differently in the trace.
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
    by an injection hook still produces events. Register the trace BEFORE the
    traced forward. Pre-hooks fire in registration order, so where the injection
    hook sits relative to ours depends on when IT was registered: an EPHEMERAL
    hook installed inside forward is appended after ours (our pre-event opens the
    window; the block nests inside), while a PERSISTENT hook installed at
    construction precedes ours (the block fires just before our pre-event).
    ``assert_gligen_pairing`` accepts both.

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

    Persistent variant — the injection hook is registered at CONSTRUCTION (before
    the trace markers), so the block fires just AHEAD of attn's own pre marker
    rather than nested inside it; assert_gligen_pairing accepts this order too.

    >>> class ToyPersistent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.attn = nn.Linear(4, 4)
    ...         self.gligen = nn.Linear(4, 4)
    ...         self.attn.register_forward_pre_hook(lambda m, a: (self.gligen(a[0]),))
    ...     def forward(self, x):
    ...         return self.attn(x)
    >>> toy = ToyPersistent()
    >>> res = trace_forward({"A0": toy.attn, "G0": toy.gligen}, lambda: toy(torch.randn(2, 4)))
    >>> [(e.path, e.kind) for e in res.events]
    [('G0', 'pre'), ('G0', 'post'), ('A0', 'pre'), ('A0', 'post')]
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
    """Check each (gligen, attn) pair fired once with gligen running in attn's pre-hook.

    The block runs as a forward-PRE-hook on its attn, so it executes on the tokens
    entering the attn, before the attn computes. Its position RELATIVE to the
    tracer's own ``attn`` pre marker depends only on hook-registration order, not
    on any semantic property of the injection (see ``trace_forward``):

      * EPHEMERAL hook (installed in forward, after the trace markers) — the block
        nests inside the window:        A_pre < G_pre < G_post < A_post
      * PERSISTENT hook (installed at construction, before the markers) — the block
        fires just ahead of the marker:  G_pre < G_post < A_pre < A_post

    Either order proves "G ran in A's pre-hook, before A's body". Per pair (G, A):

      1. G and A each fired exactly once (one pre + one post),
      2. the four events match one of the two orders above (G fully before A_post),
      3. no OTHER pair's gligen pre landed in [min(G_pre, A_pre), A_post] — catches
         index misalignment between target layers and their blocks. The window
         starts at the earlier of the two pre-events so it spans the block whether
         it nested inside A or fired just ahead of A's marker. Tolerating unrelated
         events in between is deliberate — this is nesting, not an adjacency check.
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
        nested = a_pre[0] < g_pre[0] < g_post[0] < a_post[0]        # ephemeral hook
        pre_adjacent = g_pre[0] < g_post[0] < a_pre[0] < a_post[0]  # persistent hook
        if not (nested or pre_adjacent):
            checks.append(PairCheck(g, a, False,
                f"block did not run in {a}'s pre-hook window: {a}_pre={a_pre[0]} "
                f"{g}_pre={g_pre[0]} {g}_post={g_post[0]} {a}_post={a_post[0]}"))
            continue
        lo = min(g_pre[0], a_pre[0])
        intruders = [
            other for other in gligen_labels
            if other != g
            for s in result.seq_of(other, "pre")
            if lo < s < a_post[0]
        ]
        if intruders:
            checks.append(PairCheck(g, a, False, f"other gligen in {a}'s window: {intruders}"))
            continue
        checks.append(PairCheck(g, a, True,
            f"fired once, {'nested in' if nested else 'just ahead of'} attn pre-hook"))
    return PairingReport(ok=all(c.ok for c in checks), checks=checks)
