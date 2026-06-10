"""Measure which parameters actually receive gradient after one backward pass.

The dynamic complement to :mod:`nnscope.tree`'s static ``requires_grad`` rollup:
``requires_grad`` is a FLAG (does autograd track this param); this module MEASURES
whether gradient actually arrives (is the param wired to the loss, or silently
detached / behind a broken hook / in a dead branch / multiplied by a closed gate).
The two can disagree, and only a real forward+backward reveals it.

Pure-torch, no project deps. The probe is honest OBSERVATION — it never decides
whether a "zero" gradient is a bug (a GLIGEN gate at ``tanh(0)=0`` legitimately
zeroes its projection grads at init); callers assert intent against the report,
and ``received_signal`` (>=1 param flowed under a subtree) is the gate-immune
witness that a subtree is wired to the loss.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

# param level uses only none|zero|flowed; module/rollup level may also be mixed|frozen-leak
GradState = Literal["none", "zero", "flowed", "mixed", "frozen-leak"]


def find_modules(root: nn.Module, *types: type[nn.Module]) -> list[nn.Module]:
    """Every module in ``root.modules()`` that isinstance-matches any of ``types``.

    The correct spelling of ``filter(lambda m: isinstance(m, T), model)``:
    ``modules()`` recurses (iterating the model directly yields only immediate
    children), and a tuple of types matches any. ``root`` itself is included if it
    matches. This is the "grab the frozen subtree by type" selector, e.g.
    ``find_modules(verifier, M2FFpn)`` or ``find_modules(wrapper, GatedMHA)``.

    >>> m = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    >>> [type(x).__name__ for x in find_modules(m, nn.Linear)]
    ['Linear', 'Linear']
    >>> find_modules(m)
    []
    """
    if not types:
        return []
    return [m for m in root.modules() if isinstance(m, types)]


def sum_floating(out: Any) -> torch.Tensor:
    """Sum of ``t.float().sum()`` over every floating-point tensor reachable in ``out``.

    Walks tensors nested in list/tuple/dict and dataclass fields (to reach e.g.
    ``SamDebugInfo``'s tensors in the gligen ``(logits, iou, db_info)`` output).
    The default reduction for :func:`probe_gradients`: collapses a model's whole
    output to one scalar to backprop from, model-agnostic. Raises if no floating
    tensor is found (a detached or non-float output cannot seed gradients).

    >>> float(sum_floating((torch.ones(2), torch.ones(3))))
    5.0
    """
    parts: list[torch.Tensor] = []

    def walk(o: Any) -> None:
        if isinstance(o, torch.Tensor):
            if o.is_floating_point():
                parts.append(o.float().sum())
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif dataclasses.is_dataclass(o) and not isinstance(o, type):
            for f in dataclasses.fields(o):
                walk(getattr(o, f.name))

    walk(out)
    if not parts:
        raise ValueError("sum_floating: no floating-point tensor found in model output")
    return torch.stack(parts).sum()


@dataclass(frozen=True)
class ParamGrad:
    path: str            # dotted param path from the model root
    requires_grad: bool
    state: GradState     # "none" (grad is None) | "zero" (present, |g|<=tol) | "flowed"
    grad_abs_max: float  # 0.0 when grad is None
    numel: int


@dataclass(frozen=True)
class ModuleGrad:
    path: str               # dotted module path (matches ModuleNode.path; "" = root)
    state: GradState        # combine rule in _combine
    n_params: int           # trainable params under this module (recurse=True)
    n_flowed: int
    n_zero: int
    n_none: int
    received_signal: bool   # n_flowed >= 1 — gate-immune wiring witness


@dataclass(frozen=True)
class GradReport:
    params: tuple[ParamGrad, ...]    # one per requires_grad=True param, named_parameters order
    modules: dict[str, ModuleGrad]   # keyed by module path ("" = root)
    frozen_leaks: tuple[str, ...]    # requires_grad=False params that nonetheless got a grad

    def subtree_received_signal(self, path: str = "") -> bool:
        """True iff >=1 trainable param under ``path`` flowed (path="" = whole model).

        The wiring assertion to use in policy: immune to closed gates (a GLIGEN
        block whose projections are all ``zero`` at init still reports True because
        its ``gamma`` param flows)."""
        mg = self.modules.get(path)
        return bool(mg and mg.received_signal)

    def summary(self) -> str:
        by = {"flowed": 0, "zero": 0, "none": 0}
        for p in self.params:
            by[p.state] += 1
        lines = [f"grad probe: {by['flowed']} flowed, {by['zero']} zero, {by['none']} none "
                 f"(of {len(self.params)} trainable params)"]
        disconnected = [p.path for p in self.params if p.state == "none"]
        if disconnected:
            lines.append(f"  DISCONNECTED (trainable, no grad): {disconnected}")
        if self.frozen_leaks:
            lines.append(f"  FROZEN LEAK (frozen, got grad): {list(self.frozen_leaks)}")
        return "\n".join(lines)


def _combine(n_flowed: int, n_zero: int, n_none: int, has_leak: bool) -> GradState:
    if has_leak:
        return "frozen-leak"  # the alarm dominates; propagates to ancestors (recurse=True rollup)
    n = n_flowed + n_zero + n_none
    if n == 0 or n_flowed == n:
        return "flowed" if n else "none"  # n==0 => no trainable params; renderer skips via n_params
    if n_zero == n:
        return "zero"
    if n_none == n:
        return "none"
    return "mixed"


def _rollup_modules(
    model: nn.Module,
    trainable: dict[str, ParamGrad],
    frozen_leaks: list[str],
) -> dict[str, ModuleGrad]:
    """One ModuleGrad per module: aggregate the trainable params strictly under it.

    Each module's named_parameters(recurse=True) includes a descendant leak param,
    so a frozen-leak naturally propagates up to every ancestor (root included)."""
    leak_set = set(frozen_leaks)
    modules: dict[str, ModuleGrad] = {}
    for mpath, mod in model.named_modules():
        n_flowed = n_zero = n_none = 0
        has_leak = False
        for rel, _ in mod.named_parameters(recurse=True):
            full = f"{mpath}.{rel}" if mpath else rel
            pg = trainable.get(full)
            if pg is None:
                if full in leak_set:
                    has_leak = True
                continue
            if pg.state == "flowed":
                n_flowed += 1
            elif pg.state == "zero":
                n_zero += 1
            else:
                n_none += 1
        modules[mpath] = ModuleGrad(
            path=mpath,
            state=_combine(n_flowed, n_zero, n_none, has_leak),
            n_params=n_flowed + n_zero + n_none,
            n_flowed=n_flowed,
            n_zero=n_zero,
            n_none=n_none,
            received_signal=n_flowed >= 1,
        )
    return modules


def probe_gradients(
    model: nn.Module,
    forward: Callable[[], Any],
    *,
    reduce: Callable[[Any], torch.Tensor] = sum_floating,
    zero_tol: float = 0.0,
    seed: int | None = None,
) -> GradReport:
    """Run one clean forward+backward and classify every parameter's gradient.

    ``forward`` is a THUNK closing over its own inputs (so this serves both the
    tuple-returning gligen ``wrapper(img, pts, ref)`` and the tensor-returning
    verifier ``model(obs, ...)`` without a fixed input signature); ``reduce``
    collapses its output to a scalar loss (default :func:`sum_floating`). Grads are
    zeroed before and after, and the forward runs under ``torch.enable_grad()`` so
    an ambient ``no_grad``/``inference_mode`` does not silence the probe.

    Per requires_grad param: ``none`` (grad is None — disconnected from the loss),
    ``zero`` (grad present but ``|g| <= zero_tol`` everywhere), or ``flowed``.
    requires_grad=False params that receive a non-None grad land in
    ``frozen_leaks`` (a freeze that did not take).

    >>> m = nn.Linear(3, 3)
    >>> r = probe_gradients(m, lambda: m(torch.ones(2, 3)), seed=0)
    >>> r.modules[""].state
    'flowed'
    >>> r.subtree_received_signal()
    True
    >>> dead = nn.Linear(3, 3)  # never used in the forward => disconnected
    >>> probe_gradients(dead, lambda: m(torch.ones(2, 3)), seed=0).modules[""].state
    'none'
    """
    model.zero_grad(set_to_none=True)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.enable_grad():
        out = forward()
        loss = reduce(out)
        if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
            raise ValueError(
                "probe_gradients: reduced loss does not require grad — the model "
                "output is detached from all trainable params (nothing to probe)")
        loss.backward()

    params: list[ParamGrad] = []
    trainable: dict[str, ParamGrad] = {}
    frozen_leaks: list[str] = []
    for name, p in model.named_parameters():
        g = p.grad
        if not p.requires_grad:
            if g is not None:
                frozen_leaks.append(name)
            continue
        if g is None:
            state: GradState = "none"
            amax = 0.0
        else:
            amax = float(g.detach().abs().max())
            state = "flowed" if amax > zero_tol else "zero"
        pg = ParamGrad(path=name, requires_grad=True, state=state,
                       grad_abs_max=amax, numel=p.numel())
        params.append(pg)
        trainable[name] = pg

    modules = _rollup_modules(model, trainable, frozen_leaks)
    model.zero_grad(set_to_none=True)
    return GradReport(params=tuple(params), modules=modules,
                      frozen_leaks=tuple(frozen_leaks))


@dataclass(frozen=True)
class StaticAudit:
    n_trainable_params: int            # count of requires_grad=True params
    n_frozen_params: int
    missing_from_optimizer: tuple[str, ...]  # trainable but NOT in any param group
    extra_in_optimizer: tuple[str, ...]      # frozen but still in a param group
    has_optimizer: bool
    ok: bool

    def summary(self) -> str:
        lines = [f"static audit: {self.n_trainable_params} trainable / "
                 f"{self.n_frozen_params} frozen params — "
                 f"{'PASS' if self.ok else 'FAIL'}"]
        if not self.has_optimizer:
            lines.append("  (no optimizer passed — requires_grad partition only; "
                         "param-group identity NOT checked)")
        if self.missing_from_optimizer:
            lines.append(f"  MISSING from optimizer (trainable, never optimized): "
                         f"{list(self.missing_from_optimizer)}")
        if self.extra_in_optimizer:
            lines.append(f"  EXTRA in optimizer (frozen, still optimized): "
                         f"{list(self.extra_in_optimizer)}")
        return "\n".join(lines)


def static_grad_audit(
    model: nn.Module,
    optimizer: "torch.optim.Optimizer | None" = None,
) -> StaticAudit:
    """Partition params by ``requires_grad``; if an optimizer is given, assert its
    param-group set EXACTLY equals the requires_grad=True set (by ``id``).

    Catches the freeze-after-optimizer-construction bug: a param frozen AFTER the
    optimizer captured it stays in a param group (``extra_in_optimizer``) and
    silently never updates; a param unfrozen after stays out
    (``missing_from_optimizer``). With no optimizer, only the partition is reported
    (``ok`` is True) — the param-group identity is NOT verified.
    """
    named = list(model.named_parameters())
    trainable = {id(p): name for name, p in named if p.requires_grad}
    frozen = {id(p): name for name, p in named if not p.requires_grad}
    if optimizer is None:
        return StaticAudit(len(trainable), len(frozen), (), (), has_optimizer=False, ok=True)
    opt_ids = {id(p) for grp in optimizer.param_groups for p in grp["params"]}
    missing = tuple(sorted(name for pid, name in trainable.items() if pid not in opt_ids))
    extra = tuple(sorted(name for pid, name in frozen.items() if pid in opt_ids))
    return StaticAudit(len(trainable), len(frozen), missing, extra,
                       has_optimizer=True, ok=not missing and not extra)
