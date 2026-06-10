# Plan: Gradient sanity-check preflight (nnscope mechanism + segmentation policy)

> **Status: COMPLETED** — built and verified end-to-end on CUDA.
>
> **As-built results**
> - nnscope CPU smoke: `ALL OK` (15 doctests; `GradToy` exercises flowed/zero/none/gate, the
>   detached-output raise, freeze-after-construction, and the overlay glyphs + legend).
> - gligen (CUDA): static audit `282 trainable / 751 frozen — PASS`; probe `282 flowed, 0 zero, 0 none`
>   (gates opened); `MuggledSamSegmenter` (217M params) `[frozen ok]`, no leak. HTML overlay: every node
>   ✅ or ❄.
> - verifier (CUDA): static audit `193 trainable / 1 frozen — PASS`; probe `193 flowed, 0 zero, 0 none`;
>   `SAMV2CoordinateEncoder` posenc (256 params) `[frozen ok]`; `DiftFpn` `[vacuous]` (no registered params).
>
> **As-built notes / deltas from the plan below**
> - The frozen-subtree policy distinguishes *truly vacuous* (a frozen type with **zero** registered params,
>   e.g. `DiftFpn`) from a *real fully-frozen module* (`posenc`, which has frozen params and is asserted
>   leak-free + zero-trainable). Only the former prints `[vacuous]`.
> - `preflight.py` factors out `gligen_dummy_inputs` / `verifier_dummy_inputs` and
>   `preflight_{gligen,verifier}_model(model, *, optimizer, out_path)` so the standalone scripts and the
>   `on_fit_start` hook share one path; `on_fit_start` passes Lightning's real optimizer
>   (`_unwrap_optimizer` strips the `LightningOptimizer` wrapper).
> - Verifier dummy `obs` is uint8 **RGBA** `(B,4,512,512)` — `DiftFpn.prep` is `StripAlpha → Resize(512) →
>   float[0,1]`; proposals are `rand*(W-1)` so `(xy+0.5)/W < 1`.
> - Observed-glyph map shipped as `flowed→✅ zero→⚠️ none→💀 mixed→🟡 frozen-leak→🚨`.
> - Files: `nnscope/src/nnscope/{grad.py(NEW),tree.py,html.py,__init__.py}`, `tests/smoke_nnscope.py`;
>   `segmentation/src/segmentation/preflight.py(NEW)`, `scripts/preflight_{gligen,verifier}.py(NEW)`,
>   `on_fit_start` + `preflight: bool` flag in `train.py` and `verifier/train.py`.

---

## Context

Before training **gligen** (`GligenWrapper`) and the **verifier**, we want a preflight that proves the
right things train and the right things stay frozen. Both models freeze indirectly — gligen freezes SAM
via `requires_grad_(False)` and splices trainable blocks in through *ephemeral* `register_forward_pre_hook`s
(`model.py:44-45,94-105`); the verifier freezes its FPN provider and `posenc.gaussian_matrix`
(`verifier.py:38`, `descriptor.py:85`). Both build their optimizer with
`trainable = [p for p in model.parameters() if p.requires_grad]` (`train.py:209`, `verifier/train.py:108`).
In all these idioms the `requires_grad` *flag* and the *actual gradient flow* can silently disagree: a
hook that detaches, a freeze applied after the optimizer captured a param, a branch that never runs. A
static flag-inspection can't see those; only a real forward+backward can.

This refines the user's first-draft `gradient_check(model, dummy_input, modules_need_grad, modules_frozen)`
into composable pieces, because that signature fits neither model: gligen takes 3 args and returns a tuple
`(logits, iou, db_info)`; the verifier takes 4 args and returns a tensor — so `model(dummy_input).sum()`
can't work. The user's `filter(isinstance(block, DiftFpn), model)` selector becomes `find_modules`.

**Home = `nnscope`** (not vision_core): it already owns the collapsible module-tree renderer
(`build_tree`/`render_html`, with a static ❄/🔥/🔥❄ `requires_grad` glyph at `html.py:70-77`) and the
observational hook-tracing precedent (`trace_forward`, `trace.py`), and it already enforces the
generic-mechanism / project-policy split this needs. The probe is the **dynamic complement** to
`tree.py`'s static `requires_grad` rollup.

**Decided architecture (hybrid, per user):** torchvista stays **untouched** as the independent *position*
witness (`gligen_torchvista.html` already works; spike confirmed it has no styling hook and keys nodes by
`{op_type}_{counter}`, not module path — a poor carrier for per-param grad). The grad-flow overlay rides
on the nnscope **tree** with *observed* coloring (measured `.grad`), so the *training* claim is
observational even though tree node positions are declared.

## Coordination with the active plan

`independent-verification-channels.md` (active, unbuilt) separately adds `strict` + an `Annotation(badge,
detail)` type to `render_html`, and `derive_attachments` to `trace.py`. This work is orthogonal: a new
`grad.py` module (disjoint from `trace.py`), and the observed glyph rides the **params span**
(`html.py:97`), never the badge span (`html.py:99`). Do **not** widen `annotations: dict[str,str]` (that's
the active plan's job). Whichever lands second just adds its kwarg.

---

## Stage 1 — nnscope mechanism (torch-only, CPU-testable; no CUDA, no weights)

### 1a. `src/nnscope/grad.py` (NEW) — match `trace.py` style (frozen dataclasses + doctest-as-spec)

```python
GradState = Literal["none", "zero", "flowed", "mixed", "frozen-leak"]
# param level uses only none|zero|flowed; module rollup may also be mixed|frozen-leak

@dataclass(frozen=True)
class ParamGrad:      path; requires_grad; state: GradState; grad_abs_max: float; numel: int
@dataclass(frozen=True)
class ModuleGrad:     path; state: GradState; n_params; n_flowed; n_zero; n_none; received_signal: bool
@dataclass(frozen=True)
class GradReport:
    params: tuple[ParamGrad, ...]      # one per requires_grad=True param, named_parameters order
    modules: dict[str, ModuleGrad]     # keyed by module path ("" = root)
    frozen_leaks: tuple[str, ...]      # requires_grad=False params that got a non-None grad
    def summary(self) -> str: ...
    def subtree_received_signal(self, path: str) -> bool: ...   # modules[path].received_signal
```

**`find_modules(root, *types) -> list[nn.Module]`** — the user's `filter(isinstance(...))` done right:
`[m for m in root.modules() if isinstance(m, types)]`. The "grab the frozen subtree by type" selector.

**`probe_gradients(model, forward, *, reduce=sum_floating, zero_tol=0.0, seed=None) -> GradReport`** — thunk
+ reduction form (serves tuple- and tensor-returning models):
1. `model.zero_grad(set_to_none=True)` (clean state — every `.grad` is None).
2. `seed` → `torch.manual_seed` if set.
3. `with torch.enable_grad():` (defeats ambient `no_grad`/`inference_mode`) → `out = forward()`;
   `loss = reduce(out)`; **assert `loss.requires_grad`** (fail-loud on a fully-detached output); `loss.backward()`.
4. Classify each `named_parameter`: `requires_grad=True` → `ParamGrad` state `none` (grad is None) /
   `zero` (present, `|g|<=zero_tol`) / `flowed`; `requires_grad=False` with non-None grad → `frozen_leaks`.
5. Roll module states up by dotted-prefix. **Combine rule:** all-flowed→`flowed`, all-zero→`zero`,
   all-none→`none`, else→`mixed`; `received_signal = n_flowed >= 1`.
6. `model.zero_grad(set_to_none=True)` (leave caller's grads clean).

`sum_floating(out)` — default reduction; sums `.float().sum()` over every floating tensor reachable in
tensor/list/tuple/dict (mirror the walk in `trace.py:33-47`); raises if none found. Handles the gligen
tuple and the verifier tensor.

**Decision: the probe is pure observation; assertions are policy.** No `modules_need_grad`/`modules_frozen`
args — it auto-partitions by `requires_grad`. Policy supplies the expected-frozen subtree via
`find_modules(model, *types)` and asserts on the report. `received_signal` is the **gate-immune wiring
witness** (at gligen init every projection is `zero` because `tanh(gamma)=0`, but `gamma` itself is
`flowed`, so the GligenBlock subtree reports `received_signal=True`) — policy asserts on this, never on
per-projection `flowed`.

**`static_grad_audit(model, optimizer=None) -> StaticAudit`** (no forward) — partition by `requires_grad`;
if an optimizer is given, assert its param-group set (`{id(p) for g in opt.param_groups for p in g["params"]}`)
**exactly equals** the `requires_grad=True` set. Report fields: `missing_from_optimizer`,
`extra_in_optimizer` (the freeze-after-construction bug — frozen param still in a group, silently never
updates), `ok`. Names from `named_parameters()` so diffs carry dotted paths.

**Exports** (`__init__.py:3-26`): `find_modules, probe_gradients, GradReport, ParamGrad, ModuleGrad,
GradState, static_grad_audit, StaticAudit, sum_floating, apply_grad_report`.

### 1b. `src/nnscope/tree.py` — carry observed state
- Add `grad_state: GradState | None = None` to `ModuleNode` (defaults None; `build_tree` constructs nodes
  by keyword at `tree.py:42-51`, so back-compat holds).
- `apply_grad_report(root, report)` — walk `iter_nodes` (`tree.py:72-76`), stamp `node.grad_state` from
  `report.modules[node.path]`; a node whose subtree contains a `frozen_leaks` path → `"frozen-leak"`;
  untrained/absent nodes stay None. Keeps `build_tree` untouched (build once, annotate post hoc).

### 1c. `src/nnscope/html.py` — observed overlay (second glyph, distinct from static ❄/🔥/🔥❄)
- `render_html(..., grad_states: dict[str, GradState] | None = None)`. Resolution: explicit `grad_states`
  wins, else read `node.grad_state`. Thread a `grad_lookup` into `_render_node`/`_summary`.
- `_observed_glyph(state)` map: **`flowed→✅  zero→⚠️  none→💀  mixed→🟡  frozen-leak→🚨`**; `None`/untrained→`""`.
- In `_summary` (`html.py:90-105`) the params span becomes `{static_glyph}{observed_glyph} {humanize(total)}`:
  frozen leaf `❄ 16K`; healthy trainable `🔥✅ 16K`; gated projection `🔥⚠️ 1.2M`; disconnected trainable
  `🔥💀 …`; frozen leak `❄🚨 …`. Static glyph = **declared** position; observed glyph = **observed** flow.
- Toolbar legend (`html.py:164`), only when any state is set:
  `positions ❄/🔥 = declared (requires_grad) · observed after 1 backward: ✅ flowed ⚠️ zero(gated) 💀 none(disconnected) 🚨 frozen-leak`.
  Add a muted `.legend` CSS rule near `html.py:30`.
- **Back-compat:** no `grad_states` + untouched tree → no observed glyph, no legend, byte-identical output.
  html.py stays torch-free (it only maps state→glyph; `grad.py`/`tree.py` produced the states).

### 1d. `tests/smoke_nnscope.py` — CPU verification
`GradToy` forces all fates without CUDA: a `.requires_grad_(False)` submodule (frozen), a healthy Linear
(flowed), a Linear **never used in forward** (none/disconnected), and a `tanh(gate)*gated(x)` pair with
`gate=0` (gate param flowed, `gated` weights zero — rehearses the gamma gate). Tests:
- `test_grad_probe`: assert per-param states; `modules["gated"].received_signal is False`;
  `subtree_received_signal("") is True`; **negative** — detached output raises the `loss.requires_grad` assert.
- `test_static_audit`: build optimizer over trainable params → `ok`; then `requires_grad_(False)` a module
  **after** → `extra_in_optimizer` non-empty, `ok is False`; no-optimizer → `optimizer_param_ids is None`, `ok`.
- `test_grad_overlay_html`: `apply_grad_report` + `render_html` shows ✅/⚠️/💀 + legend; no-states render
  has neither.
- Add `doctest.testmod(nnscope.grad)` to `test_doctests`; wire all into `__main__`.

**Run:** `env -u PYTHONPATH uv run python tests/smoke_nnscope.py` from `/home/jeffk/repo/nnscope` → `ALL OK`.

---

## Stage 2 — segmentation policy (CUDA, end-to-end)

### 2a. `src/segmentation/preflight.py` (NEW) — reusable composition
```python
def preflight_gradients(model, forward_fn, *, optimizer=None, expect_frozen_types=(),
                        reduce=nnscope.sum_floating, title, out_path) -> int:
```
1. `audit = static_grad_audit(model, optimizer)`; print summary; **assert `audit.ok`** (fail-loud).
2. `report = probe_gradients(model, forward_fn, reduce=reduce, seed=0)`; print summary.
3. Frozen policy: for each `m in find_modules(model, *expect_frozen_types)`, assert its subtree is fully
   frozen and `report.subtree_received_signal(path) is False`. **DiftFpn vacuity:** a frozen type with no
   registered params → vacuously true but **print a loud `[vacuous]` warning** (its SD weights live outside
   `nn.Module`, `descriptor.py:142-167` — the check proved nothing).
4. Wiring policy: assert `subtree_received_signal("") is True` and `not report.frozen_leaks`.
5. `root = build_tree(model); apply_grad_report(root, report); write_html(render_html(root, title=title,
   initial_depth=2), out_path)`.
6. Return `0` iff all assertions pass.

### 2b. gligen entry point — mirror `inspect_gligen.py:51-92` for construction + dummy inputs
- `assert torch.cuda.is_available()`; `os.chdir(...)`; `wrapper = GligenWrapper.from_config(load_gligen_wrapper(
  "configs/gligen_wrapper.yaml")).cuda().eval()`.
- Dummy inputs **verbatim** from `inspect_gligen.py:85-90` (uint8 image; `inject_dim = next(m for m in
  wrapper.mlp.modules() if isinstance(m, nn.Linear)).in_features`; rand pts; randn ref).
- `forward_fn = lambda: wrapper(img, pts, ref)`; `reduce = sum_floating` (tuple output).
- `expect_frozen_types = (type(wrapper.segmenter),)` (the SAM subtree, frozen at `model.py:44-45`).
- **Gate policy — open gates around the probe** so the overlay shows *true wiring* not the init gate:
  `with` a context that sets every `GatedMHA.gamma` (`find_modules(wrapper, GatedMHA)`) to `1.0` under
  `no_grad` and restores in `finally` (same trick as Channel B). Document: at init `tanh(0)=0` makes every
  projection grad exactly zero (correct GLIGEN, `blocks.py:15,41,58`); opening gates disambiguates
  "gated" from "disconnected" so projections read `🔥✅`. Optionally a second gates-closed probe to confirm
  `received_signal` (gamma flows) — but render the gates-open run.

### 2c. verifier entry point — construct via `verifier_from_config` (`verifier.py:192-209`)
- `model = verifier_from_config(cfg).cuda().eval()` (DiftFpn provider, `c_ref=1280`).
- Dummy inputs: `obs = tv_tensors.Image(uint8 (B,3,H,W) cuda)`; `proposals = rand(B,N,2)*[W,H]`;
  `proposals_valid_mask = ones(B,N,bool)`; `ref_tok = randn(B,1280,h,w, cuda)` (`verifier.py:145,186`).
  Call shape from `verifier/train.py:119`: `model(obs, proposals, valid, ref_tok=ref_tok)` → tensor.
- `expect_frozen_types = (type(model.posenc), type(model.point_descriptor_extractor.provider))` — posenc is
  a real frozen check (`verifier.py:38`); DiftFpn triggers the vacuous warning (use M2FFpn for a real FPN
  freeze check). No gate trick (no GatedMHA); all trainable subtrees should read `flowed`.
- **Edge — heavy DIFT forward:** the probe runs the full DIFT/SD featurizer over `obs` (`point_descriptor.py`);
  slow, needs the diffusers pipeline on GPU. B=1, run once as preflight; SD weights are invisible to
  autograd-through-`nn.Module` so never asserted.

### 2d. Invocation
Ship both: (1) standalone `scripts/preflight_gligen.py` + `scripts/preflight_verifier.py` (the
`inspect_gligen.py` pattern); (2) a call from the LightningModule's `on_fit_start`, **after**
`configure_optimizers` so the *real* optimizer is available (`self.optimizers()`), guarded by a
`cfg.preflight` flag (default off). The `on_fit_start` path is the only one that catches the
freeze-after-construction bug for Lightning's actual optimizer.

---

## Verification (end-to-end)
- nnscope CPU: `env -u PYTHONPATH uv run python tests/smoke_nnscope.py` (from nnscope) → `ALL OK`.
- gligen (CUDA): `cd /home/jeffk/repo/segmentation && env -u PYTHONPATH uv run python scripts/preflight_gligen.py`
  → audit ok; gates-open projections `flowed`; SAM subtree `received_signal False`; HTML shows `❄` on SAM,
  `🔥✅` on mlp/gligen; exit 0.
- verifier (CUDA, heavy): `… scripts/preflight_verifier.py` → posenc `❄`; DiftFpn `[vacuous]` warning;
  downshifts/ref_encoder/decoder/finisher/cls_token/logit_proj_bias `flowed`; no `none` on a used branch; exit 0.
- Eyeball both HTML overlays (params-span glyphs + legend).

## Edge-case ledger
- **gamma gate** → probe reports `zero` honestly; policy uses `received_signal`; gligen opens gates for the overlay (save/restore in `finally`).
- **DiftFpn vacuity** → frozen check on a no-registered-param provider is vacuous → loud warning, never silent pass.
- **tuple output** → `sum_floating` walks it; `loss.requires_grad` assert guards a detached output.
- **optimizer-not-passed** → partition only, identity check skipped (documented); `on_fit_start` always passes the real optimizer.
- **ambient no_grad** → probe wraps forward in `enable_grad()` (unlike `inspect_gligen.py:96`'s `no_grad` trace).
- **back-compat** → `ModuleNode.grad_state` defaults None; `render_html` without `grad_states` is byte-identical (minus the omitted legend).

## Sequencing
Stage 1 (nnscope, CPU, `GradToy` rehearses gate/disconnect/freeze-after-construction) lands and is verified
before Stage 2 (segmentation, CUDA + weights). Commit only on explicit request; branch first if on `master`.

## Critical files
- `nnscope/src/nnscope/grad.py` (NEW) · `nnscope/src/nnscope/tree.py` · `nnscope/src/nnscope/html.py` ·
  `nnscope/src/nnscope/__init__.py` · `nnscope/tests/smoke_nnscope.py`
- `segmentation/src/segmentation/preflight.py` (NEW) · `scripts/preflight_gligen.py` ·
  `scripts/preflight_verifier.py` · LightningModule `on_fit_start` in `train.py` + `verifier/train.py`
