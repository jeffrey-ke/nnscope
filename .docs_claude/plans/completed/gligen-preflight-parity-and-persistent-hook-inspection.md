# Plan: gligen preflight↔train parity + persistent-hook injection mapping & trace

> **Status: COMPLETED 2026-06-09.** Spans `segmentation` (preflight policy) and `nnscope`
> (the generic trace/render mechanism). All changes verified end-to-end on CUDA + CPU.
>
> **As-built verification**
> - nnscope: `trace.py` doctests 10/10 (ephemeral **and** persistent orderings); full smoke
>   `ALL OK` (20 doctests).
> - segmentation gligen preflight: static audit `282 trainable / 751 frozen — PASS`; probe
>   `282 flowed, 0 zero, 0 none`; `MuggledSamSegmenter` (217M) `[frozen ok]`; HTML now carries
>   **40 injection badges each direction**.
> - `inspect_gligen.py`: `pairing: PASS`, **40/40** `[ok] … just ahead of attn pre-hook`, exit 0;
>   wrote `gligen_tree.html` + `gligen_torchvista.html`.
> - All generated `*.html` are gitignored — gone from `git status` / the porcelain scan.

## Context

Two gaps surfaced while inspecting the **persistent-gligen-injection** migration
(`persistent-gligen-injection-activation-checkpointing.md`):

1. **Preflight drift.** The standalone `run_gligen_preflight` built the model its *own* way
   (`GligenWrapper.from_config(load_gligen_wrapper("gligen_wrapper.yaml"))`) — a different config
   file *and* a different construct/freeze path than training (`LGligenWrapper` from
   `gligen_training.yaml`). A sanity check that builds the model differently from training can pass
   while training is broken.
2. **Injection invisible in the tree.** gligen blocks live in a flat `self.gligen_blocks`
   `nn.ModuleList`; the wiring to their target SAM attn layers lives only in `target_layers` (path
   strings) + persistent `register_forward_pre_hook`s. `named_modules()` doesn't traverse
   `_forward_pre_hooks`, so nnscope's tree showed the blocks floating with no link to where they act.
3. **Stale trace assertion.** Persistent hooks register at *construction*, **before**
   `trace_forward`'s markers, so the block now fires *ahead of* the attn's pre-marker instead of
   *nested inside* it. `assert_gligen_pairing` hard-coded the nested order → 40 false failures even
   though injection was correct.

## Changes — segmentation (`src/segmentation/preflight.py`, `.gitignore`)

**Build the model exactly as training does.** `run_gligen_preflight` now goes through `LGligenWrapper`:

```python
cfg = load_check_config("configs/gligen_training.yaml", LightningGligenConfig)
lmodule = LGligenWrapper(cfg).cuda().eval()          # same construct + freeze (.train() override) as train
return preflight_gligen_model(lmodule.model, optimizer=None, out_path=out_path)
```

The `on_fit_start` path was already drift-free (it preflights the real `self.model`); this fixes the
*standalone* path. (`os.chdir` to the package dir stays — `configs/` lives under `src/segmentation/`.)

**Show the injection wiring** via nnscope's existing `annotations` channel (persistent hooks made the
`target_layers ↔ gligen_blocks` pairing a stable, inspectable fact):

```python
def gligen_injection_annotations(wrapper) -> dict[str, str]:
    ann = {}
    for i, path in enumerate(wrapper.target_layers):
        ann[f"segmenter.{path}"]  = f"⟵ injected by gligen_blocks.{i}"   # on the frozen SAM attn node
        ann[f"gligen_blocks.{i}"] = f"⟶ injects into segmenter.{path}"   # on the trainable block node
    return ann
```

Threaded through a new `annotations: dict[str, str] | None = None` kwarg on `preflight_gradients`
→ `nnscope.render_html(root, …, annotations=annotations)`. Generic, so the verifier path is
unaffected (passes nothing → byte-identical output). nnscope auto-expands annotated nodes, so the deep
stage-2/3 attn layers are visible on load.

**Gitignore generated HTML.** Added `*.html` to `.gitignore`. The run gate never blocked on `.html`
(it allow-lists `.py/.yaml/.yml`); this just keeps the artifacts out of the working tree and the
porcelain scan, matching how `runs/`/`logs/` are handled.

## Changes — nnscope (`src/nnscope/trace.py`)

`assert_gligen_pairing` now accepts **both** orderings — the block-vs-marker order is a hook-
registration-order artifact, not a semantic property of the injection:

```python
nested       = a_pre[0] < g_pre[0] < g_post[0] < a_post[0]   # ephemeral hook (installed in forward)
pre_adjacent = g_pre[0] < g_post[0] < a_pre[0] < a_post[0]   # persistent hook (installed at construction)
if not (nested or pre_adjacent): ...fail "block did not run in attn's pre-hook window"
```

The misalignment window generalized from `(A_pre, A_post)` to `[min(G_pre, A_pre), A_post]` so it still
spans the block whether it nested inside A or fired just ahead of A's marker. Module + `trace_forward` +
check docstrings updated to explain the registration-order dependence; added a **persistent-hook
doctest** (`ToyPersistent`, hook registered in `__init__`) alongside the existing ephemeral one as the
executable spec.

## Key facts (do not relearn)

- **A hook is not a submodule.** Persistent injection did **not** re-parent the blocks; `named_modules()`
  still can't show injection points. The annotation channel is how you render them.
- **Registration order is everything for the trace.** Construction-time hooks fire before a later-
  registered tracer marker; forward-time hooks fire after. Both mean "G ran in A's pre-hook, before A's
  body" — `assert_gligen_pairing` accepts either.
- The static module tree and the torchvista forward-graph are **unchanged** by persistence; only the
  execution-nesting trace was affected.

## Critical files
- `segmentation/src/segmentation/preflight.py` (`run_gligen_preflight`, `gligen_injection_annotations`,
  `preflight_gradients` annotations kwarg, `preflight_gligen_model`)
- `segmentation/.gitignore` (`*.html`)
- `nnscope/src/nnscope/trace.py` (`assert_gligen_pairing`, docstrings, persistent doctest)
