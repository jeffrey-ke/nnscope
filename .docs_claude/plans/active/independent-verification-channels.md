# Plan: Independent Verification Channels for the GLIGEN-Insertion Inspection Toolchain

## Goal

Today every check in the GLIGEN inspection toolchain reads from a single source of truth — `GligenWrapper.target_layers`, the result of one `fnmatch` of `layer_pattern` (`"*hiera.stages.[23].*.attn"`, `segmentation/src/segmentation/configs/segmenter.yaml`) re-rendered three ways: the registration table, the HTML badges, and the trace watch-set (`segmentation/scripts/inspect_gligen.py:62-99`). These checks prove *internal consistency* (the wrapper does what it thinks) but not *intent* (that what it thinks equals "stages 2-3 hiera attentions and nothing else"). If the pattern matched the wrong modules, the table, badges, and pairing report would all PASS coherently.

This plan adds four **independent** evidence channels (A-D) that triangulate intent from sources that do *not* derive from `target_layers`, plus a final stage of small tooling follow-ups. Generic mechanisms land in `nnscope`; gligen-specific policy lands in the `segmentation` driver, preserving the existing mechanism/policy split (nnscope is a torch-only generic package; the driver owns all SAM/GLIGEN knowledge).

## Repos and entry points

- `nnscope` — `/home/jeffk/repo/nnscope`. Generic, torch-only (`pyproject.toml` dependencies = `["torch"]`). Public API re-exported from `src/nnscope/__init__.py`. Smoke test is script-style asserts, CPU-only: `env -u PYTHONPATH uv run python tests/smoke_nnscope.py` (run from `/home/jeffk/repo/nnscope`).
- `segmentation` — `/home/jeffk/repo/segmentation`, branch `min-working`. Consumes nnscope as an editable dev dep. Driver: `scripts/inspect_gligen.py`, run with `env -u PYTHONPATH uv run python scripts/inspect_gligen.py` (CUDA mandatory — `GatedMHA` forces fused SDPA, `src/segmentation/blocks.py:35`).

## Ground truth established by reading the sources

These facts anchor the plan; cite them when implementing.

**Attachment / execution mechanism**
- `GligenWrapper.target_layers` is built by `module_find(seg, layer_pattern)` (`src/segmentation/model.py:58`), and `module_find` matches **dotted path OR type name** (`src/segmentation/utils.py:49`). With the current pattern only paths match, but the type-name branch is the lever Channel A exploits.
- One `GligenBlock` is constructed per `target_layers` entry, index-aligned (`model.py:50-53`).
- Injection is ephemeral: `create_hooks` builds one pre-hook per `(path, gligen)` pair (`model.py:72-85`); `hook_into` installs them via `register_forward_pre_hook` and removes them in a `finally` (`model.py:94-105`). Structure alone can never show these — only an execution trace can.
- The gate: `GatedMHA.forward` returns `F.tanh(self.gamma) * self.drop(out)` with `self.gamma = nn.Parameter(torch.zeros(()))` (`blocks.py:15,41`) ⇒ **exact identity at init**: `GligenBlock.forward` returns `out + toks` where `out = tanh(0)*... = 0` (`blocks.py:51-58`). This is the lever for Channel B.
- The hook wraps tokens through `SpatialSeqToggle` (`model.py:77-83`): spatial `(B,h,w,C)` tokens get reshaped to sequence and back (`utils.py:68-98`). The reshape round-trip is exact.

**Real-run ground truth (already verified)**
- 40 matches: 36 in stage 2 + 4 in stage 3.
- Hidden dims (`GligenBlock.hidden_dim`, read from the attn's `qkv.in_features`, `blocks.py:78-86`): 288 at `stage 2 block 0`; 576 at `stage 2 blocks 1-35` and `stage 3 block 0`; 1152 at `stage 3 blocks 1-3`.
- All 40 pairs PASS the `assert_gligen_pairing` nesting check (`A_pre < G_pre < G_post < A_post`, `nnscope/src/nnscope/trace.py`).

**Why those widths (grounds Channel C's arithmetic)**
- `HieraModel.__init__` computes `features_per_stage = features_per_token_1st_stage * 2**stage_idx` and `heads_per_stage = num_heads_1st_stage * 2**stage_idx` (`modified-muggled-sam/muggled_sam/v2_sam/components/hiera_model.py:67-69`).
- For `sam2.1_hiera_large` the captured config is `features_per_image_token=144`, `imgencoder_heads=2`, `imgencoder_blocks_per_stage=(2, 6, 36, 4)` (the "sam-large" config, `make_sam_v2.py:170-176`). So stage widths are `144, 288, 576, 1152`.
- `HieraStage` makes its **first** block pooled when `stage_idx > 0`: `FirstBlockModule = PooledWindowedBlock` (`hiera_model.py:189-191`). A `PooledWindowedBlock` holds `attn = PooledSelfAttention(num_heads, input_features_per_token)` where `input_features_per_token = output_features_per_token // 2` (`hiera_blocks.py:168,176`), and `PooledSelfAttention.qkv = Linear(input_features_per_token, output*3)` (`hiera_blocks.py:325`) ⇒ its `qkv.in_features = stage_width // 2`. Non-first blocks are `WindowedBlock`/`GlobalBlock`, each `attn = SelfAttention(...)` with `qkv.in_features = stage_width` (`hiera_blocks.py:95,261`).
- Therefore: stage 2 (width 576) → block 0 = 288, blocks 1-35 = 576; stage 3 (width 1152) → block 0 = 576, blocks 1-3 = 1152. **This reproduces the verified 288/576/1152 layout from config arithmetic alone** — the basis for Channel C.

**Node paths (grounds badge and watch-set paths)**
- `MuggledSamSegmenter.__init__` sets `self.sam = core.get_interactive_context()` (`src/segmentation/segmenter.py:174`); that context exposes `image_encoder`, `coordinate_encoder`, `prompt_encoder`, `mask_decoder` (`muggled_sam/v2_sam/sam_v2_model.py:165-180`).
- Inside `GligenWrapper`, hiera attns are at `segmenter.sam.image_encoder.hiera.stages.<s>.<b>.attn`; `target_layers` entries (rooted at `seg`) are `sam.image_encoder.hiera.stages.<s>.<b>.attn`; the driver badges them under `f"segmenter.{p}"`.
- **Memory encoder / memory-image-fusion attentions (`RoPEAttention`, `RoPESelfAttention`, `RoPECrossAttention`, `memory_image_fusion_attention.py:19,124,145`) are NOT in the interactive context** — they will not appear in the wrapper tree. This bounds the blind-discovery watch-set (Channel A).

**Attention class inventory (grounds Channel A's by-type enumeration)**
- Hiera (image encoder): `SelfAttention` (`hiera_blocks.py:243`), `PooledSelfAttention` (`hiera_blocks.py:302`).
- Mask decoder: `GenericAttention` (`mask_decoder_attention.py:19`), `CrossAttentionNormed` (`:115`), `SelfAttentionNormed` (`:142`), `SelfAttentionNoPosenc` (`:163`, subclass of `SelfAttentionNormed`).
- Prompt / coordinate encoders: no attention classes.
- The clean-complement check (Channel A) is meaningful precisely because the watch-set includes the mask-decoder attentions: none of them should host a gligen guest.

**HTML annotation behavior (grounds Channel D)**
- `render_html` annotations are `dict[dotted_path, str]` (`nnscope/src/nnscope/html.py:133-151`). An annotation whose path is not a node in the tree is **silently dropped**: `_summary` only emits a badge when `node.path in annotations` (`html.py:98-99`); unmatched keys never raise. This is the silent-skip Channel D fixes.

**torchvista shim (grounds Item 4 named-outputs)**
- The driver's `_NamedArgs` shim exploits torchvista's dict-**input** branch (`tracer.py:1611-1640`) to name input nodes. There is a symmetric dict-**output** branch (`tracer.py:1645-1680`). `MuggledSamSegmenter.forward` returns a tuple `(mask_logits, iou_preds, SamDebugInfo)` (`segmenter.py:238-240`), and `SamDebugInfo` is a plain `@dataclass` (`segmenter.py:30-33`), not a dict — so the dict-output branch never fires today and outputs render as `output_0/1/2`.

**Git state (verify before the commit stage)**
- `nnscope` has one commit (`8d8daeb init commit`) and a clean working tree.
- `segmentation`: `model.py`, `scripts/inspect_gligen.py`, `pyproject.toml`, `uv.lock` are **already committed** (last commit `08da76e "verifier trains!"`). The only relevant uncommitted state is `M .gitignore` plus untracked `gligen_torchvista.html`, `gligen_tree.html`. The commit stage is mostly about the *new* work this plan produces. Re-run `git status` at execution time and adapt.

## Approach and design decisions

### Channel A — Blind discovery trace (generic mechanism → nnscope; policy → driver)

**Mechanism (nnscope).** Add a generic interval-analysis function to `src/nnscope/trace.py`:

```
def derive_attachments(
    result: TraceResult,
    hosts: Iterable[str],
    guests: Iterable[str],
) -> dict[str, str | None]:
    """For each guest label, the innermost host whose [pre, post] window encloses
    the guest's pre event; None if no host encloses it. Returns one entry per guest."""
```

Semantics (the event log is balanced parentheses because hooks fire in `Module.__call__` regardless of caller):
- A guest `g` is *attached to* host `h` iff `h_pre < g_pre < g_post < h_post` for some firing of each. The derived host is the **innermost** such host: among all enclosing host windows, pick the one with the largest `pre` seq (smallest window) — the module whose execution actually invoked the guest.
- **Guests that never fire** → map to `None` (caller asserts on this).
- **Modules that fire multiple times**: require host/guest fire counts of exactly 1 for a clean derivation; on multiplicity return `None` and let the policy layer detect the discrepancy (document this; never guess).
- Hosts with no guest in their window simply never appear as a value — the *clean-complement* assertion (every non-target host maps to no guest) is **driver policy**, computed by inverting the returned dict.

Keep `derive_attachments` independent of `assert_gligen_pairing`: A derives the map *from observation*; the existing pairing check validates a *declared* map. They must agree — that agreement is the triangulation.

Export `derive_attachments` from `src/nnscope/__init__.py`.

**Policy (driver, `scripts/inspect_gligen.py`).** Add a new section:
1. Enumerate the watch-set **by type, not by pattern**: walk `wrapper.named_modules()` and collect every module whose `type(mod).__name__` is in `{"SelfAttention", "PooledSelfAttention", "GenericAttention", "CrossAttentionNormed", "SelfAttentionNormed", "SelfAttentionNoPosenc"}`, labeled by its real dotted path. Also watch all `gligen_blocks[i]` labeled `gligen_blocks.<i>`. Define the class-name set as a module-level constant in the driver with a comment pointing at the muggled-sam component files.
2. Run one traced forward over this blind watch-set (reuse the dummy inputs; or fold into the existing trace by widening the watch dict).
3. Call `derive_attachments(result, hosts=<all attn paths>, guests=<all gligen paths>)`.
4. **Triangulate against the declared map**: assert derived == `{f"gligen_blocks.{i}": f"segmenter.{target_layers[i]}"}`.
5. **Clean complement**: assert every attention path NOT in the declared host set maps to no guest (no gligen block fired inside any mask-decoder attn or any non-target hiera attn). This check currently does not exist anywhere.
6. Print a verdict; contribute to the process exit code.

This proves *intent*: the watch-set is derived by type from the live module graph, never from `layer_pattern`.

**Smoke-test coverage (nnscope, `tests/smoke_nnscope.py`).** Add `test_derive_attachments()` using a toy in the spirit of `ToyHost` but richer: a host with multiple candidate attn-like submodules where only some carry an ephemeral insert hook, plus a non-hosting "decoy" sibling. Assert: positive map; decoy hosts clean; never-fired guest → `None`; **NEGATIVE** — `derived != mis_declared` for a deliberately wrong declaration.

### Channel B — Causality probe (driver-side only)

gamma=0 ⇒ wrapper output should equal the hook-free segmenter run:
- The gligen contribution is exactly `0 + toks` (gamma=0), and `SpatialSeqToggle` is an exact reshape round-trip. The tensor handed onward is value-identical to the un-hooked input; the segmenter then runs the same kernels in both cases.
- The reshape round-trip can produce different strides/contiguity, which *can* perturb a downstream fused kernel by ULPs. **Decision:** assert `torch.equal` (bit-identical) first; if it ever fails in practice, fall back to a tight `allclose` and log the max abs diff. Start strict — bit-identity is the stronger claim and is *expected*.

Comparison policy:
1. Forward A: `logits_h, iou_h, _ = wrapper(img, pts, ref)` (hooked, gamma=0 from init).
2. Forward B: hook-free baseline. **Decision: call `wrapper.segmenter(img, pts)` directly rather than implementing the `BaselineSam` stub** (`model.py:107-117`). The segmenter is the identical module instance the wrapper hooks — the strongest possible control; a second construction path could itself diverge. `BaselineSam` is left unimplemented intentionally.
3. Assert logits/iou equality per the policy above — proves the gligen path is wired in *and* correctly identity-at-init.
4. **Perturbation:** under `torch.no_grad()`, `wrapper.gligen_blocks[0].gated_mha.gamma.fill_(1.0)`, run forward C, assert the output **differs** from baseline (`not torch.equal` plus a non-trivial max abs diff to avoid a vacuous pass). Restore the saved original in `try/finally` — proves the hook is on the *causal* path.

Cheap: 2-3 extra forwards reusing the driver's dummy inputs, all `no_grad` + `.eval()`.

### Channel C — Spec-derived expectation (driver-side only)

Derive the EXPECTED hooked-attn list and per-block widths from the captured SAM config, fully independently of `layer_pattern`:
1. Read `wrapper.segmenter._sam_config` (captured at `segmenter.py:169/:173`; also via `hermetic_args()["sam_config"]`, `segmenter.py:179-190`). Extract `features_per_image_token`, `imgencoder_blocks_per_stage`.
2. Reconstruct expectations with the model's own arithmetic (`hiera_model.py:67-74`, `hiera_blocks.py:168,176,261,325`):
   - `stage_width[s] = features_per_image_token * 2**s`.
   - attn width = `stage_width[s] // 2` if `s > 0 and b == 0` (pooled first block), else `stage_width[s]`.
   - Eligible hooked set under the "stages 2-3" intent: `{(s, b) for s in (2, 3) for b in range(blocks_per_stage[s])}` with expected paths `sam.image_encoder.hiera.stages.<s>.<b>.attn`.
3. Assert against the live wrapper: count == 36+4; multiset of `gligen_blocks[i].hidden_dim` == spec-derived widths (288/576/1152 layout); `set(target_layers)` == spec-derived path set (catches a pattern that matched extra/wrong modules even at the right count).
4. Print verdict, contribute to exit code.

Edge note: `_sam_config` is built from `locals()` in `make_sam_v2` (`make_sam_v2.py:217`) on reload, or from `make_samv2_from_original_state_dict` on the train path. Read defensively; assert the needed keys exist with a clear error.

### Channel D — Fail-loud badges (mechanism → nnscope; policy → driver)

**Mechanism (nnscope, `src/nnscope/html.py`).** Add `strict: bool = False` to `render_html`. When `strict=True`, assert `set(annotations) <= {n.path for n in iter_nodes(root)}` before rendering, raising `ValueError` listing the unknown paths. Default `False` preserves silent-skip for existing callers.

**Policy (driver).** Pass `strict=True` from `inspect_gligen.py` — a typo or path-prefix drift in badge construction becomes a loud failure instead of a blank tree.

**Smoke-test coverage (nnscope).** Extend `test_html()`: strict + valid path succeeds; strict + bogus path raises `ValueError`; non-strict + bogus path still renders silently (back-compat).

### Item 4 — Tooling follow-ups (final stage)

1. **.gitignore for HTML artifacts.** The two HTML outputs are written to the repo root (`OUT_DIR`, `inspect_gligen.py`). `.gitignore` is already modified in the working tree — verify the pending modification before editing. Add `gligen_*.html` under a comment matching the file's existing style.

2. **Named torchvista outputs.** Extend `_NamedArgs.forward` to return a **dict**, triggering torchvista's dict-output branch (`tracer.py:1645-1680`) → nodes `output_logits` / `output_iou_preds` / etc. **`db_info` decision:** `SamDebugInfo` is a dataclass with tensor fields `images_proc`, `encoded_prompts` (`segmenter.py:30-33`); torchvista's `extract_tensors_from_obj` likely does not traverse dataclasses — flatten explicitly: `{"logits": ..., "iou_preds": ..., "db_images_proc": db_info.images_proc, "db_encoded_prompts": db_info.encoded_prompts}`. Export-cosmetic only; stays inside the existing best-effort try/except.

3. **Optional "ghost subtree" badges (backward-compatible richer annotation).** Accept `annotations: dict[str, str | Annotation]` where `Annotation(badge: str, detail: str | None = None)` is a small dataclass; plain strings normalize to `Annotation(badge=s)` — existing callers unchanged. Render `detail` as a muted inline span (reuse `.extra` CSS style). Driver policy: `detail = f"GligenBlock hidden_dim={g.hidden_dim}, gamma={float(g.gated_mha.gamma):.3g}"` per badged attn — a one-line ghost of the block that lives outside the tree. Smoke-test both forms.

4. **Commits (requires explicit user confirmation before executing).** Re-verify `git status` in both repos first. Proposed structure:
   - **nnscope**: one commit adding `derive_attachments` (trace.py + export), the `strict` kwarg and `Annotation` value type (html.py), and the expanded smoke test.
   - **segmentation** (branch `min-working`): one commit for the driver changes (Channels A-D policy + named torchvista outputs + ghost details), one for the `.gitignore` update; add the plan docs under `.docs_claude/plans/`. Do **not** commit `gligen_*.html` (they become ignored).
   - **Do not push or commit without explicit user confirmation.**

## Sequencing rationale

- **nnscope mechanisms first** (Channels A and D mechanism + `Annotation` type), validated by the CPU-only smoke test — pure, dependency-light, no CUDA needed. De-risks the generic layer before any CUDA driver work.
- **Driver policy second** (Channels A-D policy, Item-4 torchvista/ghost), which depends on the new nnscope API and requires CUDA end-to-end.
- **Tooling/.gitignore and commits last.**

A-D are independent of each other; A and D have nnscope mechanism prerequisites, so those land up front.

## Staged checklist

**Stage 0 — Re-confirm state (no edits)**
- [ ] `git -C /home/jeffk/repo/nnscope status` and `git -C /home/jeffk/repo/segmentation status` — confirm the git facts above still hold; adapt the commit stage if drifted.
- [ ] Confirm CUDA available (`torch.cuda.is_available()`), required for any wrapper forward.

**Stage 1 — nnscope mechanisms (CPU, testable)**
- [ ] `src/nnscope/trace.py`: add `derive_attachments(result, hosts, guests) -> dict[str, str | None]` with innermost-enclosing-interval semantics and the multiplicity/never-fired rules; add a doctest mirroring the existing `trace_forward` toy.
- [ ] `src/nnscope/html.py`: add `strict: bool = False`; raise `ValueError` listing unknown annotation paths when strict.
- [ ] `src/nnscope/html.py`: add `Annotation(badge, detail=None)`; accept `dict[str, str | Annotation]`; normalize in `_summary`; render `detail` inline; keep `dict[str, str]` fully working.
- [ ] `src/nnscope/__init__.py`: export `derive_attachments` and `Annotation`.
- [ ] `tests/smoke_nnscope.py`: add `test_derive_attachments` (positive map, clean complement, never-fired→None, NEGATIVE mis-declared map fails); extend `test_html` for strict success/failure + back-compat + `Annotation` detail rendering; wire into `__main__`.
- [ ] Run `env -u PYTHONPATH uv run python tests/smoke_nnscope.py` from `/home/jeffk/repo/nnscope`; expect `ALL OK`.

**Stage 2 — Driver: Channel C (spec-derived, cheap, no extra forward)**
- [ ] Read `wrapper.segmenter._sam_config`, reconstruct expected stage-2/3 attn paths and widths, assert count=40, the 288/576/1152 width multiset, and exact path-set equality with `target_layers`. Contribute to exit code.

**Stage 3 — Driver: Channel A (blind discovery)**
- [ ] Define the attention-class-name constant (with file citations); enumerate watch-set by type from `wrapper.named_modules()` + gligen blocks by real path.
- [ ] Traced forward; `derive_attachments`; assert derived == declared map AND clean complement. Contribute to exit code.

**Stage 4 — Driver: Channel B (causality)**
- [ ] Forward equality: `wrapper(...)` vs `wrapper.segmenter(img, pts)` with `torch.equal` (documented fallback).
- [ ] Perturb `gligen_blocks[0].gated_mha.gamma` to 1.0 under `no_grad`, assert output changes (non-trivial diff), restore in `finally`. Contribute to exit code.

**Stage 5 — Driver: Channel D policy**
- [ ] Pass `strict=True` to the `render_html` call.

**Stage 6 — Item 4 tooling**
- [ ] `.gitignore`: add `gligen_*.html`; confirm the two artifacts become ignored.
- [ ] `_NamedArgs`: return a named-output dict (flattening `SamDebugInfo`); keep inside best-effort try/except.
- [ ] Ghost-subtree badges: build `Annotation(badge=..., detail=...)` per attn from its paired gligen block.

**Stage 7 — End-to-end + commits (commits need user confirmation)**
- [ ] Run the driver; expect all four channels PASS, exit 0; `gligen_tree.html` shows badges + ghost details; `gligen_torchvista.html` shows named outputs.
- [ ] Re-run nnscope smoke test (regression).
- [ ] Re-check `git status`; prepare the commit structure; **execute only after explicit user confirmation.**

## Decision log

- **D1 — `derive_attachments` returns innermost host; multi-fire/never-fired → `None`.** Innermost enclosing window is the module whose execution actually invoked the guest; conservatively refusing to guess on multi-fire keeps the mechanism honest and pushes discrepancy detection to the policy layer.
- **D2 — Triangulation is `derived == declared` + clean complement, not a rewrite of `assert_gligen_pairing`.** A observes from the trace; the existing pairing check validates the declaration; their agreement is the evidence. Keeps the two checks independent.
- **D3 — Channel B uses `wrapper.segmenter` directly, NOT the `BaselineSam` stub.** Same module instance = strongest control; a second construction path could itself diverge. `BaselineSam` (`model.py:107-117`) left unimplemented intentionally.
- **D4 — Channel B asserts bit-identity (`torch.equal`) first, documented `allclose` fallback.** gamma=0 gives exact `0 + toks` and `SpatialSeqToggle` is an exact reshape round-trip, so bit-identity is the expected and most informative claim.
- **D5 — Channel C reconstructs targets from `_sam_config` arithmetic, never from `layer_pattern`.** Widths from `features_per_image_token * 2**s` and the pooled-first-block halving — independence that lets C stand alone against pattern drift.
- **D6 — Channel A watch-set is by type name, never by pattern.** Enumerates directly from the live graph; the attention-class set is a documented constant pointing at the muggled-sam component files.
- **D7 — Strict badges: mechanism (`strict` kwarg) in nnscope, policy (`strict=True`) in driver.** Default stays `False` for back-compat with the silent-skip contract.
- **D8 — Richer annotations via `str | Annotation` union, default-compatible.** Plain strings normalize to `Annotation(badge=s)`.
- **D9 — torchvista named outputs flatten `SamDebugInfo` into the output dict.** Dataclass tensors need explicit hoisting to become named nodes; export-cosmetic only, stays best-effort.
- **D10 — Git facts verified at planning time** (nnscope: 1 commit; segmentation driver/pyproject/uv.lock already committed at `08da76e`). Re-verify at execution; never push/commit without explicit user confirmation.

## Risks and mitigations

- **CUDA-only execution (R1):** Channels A/B and the full driver need a GPU + SAM weights. nnscope mechanisms are fully covered by the CPU smoke test, so the generic layer is verifiable without GPU.
- **Reshape stride bit-identity (R2):** see D4 — strict first, documented fallback.
- **`_sam_config` schema (R3):** read defensively, assert required keys with a clear message.
- **torchvista dataclass traversal (R4):** confirm `extract_tensors_from_obj` behavior on dataclasses before relying on it; flatten explicitly (D9); the block is best-effort regardless.
- **gamma restore (R5):** save-and-restore in `try/finally` so a failed assertion never leaves a perturbed parameter behind.
