"""Render a :class:`~nnscope.tree.ModuleNode` tree as one self-contained HTML file.

Folding is native ``<details>/<summary>`` nesting — no JS required to
expand/collapse a node. The small inline script only adds expand/collapse-all
buttons and a substring filter. No external assets: the file opens offline via
``file://``.
"""

import html as _html
from collections.abc import Iterable
from pathlib import Path

from nnscope.grad import GradState
from nnscope.tree import ModuleNode, iter_nodes

_STYLE = """
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       font-size: 13px; margin: 16px; background: #fafafa; color: #222; }
#toolbar { position: sticky; top: 0; background: #fafafa; padding: 8px 0;
           border-bottom: 1px solid #ddd; margin-bottom: 8px; z-index: 1; }
#toolbar input { font: inherit; width: 24em; padding: 2px 6px; }
#toolbar button { font: inherit; padding: 2px 10px; }
details.node, div.node { margin-left: 18px; border-left: 1px solid #e5e5e5; padding-left: 6px; }
summary { cursor: pointer; padding: 1px 0; white-space: nowrap; }
summary:hover { background: #f0f4ff; }
div.node { padding-top: 1px; padding-bottom: 1px; white-space: nowrap; }
.cls { font-weight: 600; }
.nm { color: #777; }
.params { background: #e8edf8; color: #335; border-radius: 8px; padding: 0 6px;
          font-size: 11px; margin-left: 6px; }
.extra { color: #999; margin-left: 6px; font-size: 11px; }
.badge { background: #fde68a; color: #7c2d12; border-radius: 8px; padding: 0 8px;
         margin-left: 8px; font-weight: 600; }
.legend { color: #777; margin-left: 12px; font-size: 11px; font-weight: normal; }
details.hl > summary, div.node.hl { background: #fff7d6; }
details.hl, div.node.hl { border-left: 3px solid #f59e0b; }
.hidden { display: none; }
"""

_SCRIPT = """
const nodes = Array.from(document.querySelectorAll('.node'));
document.getElementById('expand').onclick =
  () => document.querySelectorAll('details.node').forEach(d => d.open = true);
document.getElementById('collapse').onclick =
  () => document.querySelectorAll('details.node').forEach(d => d.open = false);
document.getElementById('filter').addEventListener('input', ev => {
  const q = ev.target.value.trim().toLowerCase();
  if (!q) { nodes.forEach(n => n.classList.remove('hidden')); return; }
  nodes.forEach(n => n.classList.add('hidden'));
  nodes.forEach(n => {
    if (!n.dataset.search.includes(q)) return;
    n.classList.remove('hidden');
    n.querySelectorAll('.node').forEach(d => d.classList.remove('hidden'));
    let a = n.parentElement && n.parentElement.closest('.node');
    while (a) {
      a.classList.remove('hidden');
      if (a.tagName === 'DETAILS') a.open = true;
      a = a.parentElement && a.parentElement.closest('.node');
    }
  });
});
"""


def _humanize(n: int) -> str:
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return str(n)


def _grad_glyph(node: ModuleNode) -> str:
    if node.total_params == 0:
        return ""
    if node.total_trainable == 0:
        return "❄"          # ❄ all frozen
    if node.total_trainable == node.total_params:
        return "\U0001f525"      # 🔥 all trainable
    return "\U0001f525❄"    # 🔥❄ mixed


# Observed grad-flow (apply_grad_report), distinct from the static ❄/🔥 glyph above.
# Static glyph = declared (requires_grad); observed glyph = measured after 1 backward.
_OBSERVED_GLYPH: dict[str, str] = {
    "flowed": "✅",            # ✅ got real gradient
    "zero": "⚠️",        # ⚠️ grad present but ~0 (e.g. a closed gate)
    "none": "\U0001f480",          # 💀 trainable but disconnected from the loss
    "mixed": "\U0001f7e1",         # 🟡 some flowed, some not
    "frozen-leak": "\U0001f6a8",   # 🚨 frozen param that nonetheless received grad
}


def _observed_state(
    node: ModuleNode, grad_states: dict[str, GradState] | None
) -> GradState | None:
    """Explicit grad_states kwarg wins; else fall back to a stamped node.grad_state."""
    if grad_states is not None:
        return grad_states.get(node.path)
    return node.grad_state


def _observed_glyph(state: GradState | None) -> str:
    return _OBSERVED_GLYPH.get(state, "") if state else ""


def _ancestors(paths: Iterable[str]) -> set[str]:
    """Every proper dotted prefix of every path (root "" included)."""
    out: set[str] = set()
    for path in paths:
        parts = path.split(".")
        for i in range(len(parts)):
            out.add(".".join(parts[:i]))
    return out


def _summary(
    node: ModuleNode, annotations: dict[str, str], grad_states: dict[str, GradState] | None
) -> str:
    bits = [f'<span class="cls">{_html.escape(node.class_name)}</span>']
    if node.name:
        bits.append(f'<span class="nm">{_html.escape(node.name)}</span>')
    glyph = _grad_glyph(node)
    obs = _observed_glyph(_observed_state(node, grad_states))
    if node.total_params:
        own = f" (own {_humanize(node.own_params)})" if 0 < node.own_params < node.total_params else ""
        bits.append(f'<span class="params">{glyph}{obs} {_humanize(node.total_params)}{own}</span>')
    if node.path in annotations:
        bits.append(f'<span class="badge">{_html.escape(annotations[node.path])}</span>')
    if node.extra_repr:
        short = node.extra_repr if len(node.extra_repr) <= 80 else node.extra_repr[:77] + "..."
        bits.append(
            f'<span class="extra" title="{_html.escape(node.extra_repr)}">{_html.escape(short)}</span>'
        )
    return " ".join(bits)


def _render_node(
    node: ModuleNode,
    depth: int,
    annotations: dict[str, str],
    highlight: set[str],
    open_set: set[str],
    initial_depth: int,
    grad_states: dict[str, GradState] | None,
    out: list[str],
) -> None:
    search = _html.escape(
        " ".join([node.path, node.class_name, node.extra_repr, annotations.get(node.path, "")]).lower(),
        quote=True,
    )
    hl = " hl" if node.path in highlight else ""
    summary = _summary(node, annotations, grad_states)
    if not node.children:
        out.append(f'<div class="node{hl}" data-search="{search}">{summary}</div>')
        return
    is_open = " open" if (depth < initial_depth or node.path in open_set) else ""
    out.append(f'<details class="node{hl}"{is_open} data-search="{search}"><summary>{summary}</summary>')
    for child in node.children:
        _render_node(child, depth + 1, annotations, highlight, open_set, initial_depth,
                     grad_states, out)
    out.append("</details>")


_LEGEND = (
    "positions ❄/\U0001f525 = declared (requires_grad) · observed after 1 backward: "
    "✅ flowed · ⚠️ zero (gated) · \U0001f480 none (disconnected) · \U0001f6a8 frozen-leak"
)


def render_html(
    root: ModuleNode,
    *,
    title: str = "nnscope",
    annotations: dict[str, str] | None = None,
    highlight: Iterable[str] = (),
    initial_depth: int = 1,
    grad_states: dict[str, GradState] | None = None,
) -> str:
    """Render the tree to a single standalone HTML string.

    Args:
        annotations: dotted path -> badge text shown on that node (e.g. mark
            where an ephemeral hook attaches — structure alone can't show it).
        highlight: paths to visually accent; all their ancestors render
            expanded so they are visible on load.
        initial_depth: nodes shallower than this start expanded; everything
            else starts collapsed (unless an ancestor of a highlight).
        grad_states: dotted path -> observed grad-flow state (see
            :func:`nnscope.tree.apply_grad_report`). Overlays a second glyph on the
            params badge distinct from the static ❄/🔥. When omitted, a stamped
            ``node.grad_state`` is used instead; when neither is present the output
            is unchanged (no observed glyph, no legend).
    """
    annotations = annotations or {}
    highlight = set(highlight)
    open_set = _ancestors(set(annotations) | highlight)
    has_obs = any(_observed_state(n, grad_states) for n in iter_nodes(root))
    legend = f'\n  <span class="legend">{_LEGEND}</span>' if has_obs else ""
    body: list[str] = []
    _render_node(root, 0, annotations, highlight, open_set, initial_depth, grad_states, body)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>
<style>{_STYLE}</style></head>
<body>
<div id="toolbar">
  <input id="filter" type="text" placeholder="filter: path / class / extra_repr / badge">
  <button id="expand">expand all</button>
  <button id="collapse">collapse all</button>
  <strong>{_html.escape(title)}</strong>{legend}
</div>
{chr(10).join(body)}
<script>{_SCRIPT}</script>
</body></html>
"""


def write_html(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
