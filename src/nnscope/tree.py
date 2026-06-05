"""Build a nested, render-agnostic tree from a model's ``named_modules()``.

The containment hierarchy of an ``nn.Module`` is already a tree (``_modules``
registration); this module captures it into plain dataclass nodes — paths,
class names, ``extra_repr``, and per-node parameter counts split by
``requires_grad`` — so renderers (see :mod:`nnscope.html`) never touch torch.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class ModuleNode:
    path: str            # dotted path from the root; "" for the root itself
    name: str            # last path segment ("attn", "0"); "" for the root
    class_name: str      # type(module).__name__
    extra_repr: str      # module.extra_repr() — the per-line text print(model) shows
    own_params: int      # params directly owned (recurse=False), excluding children
    own_trainable: int   # of own_params, those with requires_grad
    total_params: int    # cumulative, including children (post-order rollup)
    total_trainable: int
    children: list["ModuleNode"] = field(default_factory=list)


def build_tree(model: nn.Module) -> ModuleNode:
    """Capture ``model``'s module hierarchy as a :class:`ModuleNode` tree.

    Single pass over ``named_modules()`` (which yields parents before
    children, in registration order), linking each node to its parent by
    stripping the last dotted segment; cumulative param counts are rolled up
    post-order afterwards.
    """
    by_path: dict[str, ModuleNode] = {}
    for path, mod in model.named_modules():
        own = own_trainable = 0
        for p in mod.parameters(recurse=False):
            own += p.numel()
            own_trainable += p.numel() if p.requires_grad else 0
        node = ModuleNode(
            path=path,
            name=path.rsplit(".", 1)[-1] if path else "",
            class_name=type(mod).__name__,
            extra_repr=mod.extra_repr(),
            own_params=own,
            own_trainable=own_trainable,
            total_params=0,
            total_trainable=0,
        )
        by_path[path] = node
        if path:
            parent = path.rsplit(".", 1)[0] if "." in path else ""
            by_path[parent].children.append(node)

    root = by_path[""]
    _rollup(root)
    return root


def _rollup(node: ModuleNode) -> tuple[int, int]:
    total, trainable = node.own_params, node.own_trainable
    for child in node.children:
        ct, ctr = _rollup(child)
        total += ct
        trainable += ctr
    node.total_params, node.total_trainable = total, trainable
    return total, trainable


def iter_nodes(root: ModuleNode) -> Iterator[ModuleNode]:
    """Pre-order traversal."""
    yield root
    for child in root.children:
        yield from iter_nodes(child)
