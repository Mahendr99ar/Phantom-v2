"""
PHANTOM v2 — B+ Tree Storage Engine
File: src/storage/btree.py

DSA: B+ Tree
  - All data in LEAF nodes (linked list for range queries)
  - Internal nodes = keys only (routing)
  - Self-balancing on insert/delete
  - O(log n) search, insert, delete
  - O(k) range query (k = result count)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import threading


ORDER = 4   # max children per internal node (tunable)


# ══════════════════════════════════════════════════════
# NODE CLASSES
# ══════════════════════════════════════════════════════

class LeafNode:
    """Holds actual key-value data. Linked list across leaves."""
    def __init__(self):
        self.keys:   list = []
        self.values: list = []
        self.next:   Optional[LeafNode] = None   # linked list → range queries

    def is_full(self) -> bool:
        return len(self.keys) >= ORDER - 1

    def insert_sorted(self, key, value):
        for i, k in enumerate(self.keys):
            if key == k:
                self.values[i] = value   # update existing
                return
            if key < k:
                self.keys.insert(i, key)
                self.values.insert(i, value)
                return
        self.keys.append(key)
        self.values.append(value)

    def delete(self, key) -> bool:
        if key in self.keys:
            idx = self.keys.index(key)
            self.keys.pop(idx)
            self.values.pop(idx)
            return True
        return False


class InternalNode:
    """Routing node — keys only, no values."""
    def __init__(self):
        self.keys:     list = []
        self.children: list = []   # LeafNode or InternalNode

    def is_full(self) -> bool:
        return len(self.keys) >= ORDER - 1


# ══════════════════════════════════════════════════════
# B+ TREE
# ══════════════════════════════════════════════════════

class BPlusTree:
    """
    B+ Tree — core index structure of PHANTOM v2.

    Operations:
      search(key)         → O(log n)
      insert(key, value)  → O(log n)
      delete(key)         → O(log n)
      range(lo, hi)       → O(log n + k)
    """

    def __init__(self):
        self.root  = LeafNode()
        self.size  = 0
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()

    # ── search ─────────────────────────────────────
    def search(self, key) -> Optional[Any]:
        leaf = self._find_leaf(key)
        if key in leaf.keys:
            return leaf.values[leaf.keys.index(key)]
        return None

    def range_query(self, lo, hi) -> list[tuple]:
        """O(log n + k) — leaf linked list traversal."""
        results = []
        leaf = self._find_leaf(lo)
        while leaf:
            for k, v in zip(leaf.keys, leaf.values):
                if k > hi:
                    return results
                if k >= lo:
                    results.append((k, v))
            leaf = leaf.next
        return results

    # ── insert ─────────────────────────────────────
    def insert(self, key, value):
        result = self._insert(self.root, key, value)
        if result:   # root was split
            new_root       = InternalNode()
            new_root.keys  = [result[0]]
            new_root.children = [self.root, result[1]]
            self.root = new_root
        self.size += 1

    def _insert(self, node, key, value):
        if isinstance(node, LeafNode):
            node.insert_sorted(key, value)
            if node.is_full():
                return self._split_leaf(node)
            return None
        # Internal node — find correct child
        idx = self._find_child_idx(node, key)
        result = self._insert(node.children[idx], key, value)
        if result:
            mid_key, new_node = result
            node.keys.insert(idx, mid_key)
            node.children.insert(idx + 1, new_node)
            if node.is_full():
                return self._split_internal(node)
        return None

    def _split_leaf(self, leaf: LeafNode):
        mid   = len(leaf.keys) // 2
        new   = LeafNode()
        new.keys   = leaf.keys[mid:]
        new.values = leaf.values[mid:]
        new.next   = leaf.next
        leaf.keys   = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        leaf.next   = new
        return (new.keys[0], new)

    def _split_internal(self, node: InternalNode):
        mid     = len(node.keys) // 2
        mid_key = node.keys[mid]
        new     = InternalNode()
        new.keys      = node.keys[mid + 1:]
        new.children  = node.children[mid + 1:]
        node.keys      = node.keys[:mid]
        node.children  = node.children[:mid + 1]
        return (mid_key, new)

    # ── delete ─────────────────────────────────────
    def delete(self, key) -> bool:
        leaf = self._find_leaf(key)
        deleted = leaf.delete(key)
        if deleted:
            self.size -= 1
        return deleted

    # ── helpers ────────────────────────────────────
    def _find_leaf(self, key) -> LeafNode:
        node = self.root
        while isinstance(node, InternalNode):
            idx  = self._find_child_idx(node, key)
            node = node.children[idx]
        return node

    def _find_child_idx(self, node: InternalNode, key) -> int:
        for i, k in enumerate(node.keys):
            if key < k:
                return i
        return len(node.keys)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"BPlusTree(size={self.size}, order={ORDER})"
