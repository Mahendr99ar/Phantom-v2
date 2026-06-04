"""
PHANTOM v2 — Neural Learned Index
File: src/ai/learned_index.py

AI/ML Concepts:
  Based on: "The Case for Learned Index Structures" (Kraska et al., Google, 2018)
  Model: 2-layer MLP learns CDF of key distribution
  Goal: predict position of key in sorted array
  Result: 3x faster than B-Tree, 70x smaller index

How it works:
  1. Sort all keys
  2. Train MLP: input=key → output=position (0 to N)
  3. At query: predict position → binary search in error bounds
"""

from __future__ import annotations
import numpy as np
import json, os, time
from typing import Optional, Any
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  [LearnedIndex] PyTorch not found — using numpy fallback")


# ══════════════════════════════════════════════════════
# NEURAL NETWORK MODEL
# ══════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class IndexMLP(nn.Module):
        """
        2-layer MLP — learns CDF of key distribution.
        Input:  normalized key (float, 0-1)
        Output: predicted position in sorted array (float)
        """
        def __init__(self, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            return self.net(x)


# ══════════════════════════════════════════════════════
# NUMPY FALLBACK (when PyTorch not installed)
# ══════════════════════════════════════════════════════

class LinearLearnedIndex:
    """
    Fallback: piecewise linear model.
    Not as accurate as MLP but zero dependencies.
    """
    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0
        self.max_err = 0

    def fit(self, keys: np.ndarray, positions: np.ndarray):
        coeffs = np.polyfit(keys, positions, 1)
        self.slope, self.intercept = coeffs
        preds = np.polyval(coeffs, keys)
        errors = np.abs(preds - positions)
        self.max_err = int(np.max(errors)) + 1

    def predict(self, key: float) -> tuple[int, int]:
        pos = int(self.slope * key + self.intercept)
        lo  = max(0, pos - self.max_err)
        hi  = pos + self.max_err
        return lo, hi


# ══════════════════════════════════════════════════════
# LEARNED INDEX  (main class)
# ══════════════════════════════════════════════════════

class LearnedIndex:
    """
    Replaces B-Tree for numeric key lookups.

    Steps:
      1. train(keys, values) — fit model on sorted keys
      2. get(key)            — predict position → verify
      3. insert(key, value)  — retrain periodically

    Performance (typical):
      B-Tree:       O(log n) = ~14 ops for 10K keys
      LearnedIndex: O(1) predict + O(log error_bound)
                  = ~1 + ~5 ops (error bound << n)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.sorted_keys:   list  = []
        self.sorted_values: list  = []
        self.key_min:       float = 0.0
        self.key_max:       float = 1.0
        self._trained:      bool  = False
        self._model_path    = model_path

        if TORCH_AVAILABLE:
            self.model  = IndexMLP(hidden=256)
            self.opt    = optim.Adam(self.model.parameters(), lr=1e-3)
            self.loss_fn = nn.MSELoss()
        else:
            self.model = LinearLearnedIndex()

        self._lookup_times_btree:   list = []
        self._lookup_times_learned: list = []

    # ── training ───────────────────────────────────
    def train(self, data: dict, epochs: int = 200):
        """
        Train model on {key: value} dict.
        Keys must be numeric (int/float).
        """
        if not data:
            return

        pairs = sorted(data.items(), key=lambda x: x[0])
        self.sorted_keys   = [float(k) for k, _ in pairs]
        self.sorted_values = [v for _, v in pairs]

        keys = np.array(self.sorted_keys, dtype=np.float32)
        positions = np.arange(len(keys), dtype=np.float32)

        self.key_min = float(keys.min())
        self.key_max = float(keys.max())

        # Normalize keys to [0, 1]
        key_range = self.key_max - self.key_min + 1e-9
        keys_norm = (keys - self.key_min) / key_range

        t0 = time.perf_counter()

        if TORCH_AVAILABLE:
            X = torch.FloatTensor(keys_norm).unsqueeze(1)
            Y = torch.FloatTensor(positions).unsqueeze(1)

            self.model.train()
            for epoch in range(epochs):
                self.opt.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)
                loss.backward()
                self.opt.step()

            self.model.eval()
            with torch.no_grad():
                preds = self.model(X).squeeze().numpy()
            errors = np.abs(preds - positions)
            self._max_error = int(np.max(errors)) + 1

        else:
            self.model.fit(keys_norm, positions)
            self._max_error = self.model.max_err

        elapsed = (time.perf_counter() - t0) * 1000
        self._trained = True
        print(f"  [LearnedIdx] Trained on {len(data)} keys in {elapsed:.1f}ms "
              f"| Max error bound: ±{self._max_error}")

    # ── lookup ─────────────────────────────────────
    def get(self, key) -> Optional[Any]:
        """
        1. Predict position using neural network (O(1))
        2. Binary search within ±error bounds (O(log err))
        Total: much faster than B-Tree O(log n)
        """
        if not self._trained or not self.sorted_keys:
            return None

        t0 = time.perf_counter_ns()

        key_f = float(key)
        key_range = self.key_max - self.key_min + 1e-9
        key_norm  = (key_f - self.key_min) / key_range

        # Predict position
        if TORCH_AVAILABLE:
            with torch.no_grad():
                inp = torch.FloatTensor([[key_norm]])
                pred_pos = int(self.model(inp).item())
        else:
            lo_b, hi_b = self.model.predict(key_norm)
            pred_pos = (lo_b + hi_b) // 2

        # Binary search within error bounds
        lo = max(0, pred_pos - self._max_error)
        hi = min(len(self.sorted_keys) - 1, pred_pos + self._max_error)
        result = self._binary_search(key_f, lo, hi)

        elapsed = time.perf_counter_ns() - t0
        self._lookup_times_learned.append(elapsed)

        return result

    def _binary_search(self, key: float, lo: int, hi: int) -> Optional[Any]:
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_key = self.sorted_keys[mid]
            if mid_key == key:
                return self.sorted_values[mid]
            elif mid_key < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    # ── benchmark ──────────────────────────────────
    def benchmark_vs_btree(self, test_keys: list, btree) -> dict:
        """Compare LearnedIndex vs B+ Tree lookup speed."""
        print("\n  [Benchmark] LearnedIndex vs B+ Tree")
        print("  " + "─" * 40)

        # B+ Tree lookups
        btree_times = []
        for k in test_keys:
            t0 = time.perf_counter_ns()
            btree.search(k)
            btree_times.append(time.perf_counter_ns() - t0)

        # Learned Index lookups
        learned_times = []
        for k in test_keys:
            t0 = time.perf_counter_ns()
            self.get(k)
            learned_times.append(time.perf_counter_ns() - t0)

        btree_avg   = np.mean(btree_times) / 1000
        learned_avg = np.mean(learned_times) / 1000
        speedup     = btree_avg / max(learned_avg, 0.001)

        print(f"  B+ Tree avg    : {btree_avg:.2f} µs")
        print(f"  LearnedIdx avg : {learned_avg:.2f} µs")
        print(f"  Speedup        : {speedup:.1f}x faster")
        print(f"  Max error bound: ±{self._max_error} positions")
        print("  " + "─" * 40)

        return {
            'btree_us':    round(btree_avg, 2),
            'learned_us':  round(learned_avg, 2),
            'speedup':     round(speedup, 1),
            'error_bound': self._max_error,
        }

    def save(self, path: str):
        """Save model weights."""
        if not TORCH_AVAILABLE:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'sorted_keys': self.sorted_keys[:1000],   # sample
            'key_min': self.key_min,
            'key_max': self.key_max,
            'max_error': self._max_error,
        }, path)
        print(f"  [LearnedIdx] Model saved → {path}")
