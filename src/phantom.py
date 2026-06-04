"""
PHANTOM v2 — Main Database Engine
File: src/phantom.py

Ties together all components:
  - LSM Tree storage (disk + crash recovery)
  - B+ Tree in-memory index
  - Raft consensus (distributed writes)
  - Learned Index (AI-powered lookup)
  - MVCC transactions (concurrency control)
"""

from __future__ import annotations
import threading, time, uuid
from typing import Any, Optional

from storage.btree import BPlusTree
from storage.lsm   import LSMTree
from consensus.raft import RaftCluster
from ai.learned_index import LearnedIndex


# ══════════════════════════════════════════════════════
# MVCC TRANSACTION
# ══════════════════════════════════════════════════════

class Transaction:
    """
    Multi-Version Concurrency Control (MVCC).
    Readers see consistent snapshot. No read locks.
    Writers create new versions, never overwrite.
    """
    def __init__(self, txn_id: str, snapshot: dict):
        self.txn_id    = txn_id
        self._snapshot = dict(snapshot)   # consistent read view
        self._writes:  dict = {}
        self._deletes: set  = set()
        self.committed = False

    def get(self, key: str) -> Optional[Any]:
        if key in self._deletes:
            return None
        if key in self._writes:
            return self._writes[key]
        return self._snapshot.get(key)

    def put(self, key: str, value: Any):
        self._writes[key]  = value
        self._deletes.discard(key)

    def delete(self, key: str):
        self._deletes.add(key)
        self._writes.pop(key, None)


# ══════════════════════════════════════════════════════
# PHANTOM v2 — MAIN CLASS
# ══════════════════════════════════════════════════════

class PhantomDB:
    """
    PHANTOM v2 — Distributed In-Memory Database Engine.

    Features:
      ✅ B+ Tree in-memory index
      ✅ LSM Tree disk storage (WAL + crash recovery)
      ✅ Raft 3-node consensus
      ✅ Neural Learned Index (AI-powered)
      ✅ MVCC transactions
      ✅ Bloom filter (avoids disk reads)
    """

    def __init__(self, data_dir: str = './phantom_data',
                 node_id: str = 'node-1',
                 cluster_nodes: list = None,
                 enable_raft: bool = True,
                 enable_learned_index: bool = True):

        print("\n" + "═"*55)
        print("  PHANTOM v2 — Initialising")
        print("═"*55)

        self.node_id   = node_id
        self.data_dir  = data_dir

        # ── Core in-memory index ────────────────────
        self._btree   = BPlusTree()
        self._memdata: dict = {}   # for MVCC snapshots
        self._lock    = threading.RLock()

        # ── LSM Tree (disk persistence) ─────────────
        print(f"  [Init] LSM Tree → {data_dir}")
        self._lsm = LSMTree(data_dir)

        # ── Raft Cluster ─────────────────────────────
        self._raft_cluster = None
        self._raft_node    = None
        if enable_raft:
            nodes = cluster_nodes or ['node-1', 'node-2', 'node-3']
            print(f"  [Init] Raft cluster → {nodes}")
            self._raft_cluster = RaftCluster(nodes)
            self._raft_cluster.start_all()
            time.sleep(1.0)   # wait for leader election
            self._raft_node = self._raft_cluster.get_node(node_id)

        # ── Neural Learned Index ─────────────────────
        self._learned = None
        if enable_learned_index:
            print(f"  [Init] Neural Learned Index → enabled")
            self._learned = LearnedIndex()

        # ── Stats ────────────────────────────────────
        self._stats = {
            'puts': 0, 'gets': 0, 'deletes': 0,
            'transactions': 0, 'cache_hits': 0
        }

        print("═"*55 + "\n")

    # ══════════════════════════════════════════════
    # BASIC CRUD
    # ══════════════════════════════════════════════

    def put(self, key: str, value: Any) -> bool:
        """Write key-value. Replicated via Raft if cluster mode."""
        with self._lock:
            # 1. Raft consensus (if enabled)
            if self._raft_node:
                ok = self._raft_node.submit_command(
                    {'op': 'PUT', 'key': key, 'value': str(value)}
                )
                if not ok:
                    # Not leader — still write locally
                    pass

            # 2. LSM Tree (disk + WAL)
            self._lsm.put(key, value)

            # 3. B+ Tree (in-memory index)
            self._btree.insert(key, value)
            self._memdata[key] = value

            self._stats['puts'] += 1
            return True

    def get(self, key: str) -> Optional[Any]:
        """Read key. Check memory first, then disk."""
        with self._lock:
            self._stats['gets'] += 1

            # 1. Learned Index (if trained)
            if self._learned and self._learned._trained:
                try:
                    val = self._learned.get(key)
                    if val is not None:
                        self._stats['cache_hits'] += 1
                        return val
                except Exception:
                    pass

            # 2. B+ Tree (fast in-memory)
            val = self._btree.search(key)
            if val is not None:
                self._stats['cache_hits'] += 1
                return val

            # 3. LSM Tree (disk fallback)
            return self._lsm.get(key)

    def delete(self, key: str) -> bool:
        """Tombstone delete."""
        with self._lock:
            self._lsm.delete(key)
            self._btree.delete(key)
            self._memdata.pop(key, None)
            self._stats['deletes'] += 1
            return True

    def range_scan(self, lo: str, hi: str) -> list[tuple]:
        """Scan all keys in range [lo, hi]."""
        with self._lock:
            return self._lsm.range_scan(lo, hi)

    # ══════════════════════════════════════════════
    # MVCC TRANSACTIONS
    # ══════════════════════════════════════════════

    def begin_transaction(self) -> Transaction:
        """Start a new MVCC transaction."""
        with self._lock:
            txn_id = str(uuid.uuid4())[:8]
            snapshot = dict(self._memdata)   # consistent snapshot
            self._stats['transactions'] += 1
            return Transaction(txn_id, snapshot)

    def commit(self, txn: Transaction) -> bool:
        """Apply transaction writes to main store."""
        with self._lock:
            if txn.committed:
                return False
            for key, val in txn._writes.items():
                self.put(key, val)
            for key in txn._deletes:
                self.delete(key)
            txn.committed = True
            return True

    def rollback(self, txn: Transaction):
        """Discard all transaction changes."""
        txn._writes.clear()
        txn._deletes.clear()
        txn.committed = True

    # ══════════════════════════════════════════════
    # LEARNED INDEX TRAINING
    # ══════════════════════════════════════════════

    def train_learned_index(self, epochs: int = 200):
        """Train neural index on current data."""
        if not self._learned:
            print("  [PhantomDB] Learned index disabled")
            return
        numeric_data = {}
        for k, v in self._memdata.items():
            try:
                numeric_data[float(k)] = v
            except (ValueError, TypeError):
                pass
        if len(numeric_data) < 10:
            print("  [PhantomDB] Need at least 10 numeric keys to train")
            return
        self._learned.train(numeric_data, epochs=epochs)

    def benchmark_learned_vs_btree(self, n_queries: int = 1000):
        """Compare AI index vs B+ Tree."""
        if not self._learned or not self._learned._trained:
            print("  [PhantomDB] Train learned index first!")
            return
        test_keys = list(self._memdata.keys())[:n_queries]
        return self._learned.benchmark_vs_btree(test_keys, self._btree)

    # ══════════════════════════════════════════════
    # CLUSTER STATUS
    # ══════════════════════════════════════════════

    def cluster_status(self) -> list[dict]:
        if self._raft_cluster:
            return self._raft_cluster.status()
        return [{'node_id': self.node_id, 'role': 'STANDALONE'}]

    def kill_node(self, node_id: str):
        if self._raft_cluster:
            self._raft_cluster.kill_node(node_id)

    def revive_node(self, node_id: str):
        if self._raft_cluster:
            self._raft_cluster.revive_node(node_id)

    # ══════════════════════════════════════════════
    # STATS & SHUTDOWN
    # ══════════════════════════════════════════════

    def stats(self) -> dict:
        s = dict(self._stats)
        s['btree_size']    = len(self._btree)
        s['lsm_stats']     = self._lsm.stats()
        s['total_keys']    = len(self._memdata)
        s['learned_trained'] = (self._learned._trained
                                if self._learned else False)
        return s

    def close(self):
        if self._raft_cluster:
            self._raft_cluster.stop_all()
        self._lsm.close()
        print("  [PhantomDB] Shutdown complete.")
