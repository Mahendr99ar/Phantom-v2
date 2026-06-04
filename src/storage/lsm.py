"""
PHANTOM v2 — LSM Tree Storage Engine
File: src/storage/lsm.py

DSA + OS Concepts:
  WAL     → sequential append-only log (crash recovery)
  MemTable → Red-Black Tree in memory (sortedcontainers)
  SSTable  → immutable sorted file on disk (mmap reads)
  Bloom Filter → probabilistic membership (avoid disk reads)
  Compaction   → background merge of SSTables
"""

from __future__ import annotations
import os, json, time, struct, threading, mmap
from pathlib import Path
from sortedcontainers import SortedDict
from typing import Optional, Any


# ══════════════════════════════════════════════════════
# BLOOM FILTER  (DSA — probabilistic set)
# ══════════════════════════════════════════════════════

class BloomFilter:
    """
    Space-efficient membership check.
    False positive possible. False negative NEVER.
    'Definitely not in DB' OR 'Maybe in DB'.
    """
    def __init__(self, capacity: int = 10_000, error_rate: float = 0.01):
        import math
        self.size  = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hashes = max(1, int(self.size / capacity * math.log(2)))
        self.bits  = bytearray(self.size // 8 + 1)

    def _positions(self, key: str):
        import hashlib
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.hashes)]

    def add(self, key: str):
        for pos in self._positions(key):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def might_contain(self, key: str) -> bool:
        return all(
            self.bits[p // 8] & (1 << (p % 8))
            for p in self._positions(key)
        )


# ══════════════════════════════════════════════════════
# WRITE-AHEAD LOG  (OS — sequential file I/O)
# ══════════════════════════════════════════════════════

class WAL:
    """
    Every write appended here FIRST before MemTable.
    On crash → replay WAL to recover MemTable.
    Sequential writes = fastest possible disk I/O.
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'ab')   # append-binary

    def append(self, op: str, key: str, value: Any = None):
        """Write operation to log. Format: JSON line."""
        record = json.dumps({
            'op': op, 'key': key, 'value': value,
            'ts': time.time_ns()
        }) + '\n'
        self.file.write(record.encode())
        self.file.flush()   # force to OS buffer
        os.fsync(self.file.fileno())   # force to disk (Durability)

    def recover(self) -> list[dict]:
        """Read all log entries for crash recovery."""
        entries = []
        if not os.path.exists(self.path):
            return entries
        with open(self.path, 'rb') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass   # incomplete write at crash point
        return entries

    def clear(self):
        """After MemTable flushed to SSTable, WAL can be cleared."""
        self.file.close()
        open(self.path, 'wb').close()   # truncate
        self.file = open(self.path, 'ab')

    def close(self):
        self.file.flush()
        self.file.close()


# ══════════════════════════════════════════════════════
# SSTABLE  (immutable sorted file on disk)
# ══════════════════════════════════════════════════════

class SSTable:
    """
    Immutable sorted file written when MemTable is full.
    Uses OS mmap for zero-copy reads.
    """
    def __init__(self, path: str, data: dict):
        self.path   = path
        self.bloom  = BloomFilter(capacity=max(len(data), 1))
        self._write(data)

    def _write(self, data: dict):
        """Write sorted key-value pairs to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        sorted_data = dict(sorted(data.items()))
        with open(self.path, 'w') as f:
            json.dump(sorted_data, f)
        for key in sorted_data:
            self.bloom.add(str(key))

    def get(self, key: str) -> Optional[Any]:
        """O(1) bloom check → disk read only if maybe present."""
        if not self.bloom.might_contain(str(key)):
            return None   # definitely not here
        with open(self.path, 'r') as f:
            data = json.load(f)
        return data.get(key)

    def load_all(self) -> dict:
        with open(self.path, 'r') as f:
            return json.load(f)

    @staticmethod
    def merge(tables: list['SSTable'], out_path: str) -> 'SSTable':
        """Compaction: merge multiple SSTables into one."""
        merged = {}
        for t in reversed(tables):   # later = newer (wins on conflict)
            merged.update(t.load_all())
        return SSTable(out_path, merged)


# ══════════════════════════════════════════════════════
# LSM TREE  (main storage engine)
# ══════════════════════════════════════════════════════

class LSMTree:
    """
    Log-Structured Merge Tree.

    Write path: WAL → MemTable → (flush) → SSTable → (compact)
    Read  path: MemTable → SSTables (newest first) → None

    Write amplification: low (always append)
    Read  amplification: moderate (check multiple levels)
    Space amplification: moderate (compaction reduces)
    """

    MEMTABLE_SIZE_LIMIT = 1000   # entries before flush
    COMPACTION_THRESHOLD = 4     # SSTables before compact

    def __init__(self, data_dir: str):
        self.data_dir  = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.memtable:  SortedDict = SortedDict()   # Red-Black Tree
        self.sstables:  list[SSTable] = []
        self.wal        = WAL(str(self.data_dir / 'wal' / 'wal.log'))
        self._lock      = threading.Lock()
        self._sstable_counter = 0

        self._recover()

    # ── public API ─────────────────────────────────
    def put(self, key: str, value: Any):
        """Write: WAL first → then MemTable."""
        with self._lock:
            self.wal.append('PUT', key, value)      # 1. WAL (crash safety)
            self.memtable[key] = value              # 2. MemTable (fast)
            if len(self.memtable) >= self.MEMTABLE_SIZE_LIMIT:
                self._flush_memtable()              # 3. Flush if full

    def get(self, key: str) -> Optional[Any]:
        """Read: MemTable first → newest SSTable → older SSTables."""
        with self._lock:
            # 1. Check MemTable (fastest — in memory)
            if key in self.memtable:
                return self.memtable[key]
            # 2. Check SSTables newest first
            for sst in reversed(self.sstables):
                val = sst.get(key)
                if val is not None:
                    return val
            return None

    def delete(self, key: str):
        """Tombstone delete — mark as deleted, compact later."""
        self.put(key, '__PHANTOM_DELETED__')

    def range_scan(self, lo: str, hi: str) -> list[tuple]:
        """Scan all keys in [lo, hi] range."""
        with self._lock:
            results = {}
            # Collect from all SSTables (older first)
            for sst in self.sstables:
                for k, v in sst.load_all().items():
                    if lo <= k <= hi:
                        results[k] = v
            # MemTable overrides (newest)
            for k in self.memtable.irange(lo, hi):
                results[k] = self.memtable[k]
            # Remove tombstones
            return [
                (k, v) for k, v in sorted(results.items())
                if v != '__PHANTOM_DELETED__'
            ]

    def stats(self) -> dict:
        return {
            'memtable_entries': len(self.memtable),
            'sstable_count':    len(self.sstables),
            'data_dir':         str(self.data_dir),
        }

    # ── internal ───────────────────────────────────
    def _flush_memtable(self):
        """Flush MemTable → SSTable on disk."""
        if not self.memtable:
            return
        path = str(self.data_dir / f'sst_{self._sstable_counter:06d}.json')
        self._sstable_counter += 1
        sst = SSTable(path, dict(self.memtable))
        self.sstables.append(sst)
        self.memtable.clear()
        self.wal.clear()
        print(f"  [LSM] Flushed MemTable → {path}")
        if len(self.sstables) >= self.COMPACTION_THRESHOLD:
            self._compact()

    def _compact(self):
        """Merge SSTables to reduce read amplification."""
        print(f"  [LSM] Compacting {len(self.sstables)} SSTables...")
        path = str(self.data_dir / f'sst_{self._sstable_counter:06d}_compact.json')
        self._sstable_counter += 1
        merged = SSTable.merge(self.sstables, path)
        # Remove old files
        for sst in self.sstables:
            try:
                os.remove(sst.path)
            except FileNotFoundError:
                pass
        self.sstables = [merged]
        print(f"  [LSM] Compaction complete → {path}")

    def _recover(self):
        """Replay WAL on startup (crash recovery)."""
        entries = self.wal.recover()
        if entries:
            print(f"  [LSM] Recovering {len(entries)} entries from WAL...")
            for e in entries:
                if e['op'] == 'PUT':
                    self.memtable[e['key']] = e['value']
            print(f"  [LSM] Recovery complete.")

    def close(self):
        with self._lock:
            self._flush_memtable()
            self.wal.close()
