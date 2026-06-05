# PHANTOM v2

I built this because I wanted to understand what actually happens inside a database — not just call `db.insert()` and move on. So I wrote one from scratch.

PHANTOM v2 is a distributed in-memory database engine. It does what MySQL or Redis does, but every single component — the index structure, the storage layer, the consensus protocol, the caching — is written by hand in Python.

**Mahendra Meena | IIIT Gwalior | B.Tech EEE 2027**

---

## What it can do

- Store and retrieve key-value data at **85,000+ writes per second**
- Survive a server crash and recover all data automatically (WAL)
- Run as a **3-node distributed cluster** — kill one node, the system keeps running
- Use a **neural network instead of a B+ Tree** for faster key lookups
- Handle multiple concurrent transactions without readers blocking writers (MVCC)

---

## How I built it — and why each piece exists

### B+ Tree — the index
I needed fast lookups. A plain array means scanning everything — O(n). A B+ Tree gives O(log n) for any key, and its leaf nodes form a linked list so range queries (`give me all keys between 500 and 600`) are fast too.

I wrote this from scratch — splitting, merging, self-balancing — without any library. It's in `src/storage/btree.py`.

### LSM Tree + Write-Ahead Log — the storage
The B+ Tree lives in memory. If the process dies, everything is gone. I needed persistence.

LSM Tree works like this: every write first goes to a log file on disk (WAL), then to an in-memory sorted structure (MemTable). When the MemTable fills up, it gets flushed to an immutable file on disk (SSTable). On startup, the WAL replays any writes that hadn't been flushed — crash recovery.

I also added a Bloom Filter so reads don't needlessly hit disk. If a key is definitely not in an SSTable, the Bloom Filter says so in O(1) — no disk read needed.

### Raft Consensus — the cluster
A single node is a single point of failure. I wanted the database to keep working even if one machine goes down.

Raft is the algorithm that makes this possible. Three nodes elect a leader. All writes go through the leader, which replicates them to the other two. As long as two of three nodes are alive, the cluster works. I implemented leader election, heartbeats, and log replication from scratch in `src/consensus/raft.py`.

### Neural Learned Index — the AI part
This one came from a 2018 Google Research paper — *"The Case for Learned Index Structures"* by Tim Kraska et al. The idea is: a B+ Tree is just a function that maps a key to a position in sorted data. A neural network can learn that function and often do it faster.

I trained a small 2-layer MLP (PyTorch) on the key distribution. At query time, the model predicts the approximate position, then a binary search within a small error window finds the exact value. On numeric keys, it runs 2–3x faster than the B+ Tree.

### MVCC — transactions
When two things read and write at the same time, you need to be careful. MVCC (Multi-Version Concurrency Control) gives every transaction a consistent snapshot of the data at the moment it started. Readers don't block writers. Writers don't block readers. This is how PostgreSQL handles concurrency.

---

## Numbers

| What | How much |
|------|----------|
| Write throughput | 85,000+ ops/sec |
| Write latency (P50) | ~11 µs |
| Read latency (P50) | ~4 µs |
| Learned Index speedup | 2–3x vs B+ Tree |
| Cluster size | 3 nodes |
| Tests | 35 passing |

---

## Running it

```bash
git clone https://github.com/Mahendr99ar/Phantom-v2.git
cd Phantom-v2

pip install -r requirements.txt

# See everything working
cd src
set PYTHONPATH=.
python ../demo.py

# Run tests
cd ..
set PYTHONPATH=src
python -m pytest tests/ -v
```

The demo walks through each component one by one — B+ Tree, LSM Tree with crash recovery, MVCC transactions, Raft consensus, and the Learned Index benchmark.

---

## Running the 3-node cluster with Docker

```bash
cd docker
docker-compose up --build
```

Three containers start up, elect a leader, and start accepting writes. Kill one:

```bash
docker stop phantom-node-3
```

The other two keep running. Revive it:

```bash
docker start phantom-node-3
```

It catches up automatically.

---

## Project layout

```
phantom-v2/
├── src/
│   ├── storage/
│   │   ├── btree.py         ← B+ Tree from scratch
│   │   └── lsm.py           ← WAL, MemTable, SSTable, Bloom Filter
│   ├── consensus/
│   │   └── raft.py          ← Raft leader election + log replication
│   ├── ai/
│   │   └── learned_index.py ← Neural index (PyTorch)
│   └── phantom.py           ← Puts it all together + MVCC
├── tests/
│   └── test_phantom.py      ← 35 tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   └── deployment.md
├── demo.py
└── requirements.txt
```

---

## What I learned building this

The thing that surprised me most was how much a Bloom Filter helps. Without it, every read that misses the MemTable has to check every SSTable on disk. With it, most misses are caught in memory in microseconds.

The Raft implementation was the hardest part — getting leader election to work correctly with random timeouts, and making sure log entries only commit once a majority of nodes acknowledge them.

The Learned Index was the most fun. The idea that you can replace a classical data structure with a neural network and get it to work faster felt genuinely surprising the first time it ran correctly.

---

## Author

**Mahendra Meena**
[LinkedIn](https://www.linkedin.com/in/mahendra-meena-72047b201/)
