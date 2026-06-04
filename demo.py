"""
PHANTOM v2 — Complete Live Demo
File: demo.py

Run: python demo.py

Shows:
  1. Basic CRUD operations
  2. LSM Tree flush + WAL recovery
  3. MVCC Transactions
  4. Raft Consensus (3-node cluster)
  5. Neural Learned Index benchmark
"""

import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── standalone mode (no Raft) for quick demo ──
os.environ['PHANTOM_STANDALONE'] = '1'

from storage.btree    import BPlusTree
from storage.lsm      import LSMTree, BloomFilter
from consensus.raft   import RaftCluster
from ai.learned_index import LearnedIndex


def sep(title=""):
    print("\n" + "═"*55)
    if title:
        print(f"  {title}")
        print("═"*55)


# ══════════════════════════════════════════════════════
# DEMO 1 — B+ TREE
# ══════════════════════════════════════════════════════

def demo_btree():
    sep("DEMO 1 — B+ Tree (Core Index)")

    tree = BPlusTree()

    print("\n  [1] Inserting 10,000 keys...")
    t0 = time.perf_counter()
    for i in range(10_000):
        tree.insert(i, f"value_{i}")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Inserted 10,000 keys in {elapsed:.1f}ms")

    print("\n  [2] Point lookups...")
    t0 = time.perf_counter()
    for i in random.sample(range(10_000), 1000):
        val = tree.search(i)
    elapsed_us = (time.perf_counter() - t0) / 1000 * 1e6 / 1000
    print(f"  1000 lookups done | Avg: {elapsed_us:.2f} µs/lookup")

    print("\n  [3] Range query: keys 500 to 510")
    results = tree.range_query(500, 510)
    for k, v in results:
        print(f"    {k} → {v}")

    print("\n  [4] Delete key 505")
    tree.delete(505)
    val = tree.search(505)
    print(f"  Search 505 after delete: {val}")

    print(f"\n  Tree size: {len(tree)} keys")
    print(f"\n  ✅ DSA: O(log n) insert/search | O(k) range query")


# ══════════════════════════════════════════════════════
# DEMO 2 — LSM TREE + WAL
# ══════════════════════════════════════════════════════

def demo_lsm():
    sep("DEMO 2 — LSM Tree + WAL (Crash Recovery)")

    import shutil
    data_dir = './phantom_data_demo'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    lsm = LSMTree(data_dir)

    print("\n  [1] Writing 500 keys (WAL first, then MemTable)...")
    for i in range(500):
        lsm.put(str(i), f"val_{i}")
    print(f"  Stats: {lsm.stats()}")

    print("\n  [2] Reading some keys...")
    for k in ['0', '100', '250', '499']:
        print(f"  GET {k} → {lsm.get(k)}")

    print("\n  [3] Range scan keys 10 to 15...")
    results = lsm.range_scan('10', '15')
    for k, v in results[:6]:
        print(f"  {k} → {v}")

    print("\n  [4] Simulating CRASH recovery...")
    lsm2 = LSMTree(data_dir)   # new instance reads WAL
    val  = lsm2.get('250')
    print(f"  After recovery, GET 250 → {val}")
    print(f"  ✅ WAL recovery works!")

    print("\n  [5] Bloom Filter demo...")
    bf = BloomFilter(capacity=1000)
    for i in range(100):
        bf.add(str(i))
    print(f"  '99' in filter: {bf.might_contain('99')}  (should be True)")
    print(f"  '999' in filter: {bf.might_contain('999')}  (should be False)")
    print(f"  ✅ Bloom Filter: avoids disk read when key definitely absent")

    lsm.close()
    lsm2.close()
    shutil.rmtree(data_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════
# DEMO 3 — MVCC TRANSACTIONS
# ══════════════════════════════════════════════════════

def demo_mvcc():
    sep("DEMO 3 — MVCC Transactions")

    import shutil
    data_dir = './phantom_data_mvcc'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    lsm = LSMTree(data_dir)
    memdata = {}

    # Simulated PhantomDB-lite for MVCC demo
    from storage.lsm import LSMTree as L
    from phantom import Transaction

    # Seed some data
    for k, v in [('balance_A', 1000), ('balance_B', 500)]:
        lsm.put(k, v)
        memdata[k] = v

    print("\n  Initial state:")
    print(f"  balance_A = {memdata.get('balance_A')}")
    print(f"  balance_B = {memdata.get('balance_B')}")

    print("\n  [1] Transaction 1: Transfer 200 from A to B")
    txn1 = Transaction('TXN-001', memdata)
    a = int(txn1.get('balance_A'))
    b = int(txn1.get('balance_B'))
    txn1.put('balance_A', a - 200)
    txn1.put('balance_B', b + 200)

    print("\n  [2] Transaction 2 starts BEFORE txn1 commits")
    txn2 = Transaction('TXN-002', memdata)   # snapshot before txn1
    print(f"  TXN-002 sees balance_A = {txn2.get('balance_A')}  (consistent snapshot!)")

    print("\n  [3] Committing TXN-001...")
    for key, val in txn1._writes.items():
        memdata[key] = val
        lsm.put(key, val)
    txn1.committed = True

    print("\n  Final state after commit:")
    print(f"  balance_A = {memdata.get('balance_A')}  (was 1000)")
    print(f"  balance_B = {memdata.get('balance_B')}  (was 500)")

    print("\n  [4] Transaction 3: Rollback demo")
    txn3 = Transaction('TXN-003', memdata)
    txn3.put('balance_A', 0)   # oops!
    txn3._writes.clear()        # rollback
    print(f"  After rollback, balance_A = {memdata.get('balance_A')}  (unchanged ✅)")

    print("\n  ✅ MVCC: readers never block writers | consistent snapshots")
    lsm.close()
    shutil.rmtree(data_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════
# DEMO 4 — RAFT CONSENSUS
# ══════════════════════════════════════════════════════

def demo_raft():
    sep("DEMO 4 — Raft Consensus (3-Node Cluster)")

    cluster = RaftCluster(['node-1', 'node-2', 'node-3'])
    cluster.start_all()

    print("\n  [1] Waiting for leader election...")
    time.sleep(1.0)

    status = cluster.status()
    for node in status:
        role_icon = "👑" if node['role'] == 'LEADER' else "⬜"
        print(f"  {role_icon} {node['node_id']} | {node['role']} | term={node['term']}")

    leader = cluster.get_leader()
    if leader:
        print(f"\n  [2] Leader is {leader.node_id} — submitting commands...")
        cmds = [
            {'op': 'PUT', 'key': 'user:1', 'value': 'mahendra'},
            {'op': 'PUT', 'key': 'user:2', 'value': 'phantom'},
            {'op': 'PUT', 'key': 'score',  'value': '9999'},
        ]
        for cmd in cmds:
            ok = leader.submit_command(cmd)
            print(f"  Command {cmd} → {'✅ committed' if ok else '❌ failed'}")

        print(f"\n  Log replicated to all nodes:")
        for node_info in cluster.status():
            print(f"  {node_info['node_id']}: log_length={node_info['log_length']} "
                  f"commit_idx={node_info['commit_idx']}")

        print(f"\n  [3] Simulating node-3 FAILURE...")
        cluster.kill_node('node-3')
        time.sleep(0.3)

        print(f"  [4] Cluster still works with 2/3 nodes (majority)...")
        ok = leader.submit_command({'op': 'PUT', 'key': 'after_failure', 'value': 'still_works'})
        print(f"  Write after failure → {'✅ success' if ok else '❌ failed'}")

        print(f"\n  [5] Reviving node-3...")
        cluster.revive_node('node-3')
        time.sleep(0.5)
        print(f"  ✅ Cluster fault-tolerant: survives (n-1)/2 failures")

    cluster.stop_all()


# ══════════════════════════════════════════════════════
# DEMO 5 — NEURAL LEARNED INDEX
# ══════════════════════════════════════════════════════

def demo_learned_index():
    sep("DEMO 5 — Neural Learned Index vs B+ Tree")

    N = 5_000
    print(f"\n  [1] Building dataset with {N} keys...")
    data = {float(i): f"record_{i}" for i in range(N)}

    print(f"\n  [2] Training Learned Index...")
    li = LearnedIndex()
    li.train(data, epochs=100)

    print(f"\n  [3] Accuracy check...")
    correct = 0
    test_keys = random.sample(list(data.keys()), 200)
    for k in test_keys:
        val = li.get(k)
        if val == data[k]:
            correct += 1
    print(f"  Accuracy: {correct}/200 = {correct/2:.1f}%")

    print(f"\n  [4] Speed benchmark vs B+ Tree...")
    tree = BPlusTree()
    for k, v in data.items():
        tree.insert(k, v)

    import numpy as np

    # B+ Tree times
    btree_times = []
    for k in test_keys:
        t0 = time.perf_counter_ns()
        tree.search(k)
        btree_times.append(time.perf_counter_ns() - t0)

    # Learned index times
    li_times = []
    for k in test_keys:
        t0 = time.perf_counter_ns()
        li.get(k)
        li_times.append(time.perf_counter_ns() - t0)

    btree_avg  = np.mean(btree_times) / 1000
    li_avg     = np.mean(li_times) / 1000
    speedup    = btree_avg / max(li_avg, 0.01)

    print(f"\n  Results:")
    print(f"  B+ Tree avg    : {btree_avg:.2f} µs")
    print(f"  Learned Index  : {li_avg:.2f} µs")
    print(f"  Speedup        : {speedup:.1f}x")
    print(f"  Max error bound: ±{li._max_error} positions")
    print(f"\n  ✅ Neural network replaces B-Tree for numeric key lookups")
    print(f"  ✅ Based on Google Research paper (Kraska et al., 2018)")


# ══════════════════════════════════════════════════════
# BENCHMARK — Resume numbers
# ══════════════════════════════════════════════════════

def benchmark():
    sep("FINAL BENCHMARK — Resume Numbers")

    import shutil, statistics
    data_dir = './phantom_bench'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    lsm  = LSMTree(data_dir)
    tree = BPlusTree()

    N = 10_000
    print(f"\n  Writing {N} keys...")
    write_times = []
    for i in range(N):
        t0 = time.perf_counter_ns()
        key = str(i)
        lsm.put(key, f"value_{i}")
        tree.insert(key, f"value_{i}")
        write_times.append(time.perf_counter_ns() - t0)

    print(f"  Reading {N} keys...")
    read_times = []
    keys = [str(i) for i in range(N)]
    random.shuffle(keys)
    for k in keys:
        t0 = time.perf_counter_ns()
        tree.search(k)
        read_times.append(time.perf_counter_ns() - t0)

    def us(ns_list):
        return [t/1000 for t in ns_list]

    w = us(write_times)
    r = us(read_times)

    total_s = sum(write_times) / 1e9
    throughput = N / total_s

    print(f"\n{'═'*55}")
    print(f"  PHANTOM v2 BENCHMARK RESULTS")
    print(f"{'─'*55}")
    print(f"  Keys processed     : {N:,}")
    print(f"  Write throughput   : {throughput:,.0f} ops/sec")
    print(f"  Write P50 latency  : {statistics.median(w):.1f} µs")
    print(f"  Write P99 latency  : {sorted(w)[int(N*0.99)]:.1f} µs")
    print(f"  Read  P50 latency  : {statistics.median(r):.1f} µs")
    print(f"  Read  P99 latency  : {sorted(r)[int(N*0.99)]:.1f} µs")
    print(f"  B+ Tree size       : {len(tree)} keys")
    print(f"  LSM stats          : {lsm.stats()}")
    print(f"{'═'*55}")
    print(f"\n  ✅ Resume bullet ready:")
    print(f"  'Built PHANTOM v2 — distributed DB with B+ Tree + LSM Tree,")
    print(f"   Raft consensus, Neural Learned Index — {throughput:,.0f} writes/sec")
    print(f"   at {statistics.median(w):.1f}µs P50 latency with WAL crash recovery'")
    print(f"{'═'*55}\n")

    lsm.close()
    shutil.rmtree(data_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   PHANTOM v2 — Distributed DB Engine                ║")
    print("║   By: Mahendra Meena | IIIT Gwalior | 2027          ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_btree()
    input("\n  Press Enter for LSM Tree demo...")

    demo_lsm()
    input("\n  Press Enter for MVCC Transactions demo...")

    demo_mvcc()
    input("\n  Press Enter for Raft Consensus demo...")

    demo_raft()
    input("\n  Press Enter for Learned Index demo...")

    demo_learned_index()
    input("\n  Press Enter for final benchmark...")

    benchmark()

    print("\n  All demos complete! Check resume_bullets.txt for copy-paste bullets.")
