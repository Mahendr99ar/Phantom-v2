"""
PHANTOM v2 — Unit Tests
File: tests/test_phantom.py

Run: pytest tests/ -v
"""

import sys, os, shutil, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage.btree    import BPlusTree
from storage.lsm      import LSMTree, BloomFilter, WAL
from consensus.raft   import RaftCluster
from ai.learned_index import LearnedIndex


# ════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════

def fresh_lsm(path='./test_data_tmp'):
    if os.path.exists(path):
        shutil.rmtree(path)
    return LSMTree(path), path

def cleanup(path='./test_data_tmp'):
    shutil.rmtree(path, ignore_errors=True)


# ════════════════════════════════════════════════════
# 1. B+ TREE TESTS
# ════════════════════════════════════════════════════

class TestBPlusTree:

    def test_insert_and_search(self):
        t = BPlusTree()
        t.insert(10, 'ten')
        assert t.search(10) == 'ten'

    def test_search_missing_returns_none(self):
        t = BPlusTree()
        assert t.search(999) is None

    def test_update_existing_key(self):
        t = BPlusTree()
        t.insert(1, 'old')
        t.insert(1, 'new')
        assert t.search(1) == 'new'

    def test_delete(self):
        t = BPlusTree()
        t.insert(5, 'five')
        t.delete(5)
        assert t.search(5) is None

    def test_range_query(self):
        t = BPlusTree()
        for i in range(20):
            t.insert(i, f'v{i}')
        results = t.range_query(5, 10)
        keys = [k for k, _ in results]
        assert keys == [5, 6, 7, 8, 9, 10]

    def test_range_query_empty(self):
        t = BPlusTree()
        for i in range(5):
            t.insert(i, i)
        assert t.range_query(10, 20) == []

    def test_large_insert(self):
        t = BPlusTree()
        for i in range(1000):
            t.insert(i, f'val_{i}')
        assert len(t) == 1000
        assert t.search(500) == 'val_500'
        assert t.search(999) == 'val_999'

    def test_self_balancing(self):
        """Insert descending — tree must still find correctly."""
        t = BPlusTree()
        for i in range(100, 0, -1):
            t.insert(i, i * 2)
        for i in range(1, 101):
            assert t.search(i) == i * 2

    def test_size_tracking(self):
        t = BPlusTree()
        assert len(t) == 0
        t.insert('a', 1)
        t.insert('b', 2)
        assert len(t) == 2

    def test_string_keys(self):
        t = BPlusTree()
        t.insert('apple', 1)
        t.insert('banana', 2)
        t.insert('cherry', 3)
        assert t.search('banana') == 2


# ════════════════════════════════════════════════════
# 2. BLOOM FILTER TESTS
# ════════════════════════════════════════════════════

class TestBloomFilter:

    def test_added_key_found(self):
        bf = BloomFilter(capacity=100)
        bf.add('hello')
        assert bf.might_contain('hello') is True

    def test_absent_key_likely_false(self):
        bf = BloomFilter(capacity=1000)
        for i in range(100):
            bf.add(str(i))
        # Key definitely not added — should return False (no false negative)
        # Note: small false positive rate exists but very rare
        not_added = [bf.might_contain(f'zz_{i}') for i in range(50)]
        # At least 80% should be correctly False
        false_positive_rate = sum(not_added) / len(not_added)
        assert false_positive_rate < 0.2

    def test_multiple_keys(self):
        bf = BloomFilter(capacity=500)
        keys = [f'key_{i}' for i in range(100)]
        for k in keys:
            bf.add(k)
        for k in keys:
            assert bf.might_contain(k) is True

    def test_no_false_negatives(self):
        """Once added, ALWAYS found — no false negatives."""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        for i in range(500):
            bf.add(f'item_{i}')
        for i in range(500):
            assert bf.might_contain(f'item_{i}') is True


# ════════════════════════════════════════════════════
# 3. WAL TESTS
# ════════════════════════════════════════════════════

class TestWAL:

    def test_append_and_recover(self):
        path = './test_wal_tmp/wal.log'
        if os.path.exists('./test_wal_tmp'):
            shutil.rmtree('./test_wal_tmp')
        wal = WAL(path)
        wal.append('PUT', 'key1', 'val1')
        wal.append('PUT', 'key2', 'val2')
        wal.append('PUT', 'key3', 'val3')
        wal.close()

        wal2 = WAL(path)
        entries = wal2.recover()
        assert len(entries) == 3
        assert entries[0]['key'] == 'key1'
        assert entries[2]['value'] == 'val3'
        wal2.close()
        shutil.rmtree('./test_wal_tmp', ignore_errors=True)

    def test_clear_removes_entries(self):
        path = './test_wal_clear/wal.log'
        if os.path.exists('./test_wal_clear'):
            shutil.rmtree('./test_wal_clear')
        wal = WAL(path)
        wal.append('PUT', 'x', 1)
        wal.clear()
        entries = wal.recover()
        assert entries == []
        wal.close()
        shutil.rmtree('./test_wal_clear', ignore_errors=True)


# ════════════════════════════════════════════════════
# 4. LSM TREE TESTS
# ════════════════════════════════════════════════════

class TestLSMTree:

    def test_put_and_get(self):
        lsm, path = fresh_lsm('./test_lsm_1')
        lsm.put('k1', 'v1')
        assert lsm.get('k1') == 'v1'
        lsm.close()
        cleanup('./test_lsm_1')

    def test_get_missing_returns_none(self):
        lsm, path = fresh_lsm('./test_lsm_2')
        assert lsm.get('ghost') is None
        lsm.close()
        cleanup('./test_lsm_2')

    def test_overwrite(self):
        lsm, path = fresh_lsm('./test_lsm_3')
        lsm.put('k', 'old')
        lsm.put('k', 'new')
        assert lsm.get('k') == 'new'
        lsm.close()
        cleanup('./test_lsm_3')

    def test_delete_tombstone(self):
        lsm, path = fresh_lsm('./test_lsm_4')
        lsm.put('k', 'v')
        lsm.delete('k')
        # After delete, key should not appear in range scan
        results = lsm.range_scan('k', 'k')
        assert len(results) == 0
        lsm.close()
        cleanup('./test_lsm_4')

    def test_range_scan(self):
        lsm, path = fresh_lsm('./test_lsm_5')
        for i in range(10):
            lsm.put(str(i), f'v{i}')
        results = lsm.range_scan('2', '5')
        keys = [k for k, _ in results]
        assert '2' in keys
        assert '5' in keys
        lsm.close()
        cleanup('./test_lsm_5')

    def test_wal_crash_recovery(self):
        """Simulate crash: new LSMTree instance recovers from WAL."""
        path = './test_lsm_recovery'
        if os.path.exists(path):
            shutil.rmtree(path)
        lsm = LSMTree(path)
        lsm.put('recover_me', 'safe_value')
        # Do NOT call close() — simulate crash
        # New instance recovers from WAL
        lsm2 = LSMTree(path)
        val = lsm2.get('recover_me')
        assert val == 'safe_value', f"Recovery failed, got: {val}"
        lsm2.close()
        cleanup(path)

    def test_memtable_flush_on_large_insert(self):
        """Writing many keys should trigger MemTable flush to SSTable."""
        path = './test_lsm_flush'
        if os.path.exists(path):
            shutil.rmtree(path)
        lsm = LSMTree(path)
        lsm.MEMTABLE_SIZE_LIMIT = 50   # low threshold for test
        for i in range(200):
            lsm.put(str(i), f'v{i}')
        # Should have flushed at least once
        assert lsm.stats()['sstable_count'] >= 1
        # Data should still be accessible
        assert lsm.get('100') == 'v100'
        lsm.close()
        cleanup(path)


# ════════════════════════════════════════════════════
# 5. RAFT CONSENSUS TESTS
# ════════════════════════════════════════════════════

class TestRaft:

    def test_leader_elected(self):
        cluster = RaftCluster(['n1', 'n2', 'n3'])
        cluster.start_all()
        time.sleep(1.2)
        leaders = [n for n in cluster._nodes.values()
                   if n.role.value == 'LEADER']
        cluster.stop_all()
        assert len(leaders) == 1, f"Expected 1 leader, got {len(leaders)}"

    def test_only_one_leader(self):
        cluster = RaftCluster(['a', 'b', 'c'])
        cluster.start_all()
        time.sleep(1.5)
        leaders = [n for n in cluster._nodes.values()
                   if n.role.value == 'LEADER']
        cluster.stop_all()
        assert len(leaders) <= 1

    def test_leader_accepts_command(self):
        cluster = RaftCluster(['x1', 'x2', 'x3'])
        cluster.start_all()
        leader = cluster.get_leader()
        assert leader is not None
        ok = leader.submit_command({'op': 'PUT', 'key': 'test', 'value': '42'})
        assert ok is True
        cluster.stop_all()

    def test_log_replicated_to_majority(self):
        cluster = RaftCluster(['p1', 'p2', 'p3'])
        cluster.start_all()
        leader = cluster.get_leader()
        leader.submit_command({'op': 'PUT', 'key': 'x', 'value': '1'})
        leader.submit_command({'op': 'PUT', 'key': 'y', 'value': '2'})
        time.sleep(0.3)
        # At least 2 of 3 nodes should have log
        nodes_with_log = [
            n for n in cluster._nodes.values()
            if len(n.state.log) > 0
        ]
        cluster.stop_all()
        assert len(nodes_with_log) >= 2

    def test_cluster_survives_one_failure(self):
        """Cluster works with 2/3 nodes (majority = 2)."""
        cluster = RaftCluster(['q1', 'q2', 'q3'])
        cluster.start_all()
        leader = cluster.get_leader()
        assert leader is not None
        # Kill non-leader
        victims = [n for n in cluster._nodes.values()
                   if n.role.value != 'LEADER']
        cluster.kill_node(victims[0].node_id)
        time.sleep(0.3)
        # Leader should still accept commands
        ok = leader.submit_command({'op': 'PUT', 'key': 'survive', 'value': '1'})
        cluster.stop_all()
        assert ok is True


# ════════════════════════════════════════════════════
# 6. LEARNED INDEX TESTS
# ════════════════════════════════════════════════════

class TestLearnedIndex:

    def _make_trained(self, n=500):
        li = LearnedIndex()
        data = {float(i): f'v{i}' for i in range(n)}
        li.train(data, epochs=50)
        return li, data

    def test_trained_flag(self):
        li, _ = self._make_trained(100)
        assert li._trained is True

    def test_lookup_accuracy(self):
        li, data = self._make_trained(500)
        correct = sum(
            1 for k in range(0, 500, 5)
            if li.get(float(k)) == data[float(k)]
        )
        total = len(range(0, 500, 5))
        accuracy = correct / total
        assert accuracy >= 0.85, f"Accuracy too low: {accuracy:.2f}"

    def test_missing_key_returns_none(self):
        li, _ = self._make_trained(100)
        val = li.get(99999.0)
        assert val is None

    def test_untrained_returns_none(self):
        li = LearnedIndex()
        assert li.get(1.0) is None

    def test_error_bound_set(self):
        li, _ = self._make_trained(200)
        assert hasattr(li, '_max_error')
        assert li._max_error >= 0


# ════════════════════════════════════════════════════
# 7. INTEGRATION TEST
# ════════════════════════════════════════════════════

class TestIntegration:

    def test_btree_and_lsm_consistent(self):
        """Data written to LSM should match B+ Tree."""
        path = './test_integration'
        if os.path.exists(path):
            shutil.rmtree(path)

        lsm  = LSMTree(path)
        tree = BPlusTree()

        data = {str(i): f'val_{i}' for i in range(100)}
        for k, v in data.items():
            lsm.put(k, v)
            tree.insert(k, v)

        # Both should return same values
        for k, v in data.items():
            assert lsm.get(k)    == v
            assert tree.search(k) == v

        lsm.close()
        cleanup(path)

    def test_full_pipeline_write_read_delete(self):
        path = './test_pipeline'
        if os.path.exists(path):
            shutil.rmtree(path)

        lsm = LSMTree(path)
        lsm.put('alpha', 100)
        lsm.put('beta',  200)
        lsm.put('gamma', 300)

        assert lsm.get('alpha') == 100
        assert lsm.get('beta')  == 200

        lsm.delete('beta')
        results = lsm.range_scan('alpha', 'gamma')
        result_keys = [k for k, _ in results]
        assert 'beta' not in result_keys

        lsm.close()
        cleanup(path)
