"""
PHANTOM v2 — Raft Consensus Protocol
File: src/consensus/raft.py

System Design Concepts:
  Leader Election   → random timeout, majority voting
  Log Replication   → AppendEntries RPC, commit on majority
  Fault Tolerance   → cluster survives (n-1)/2 node failures
  Heartbeat         → leader sends every 150ms
"""

from __future__ import annotations
import threading, time, random, json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ══════════════════════════════════════════════════════
# RAFT STATE MACHINE
# ══════════════════════════════════════════════════════

class Role(Enum):
    FOLLOWER  = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER    = "LEADER"


@dataclass
class LogEntry:
    term:    int
    index:   int
    command: dict   # {'op': 'PUT', 'key': 'x', 'value': 42}


@dataclass
class RaftState:
    node_id:      str
    current_term: int = 0
    voted_for:    Optional[str] = None
    role:         Role = Role.FOLLOWER
    leader_id:    Optional[str] = None
    log:          list = field(default_factory=list)
    commit_index: int  = 0
    last_applied: int  = 0
    votes_received: set = field(default_factory=set)


class RaftNode:
    """
    Simplified Raft node for PHANTOM cluster.

    In production: nodes communicate over TCP/gRPC.
    Here: in-process message passing (simulation).
    Can be extended to real sockets easily.
    """

    HEARTBEAT_MS   = 150
    ELECTION_MIN   = 300
    ELECTION_MAX   = 600

    def __init__(self, node_id: str, peers: list[str],
                 cluster: 'RaftCluster'):
        self.state   = RaftState(node_id=node_id)
        self.peers   = peers
        self.cluster = cluster
        self._lock   = threading.Lock()
        self._timer  = None
        self._running = False
        self._applied_commands: list[dict] = []

    @property
    def node_id(self): return self.state.node_id

    @property
    def role(self): return self.state.role

    # ── lifecycle ──────────────────────────────────
    def start(self):
        self._running = True
        self._reset_election_timer()
        print(f"  [Raft] Node {self.node_id} started as FOLLOWER")

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()

    # ── election timer ─────────────────────────────
    def _reset_election_timer(self):
        if self._timer:
            self._timer.cancel()
        if not self._running:
            return
        timeout = random.randint(self.ELECTION_MIN, self.ELECTION_MAX) / 1000
        self._timer = threading.Timer(timeout, self._start_election)
        self._timer.daemon = True
        self._timer.start()

    def _start_election(self):
        with self._lock:
            if not self._running:
                return
            self.state.current_term += 1
            self.state.role      = Role.CANDIDATE
            self.state.voted_for = self.node_id
            self.state.votes_received = {self.node_id}
            term = self.state.current_term

        print(f"  [Raft] {self.node_id} starting election for term {term}")

        # Request votes from all peers
        for peer_id in self.peers:
            threading.Thread(
                target=self._request_vote,
                args=(peer_id, term),
                daemon=True
            ).start()

        self._reset_election_timer()

    def _request_vote(self, peer_id: str, term: int):
        peer = self.cluster.get_node(peer_id)
        if not peer:
            return
        granted = peer.handle_vote_request(
            candidate_id=self.node_id,
            term=term,
            last_log_index=len(self.state.log),
            last_log_term=self.state.log[-1].term if self.state.log else 0
        )
        if granted:
            with self._lock:
                self.state.votes_received.add(peer_id)
                total_nodes = len(self.peers) + 1
                majority    = total_nodes // 2 + 1
                if (len(self.state.votes_received) >= majority
                        and self.state.role == Role.CANDIDATE
                        and self.state.current_term == term):
                    self._become_leader()

    def _become_leader(self):
        self.state.role      = Role.LEADER
        self.state.leader_id = self.node_id
        if self._timer:
            self._timer.cancel()
        print(f"  [Raft] ✅ {self.node_id} became LEADER for term {self.state.current_term}")
        self._send_heartbeats()

    def _send_heartbeats(self):
        if not self._running or self.state.role != Role.LEADER:
            return
        for peer_id in self.peers:
            peer = self.cluster.get_node(peer_id)
            if peer:
                peer.handle_append_entries(
                    leader_id=self.node_id,
                    term=self.state.current_term,
                    entries=[],   # empty = heartbeat
                    commit_index=self.state.commit_index
                )
        # Schedule next heartbeat
        t = threading.Timer(self.HEARTBEAT_MS / 1000, self._send_heartbeats)
        t.daemon = True
        t.start()

    # ── RPC handlers ───────────────────────────────
    def handle_vote_request(self, candidate_id: str, term: int,
                            last_log_index: int, last_log_term: int) -> bool:
        with self._lock:
            if term < self.state.current_term:
                return False
            if term > self.state.current_term:
                self.state.current_term = term
                self.state.role         = Role.FOLLOWER
                self.state.voted_for    = None

            can_vote = (self.state.voted_for is None
                        or self.state.voted_for == candidate_id)
            if can_vote:
                self.state.voted_for = candidate_id
                self._reset_election_timer()
                return True
            return False

    def handle_append_entries(self, leader_id: str, term: int,
                               entries: list, commit_index: int) -> bool:
        with self._lock:
            if term < self.state.current_term:
                return False
            self.state.current_term = term
            self.state.role         = Role.FOLLOWER
            self.state.leader_id    = leader_id
            self._reset_election_timer()

            # Append new entries to log
            for entry_dict in entries:
                entry = LogEntry(**entry_dict)
                self.state.log.append(entry)

            # Apply committed entries
            if commit_index > self.state.commit_index:
                self.state.commit_index = commit_index
                self._apply_committed()
            return True

    def _apply_committed(self):
        while self.state.last_applied < self.state.commit_index:
            self.state.last_applied += 1
            if self.state.last_applied <= len(self.state.log):
                entry = self.state.log[self.state.last_applied - 1]
                self._applied_commands.append(entry.command)

    # ── client interface ───────────────────────────
    def submit_command(self, command: dict) -> bool:
        """Leader accepts client command, replicates to cluster."""
        with self._lock:
            if self.state.role != Role.LEADER:
                return False
            entry = LogEntry(
                term=self.state.current_term,
                index=len(self.state.log) + 1,
                command=command
            )
            self.state.log.append(entry)

        # Replicate to all peers
        acks = 1   # leader counts as 1
        for peer_id in self.peers:
            peer = self.cluster.get_node(peer_id)
            if peer:
                ok = peer.handle_append_entries(
                    leader_id=self.node_id,
                    term=self.state.current_term,
                    entries=[{'term': entry.term,
                               'index': entry.index,
                               'command': entry.command}],
                    commit_index=self.state.commit_index
                )
                if ok:
                    acks += 1

        # Commit if majority acked
        majority = (len(self.peers) + 1) // 2 + 1
        if acks >= majority:
            with self._lock:
                self.state.commit_index += 1
                self._apply_committed()
            return True
        return False

    def info(self) -> dict:
        return {
            'node_id':     self.node_id,
            'role':        self.state.role.value,
            'term':        self.state.current_term,
            'leader':      self.state.leader_id,
            'log_length':  len(self.state.log),
            'commit_idx':  self.state.commit_index,
        }


# ══════════════════════════════════════════════════════
# RAFT CLUSTER  (manages all nodes)
# ══════════════════════════════════════════════════════

class RaftCluster:
    """3-node Raft cluster — survives 1 node failure."""

    def __init__(self, node_ids: list[str]):
        self._nodes: dict[str, RaftNode] = {}
        for nid in node_ids:
            peers = [p for p in node_ids if p != nid]
            self._nodes[nid] = RaftNode(nid, peers, self)

    def get_node(self, node_id: str) -> Optional[RaftNode]:
        return self._nodes.get(node_id)

    def start_all(self):
        for node in self._nodes.values():
            node.start()

    def stop_all(self):
        for node in self._nodes.values():
            node.stop()

    def get_leader(self) -> Optional[RaftNode]:
        time.sleep(0.8)   # wait for election
        for node in self._nodes.values():
            if node.role == Role.LEADER:
                return node
        return None

    def kill_node(self, node_id: str):
        """Simulate node failure."""
        node = self._nodes.get(node_id)
        if node:
            node.stop()
            print(f"  [Cluster] ⚠️  Node {node_id} KILLED")

    def revive_node(self, node_id: str):
        """Simulate node recovery."""
        node = self._nodes.get(node_id)
        if node:
            node._running = True
            node._reset_election_timer()
            print(f"  [Cluster] ✅ Node {node_id} REVIVED")

    def status(self) -> list[dict]:
        return [node.info() for node in self._nodes.values()]
