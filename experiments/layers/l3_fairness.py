"""
Layer 3 — Fairness Allocator for Paper 4 experiments.

Implements M1 (token bucket), M2 (round-robin), M3 (actor-aware rate limiting)
as defined in Paper 3 (Fair Atomic Governance).

Each mechanism is a stateful allocator that decides WHICH pending request
reaches the L0/L1 boundary at each step.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AllocationRequest:
    agent_id: str
    actor_id: str
    capability: str
    resource: str
    resource_class: str
    step_index: int
    scenario: str


@dataclass
class AllocationResult:
    selected: Optional[AllocationRequest]
    reason: str          # "m1_token"|"m2_rr"|"m3_actor"|"quota_exhausted"|"queue_empty"
    actor_share: dict    # actor_id -> fraction of total allows
    agent_share: dict    # agent_id -> count of allows


class M1TokenBucket:
    """
    M1: Per-Agent Token Bucket (ACP baseline).
    Each agent has independent counter bounded by k0.
    Requests processed in arrival order; no cross-agent coordination.
    Fails actor-level proportionality (Thm 5.1 of Paper 3).
    """
    def __init__(self, k0: int = 10):
        self.k0 = k0
        self.counters: dict[str, int] = defaultdict(int)  # agent_id -> allows
        self.total_allows = 0
        self.actor_allows: dict[str, int] = defaultdict(int)

    def select(self, pending: list[AllocationRequest]) -> AllocationResult:
        for req in pending:
            if self.counters[req.agent_id] < self.k0:
                self.counters[req.agent_id] += 1
                self.total_allows += 1
                self.actor_allows[req.actor_id] += 1
                return AllocationResult(
                    selected=req,
                    reason="m1_token",
                    actor_share=self._actor_share(),
                    agent_share=dict(self.counters),
                )
        return AllocationResult(None, "quota_exhausted",
                                self._actor_share(), dict(self.counters))

    def _actor_share(self) -> dict:
        if self.total_allows == 0:
            return {}
        return {a: v / self.total_allows for a, v in self.actor_allows.items()}


class M2RoundRobin:
    """
    M2: Round-Robin Fair Queuing.
    Cyclic rotation over agents; skips agents with no pending request or
    exhausted quota. Achieves exact 0-share fairness but fails actor-level
    proportionality (identity-oblivious).
    """
    def __init__(self, k0: int = 10):
        self.k0 = k0
        self.rotation: list[str] = []  # ordered agent_id list
        self.pointer = 0
        self.counters: dict[str, int] = defaultdict(int)
        self.total_allows = 0
        self.actor_allows: dict[str, int] = defaultdict(int)

    def select(self, pending: list[AllocationRequest]) -> AllocationResult:
        if not pending:
            return AllocationResult(None, "queue_empty",
                                    self._actor_share(), dict(self.counters))

        # Register new agents
        pending_ids = {r.agent_id for r in pending}
        for r in pending:
            if r.agent_id not in self.rotation:
                self.rotation.append(r.agent_id)

        # Cyclic scan
        n = len(self.rotation)
        for _ in range(n):
            aid = self.rotation[self.pointer % n]
            self.pointer = (self.pointer + 1) % n
            if aid not in pending_ids:
                continue
            if self.counters[aid] >= self.k0:
                continue
            # Found eligible agent
            req = next(r for r in pending if r.agent_id == aid)
            self.counters[req.agent_id] += 1
            self.total_allows += 1
            self.actor_allows[req.actor_id] += 1
            return AllocationResult(
                selected=req,
                reason="m2_rr",
                actor_share=self._actor_share(),
                agent_share=dict(self.counters),
            )
        return AllocationResult(None, "quota_exhausted",
                                self._actor_share(), dict(self.counters))

    def _actor_share(self) -> dict:
        if self.total_allows == 0:
            return {}
        return {a: v / self.total_allows for a, v in self.actor_allows.items()}


class M3ActorAware:
    """
    M3: Actor-Aware Rate Limiting.
    Per-actor quota K_U = K_cap / M. Maintains a shared per-actor counter
    across all agents of the actor. Achieves actor-level proportionality
    and strategy-proofness (Thm 5.1 Part iii of Paper 3).
    """
    def __init__(self, k_cap: int, n_actors: int):
        self.k_cap = k_cap
        self.n_actors = max(1, n_actors)
        self.k_u = k_cap // self.n_actors  # per-actor quota
        self.actor_counters: dict[str, int] = defaultdict(int)
        self.agent_counters: dict[str, int] = defaultdict(int)
        self.total_allows = 0
        self.actor_allows: dict[str, int] = defaultdict(int)
        # Round-robin over actors
        self.actor_rotation: list[str] = []
        self.pointer = 0

    def select(self, pending: list[AllocationRequest]) -> AllocationResult:
        if not pending:
            return AllocationResult(None, "queue_empty",
                                    self._actor_share(), dict(self.agent_counters))

        # Register new actors
        pending_actors = {r.actor_id for r in pending}
        for r in pending:
            if r.actor_id not in self.actor_rotation:
                self.actor_rotation.append(r.actor_id)

        n = len(self.actor_rotation)
        for _ in range(n):
            aid = self.actor_rotation[self.pointer % n]
            self.pointer = (self.pointer + 1) % n
            if aid not in pending_actors:
                continue
            if self.actor_counters[aid] >= self.k_u:
                continue
            # Select first request from this actor
            req = next(r for r in pending if r.actor_id == aid)
            self.actor_counters[req.actor_id] += 1
            self.agent_counters[req.agent_id] += 1
            self.total_allows += 1
            self.actor_allows[req.actor_id] += 1
            return AllocationResult(
                selected=req,
                reason="m3_actor",
                actor_share=self._actor_share(),
                agent_share=dict(self.agent_counters),
            )
        return AllocationResult(None, "quota_exhausted",
                                self._actor_share(), dict(self.agent_counters))

    def _actor_share(self) -> dict:
        if self.total_allows == 0:
            return {}
        return {a: v / self.total_allows for a, v in self.actor_allows.items()}


def make_allocator(mechanism: str, n_agents: int = 10, n_actors: int = 3):
    """Factory: returns the allocator for the given mechanism name."""
    k0 = max(10, n_agents * 2)
    if mechanism == "M1":
        return M1TokenBucket(k0=k0)
    elif mechanism == "M2":
        return M2RoundRobin(k0=k0)
    elif mechanism == "M3":
        return M3ActorAware(k_cap=k0 * n_actors, n_actors=n_actors)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
