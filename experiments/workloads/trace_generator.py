"""
Adversarial trace generator for Paper 4 experiments.

Four scenarios matching the experimental matrix:
  drift   — monotone behavioral drift (tool distribution shift)
  sybil   — Sybil amplification (1 actor, many agents)
  atomic  — atomic boundary violations (restricted resources)
  mixed   — combination of all three
"""
import random
from dataclasses import dataclass


@dataclass
class TraceRequest:
    agent_id: str
    actor_id: str
    capability: str
    resource: str
    resource_class: str  # public | sensitive | restricted
    tool: str            # for IML monitoring
    depth: int           # delegation depth
    risk_score_hint: float
    step_index: int
    scenario: str


SAFE_CAPS = [
    ("acp:cap:data.read",    "metrics/public",   "public",     "safe_read",   0.10),
    ("acp:cap:data.query",   "data/reports",     "public",     "safe_query",  0.10),
]
MODERATE_CAPS = [
    ("acp:cap:data.write",   "records/main",     "sensitive",  "moderate_write", 0.50),
    ("acp:cap:msg.send",     "queue/outbound",   "sensitive",  "moderate_send",  0.60),
]
RISKY_CAPS = [
    ("acp:cap:code.execute", "runtime/sandbox",  "restricted", "risky_execute",  0.85),
    ("acp:cap:agent.delegate", "agents/pool",    "restricted", "risky_delegate", 0.90),
]
FINANCIAL_CAPS = [
    ("acp:cap:financial.transfer", "accounts/main",     "restricted", "risky_execute",  0.90),
    ("acp:cap:financial.transfer", "accounts/restricted-fund", "restricted", "risky_delegate", 0.95),
]


def generate_drift(n_agents: int = 5, n_actors: int = 2,
                   steps: int = 500, seed: int = 42) -> list[TraceRequest]:
    """
    Drift scenario: agent behavior gradually shifts from safe→risky tools.
    Simulates the behavioral drift IML must detect (Paper 2).
    """
    rng = random.Random(seed)
    agents = [f"a{i}" for i in range(n_agents)]
    actors = {agents[i]: f"u{i % n_actors}" for i in range(n_agents)}
    reqs = []

    for step in range(steps):
        # Drift fraction: 0 at start, 1 at end
        drift_frac = step / steps
        agent = rng.choice(agents)

        if rng.random() > drift_frac:
            cap, res, rc, tool, rs = rng.choice(SAFE_CAPS)
        else:
            cap, res, rc, tool, rs = rng.choice(RISKY_CAPS)

        reqs.append(TraceRequest(
            agent_id=agent, actor_id=actors[agent],
            capability=cap, resource=res, resource_class=rc,
            tool=tool, depth=1, risk_score_hint=rs,
            step_index=step, scenario="drift",
        ))
    return reqs


def generate_sybil(n_honest_actors: int = 2, sybil_agents: int = 8,
                   steps: int = 500, seed: int = 42) -> list[TraceRequest]:
    """
    Sybil scenario: one adversarial actor registers many agents to capture
    allocation share (Paper 3 Thm 5.1).
    """
    rng = random.Random(seed)
    honest_agents = [f"honest_{i}" for i in range(4)]
    sybil_agent_ids = [f"sybil_{i}" for i in range(sybil_agents)]
    all_agents = honest_agents + sybil_agent_ids
    actors = ({a: f"honest_actor_{i % n_honest_actors}" for i, a in enumerate(honest_agents)} |
              {a: "adversary" for a in sybil_agent_ids})
    reqs = []

    for step in range(steps):
        agent = rng.choice(all_agents)
        cap, res, rc, tool, rs = rng.choice(SAFE_CAPS + MODERATE_CAPS)
        reqs.append(TraceRequest(
            agent_id=agent, actor_id=actors[agent],
            capability=cap, resource=res, resource_class=rc,
            tool=tool, depth=1, risk_score_hint=rs,
            step_index=step, scenario="sybil",
        ))
    return reqs


def generate_atomic(n_agents: int = 5, n_actors: int = 2,
                    steps: int = 500, seed: int = 42) -> list[TraceRequest]:
    """
    Atomic boundary violation scenario: agents repeatedly attempt restricted/
    financial resources to stress L0 + L1 boundary enforcement.
    """
    rng = random.Random(seed)
    agents = [f"a{i}" for i in range(n_agents)]
    actors = {agents[i]: f"u{i % n_actors}" for i in range(n_agents)}
    reqs = []

    for step in range(steps):
        agent = rng.choice(agents)
        # 40% chance of financial/restricted attempt
        if rng.random() < 0.40:
            cap, res, rc, tool, rs = rng.choice(FINANCIAL_CAPS)
        else:
            cap, res, rc, tool, rs = rng.choice(SAFE_CAPS)
        reqs.append(TraceRequest(
            agent_id=agent, actor_id=actors[agent],
            capability=cap, resource=res, resource_class=rc,
            tool=tool, depth=1, risk_score_hint=rs,
            step_index=step, scenario="atomic",
        ))
    return reqs


def generate_mixed(n_agents: int = 8, n_actors: int = 3,
                   sybil_agents: int = 4, steps: int = 500,
                   seed: int = 42) -> list[TraceRequest]:
    """
    Mixed scenario: combines drift + Sybil + atomic violations.
    Used for Thm 3 (irreducibility) full-stack stress test.
    """
    rng = random.Random(seed)
    honest = [f"a{i}" for i in range(n_agents)]
    sybils = [f"sybil_{i}" for i in range(sybil_agents)]
    all_agents = honest + sybils
    actors = ({a: f"u{i % n_actors}" for i, a in enumerate(honest)} |
              {s: "adversary" for s in sybils})
    reqs = []

    for step in range(steps):
        drift_frac = step / steps
        agent = rng.choice(all_agents)
        is_sybil = agent.startswith("sybil")

        r = rng.random()
        if is_sybil:
            # Sybil agents send mostly safe requests to grab allocation share
            cap, res, rc, tool, rs = rng.choice(SAFE_CAPS + MODERATE_CAPS)
        elif r < drift_frac * 0.5:
            cap, res, rc, tool, rs = rng.choice(RISKY_CAPS)
        elif r < 0.15:
            cap, res, rc, tool, rs = rng.choice(FINANCIAL_CAPS)
        else:
            cap, res, rc, tool, rs = rng.choice(SAFE_CAPS)

        reqs.append(TraceRequest(
            agent_id=agent, actor_id=actors[agent],
            capability=cap, resource=res, resource_class=rc,
            tool=tool, depth=1, risk_score_hint=rs,
            step_index=step, scenario="mixed",
        ))
    return reqs


def generate(scenario: str, seed: int = 42, steps: int = 500) -> list[TraceRequest]:
    """Dispatcher."""
    fns = {
        "drift":  generate_drift,
        "sybil":  generate_sybil,
        "atomic": generate_atomic,
        "mixed":  generate_mixed,
    }
    return fns[scenario](steps=steps, seed=seed)
