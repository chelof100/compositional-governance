# Compositional Governance — Paper 4

**Paper 4 of the Agent Governance Series.**

> *Correctness is per-layer. Governance is the composition.*  
> *Four projections. Each necessary. None redundant.*

---

## Paper

**Irreducible Multi-Scale Governance: Composition and Limits of Atomic Admission Systems**  
Marcelo Fernandez (TraslaIA), 2026

DOI: [TBD — Zenodo] &nbsp;·&nbsp; arXiv: [TBD]

---

## What this is

This repository contains the LaTeX source and experiment code for **Paper 4** of the Agent Governance Series — the synthesis paper that composes the four governance layers from Papers 0–3 into a single formal architecture and proves it is the *minimal* composition that satisfies all runtime governance guarantees.

**The core problem:** Papers 0–3 each address a distinct governance failure mode. But do those layers compose correctly? Can any subset of three layers cover all four guarantees? The answer is no — and this paper proves it.

The paper introduces:
- **Synchronous composition** `⊗_κ` and **asynchronous coupling** `→_κ` for governance layers, with formal interface compatibility conditions (C1–C3).
- **Theorem 1 (Interface Compatibility):** The composition `L0 ⊗ L1 ⊗ L2 ⊗ L3` satisfies the four-layer interface contract under conditions C1–C3.
- **Theorem 2 (Feedback Convergence):** Under assumptions FC1–FC3, the closed-loop IML feedback system converges: `lim sup D̂_t ≤ ε`.
- **Theorem 3 (Irreducibility):** Under finite observability (finite state spaces, bounded inter-layer summaries, local decision-making), no composition of fewer than four layers from `{L0, L1, L2, L3}` simultaneously satisfies all four governance guarantees.
- **Lemma 5.2 (Contraction):** The feedback map `Φ(x) = ρx + ε_b` with `ρ = 1 − Kη ∈ (0,1)` is a contraction. Empirically: `K = η = 0.50`, `ρ = 0.70`.

**Empirical validation:** 160-trial ablation study, 10-seed feedback convergence experiment, 20-trial compatibility check — using the real ACP risk engine and IML deviation estimator.

**Paper 0 (DBM):** https://github.com/chelof100/decision-boundary-model  
**Paper 1 (ACP):** https://github.com/chelof100/acp-framework-en  
**Paper 2 (IML):** https://github.com/chelof100/iml-benchmark  
**Paper 3 (Fairness):** https://github.com/chelof100/fair-atomic-governance  
**arXiv (ACP):** https://arxiv.org/abs/2603.18829

---

## Repository contents

```
compositional-governance/
├── main.tex                          # Full LaTeX source (20 pages)
├── references.bib                    # Bibliography
├── main.pdf                          # Compiled paper
├── experiments/
│   ├── orchestrator.py               # Main experiment runner (3 experiments)
│   ├── layers/
│   │   ├── l0_atomic.py              # L0: atomic decision boundary
│   │   ├── l1_acp.py                 # L1: ACP risk engine (wraps Go binary)
│   │   ├── l2_iml.py                 # L2: IML deviation estimator
│   │   └── l3_fairness.py            # L3: fairness allocation (M1/M2/M3)
│   ├── analysis/
│   │   └── plots.py                  # Figure generator (3 PDF figures)
│   ├── results/                      # CSV + JSON outputs
│   └── figures/                      # Generated PDF figures
├── README.md
├── LICENSE
└── .gitignore
```

---

## Three experiments

### Experiment A — Ablation (Theorem 3: Irreducibility)

8 layer subsets × 4 guarantees × 20 trials = **160 trials**.

| Subset | P1 Atomicity | P2 Drift | P3 Fairness | P4 Sybil |
|--------|:---:|:---:|:---:|:---:|
| L0+L1+L2+L3 (full) | ✓ | ✓ | ✓ | ✓ |
| L0+L1+L2 | ✓ | ✓ | ✗ | ✗ |
| L0+L1+L3 | ✓ | ✗ | ✓ | ✓ |
| L0+L2+L3 | ✗ | ✓ | ✓ | ✓ |
| L1+L2+L3 | ✓ | ✓ | ✓ | ✓ |
| L0+L1 | ✓ | ✗ | ✗ | ✗ |
| L1+L2 | ✓ | ✓ | ✗ | ✗ |
| L1+L3 | ✓ | ✗ | ✓ | ✓ |

The L1+L2+L3 row passes all guarantees empirically — this is a *runtime equivalence case* (L1 subsumes L0 in steady state) documented in §7.2, not a counterexample to irreducibility.

### Experiment B — Feedback Convergence (Theorem 2)

10 seeds, 1000 steps each, drift scenario.

| Condition | `lim sup D̂` | Converges |
|-----------|:-----------:|:---------:|
| With feedback (FC1+FC2, K=η=0.50, ρ=0.70) | 0.128 ± 0.026 | 10/10 |
| Without feedback (open-loop) | 0.419 ± 0.007 | 0/10 |

Measured contraction: `0.128/0.419 ≈ 0.31 ≈ ρ²` — consistent with Lemma 5.2.

### Experiment C — Interface Compatibility (Theorem 1)

20 trials across 4 scenarios (drift, sybil, atomic, mixed).

All scenarios achieve P1 compatibility = 1.00, P2 compatibility = 1.00,  
decision equivalence ≥ 0.95.

---

## Reproduce experiments

```bash
git clone https://github.com/chelof100/compositional-governance
cd compositional-governance/experiments

# Install dependencies
pip install pandas numpy matplotlib

# Run all experiments (generates CSVs in results/)
python orchestrator.py

# Generate figures (PDFs in figures/)
python analysis/plots.py
```

---

## The four-layer architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  L3 — Fairness Allocation     [Paper 3, fair-atomic-governance]  │  Who gets to act?
│  Population-level: actor shares, Sybil resistance                │
├──────────────────────────────────────────────────────────────────┤
│  L2 — IML Drift Monitor       [Paper 2, iml-benchmark]           │  Has behavior drifted?
│  Behavioral: D̂_t, feedback to ACP threshold                     │
├──────────────────────────────────────────────────────────────────┤
│  L1 — ACP Admission Control   [Paper 1, acp-framework-en]        │  Is this action admissible?
│  State: capability token, audit ledger, risk scoring             │
├──────────────────────────────────────────────────────────────────┤
│  L0 — Atomic Decision Boundary [Paper 0, decision-boundary-model]│  Can guarantees be made?
│  Temporal: decision ⊗ transition as single indivisible step      │
└──────────────────────────────────────────────────────────────────┘
```

The four layers correspond to four orthogonal projections of the governance space:
**temporal** (atomicity) · **state** (enforcement history) · **behavioral** (drift monitoring) · **population** (fair allocation).

Irreducibility (Theorem 3): no single mechanism can cover two projections simultaneously under finite observability.

---

## Position in the series

| Paper | Title | Repo | Status |
|---|---|---|---|
| **Paper 0** | Atomic Decision Boundaries | [decision-boundary-model](https://github.com/chelof100/decision-boundary-model) | In preparation |
| **Paper 1** | Agent Control Protocol (ACP) | [acp-framework-en](https://github.com/chelof100/acp-framework-en) | [Published — arXiv:2603.18829](https://arxiv.org/abs/2603.18829) |
| **Paper 2** | From Admission to Invariants (IML) | [iml-benchmark](https://github.com/chelof100/iml-benchmark) | In preparation |
| **Paper 3** | Fair Atomic Governance | [fair-atomic-governance](https://github.com/chelof100/fair-atomic-governance) | In preparation |
| **Paper 4** | Irreducible Multi-Scale Governance (this repo) | [compositional-governance](https://github.com/chelof100/compositional-governance) | In preparation |

**Series logic:**
- Paper 0 proves *when* admissibility can be guaranteed (structural necessity).
- Paper 1 builds a protocol that satisfies that condition (ACP, TLA+ verified).
- Paper 2 detects behavioral drift invisible to enforcement (IML, above the boundary).
- Paper 3 proves correct enforcement does not imply fair allocation (allocation layer).
- Paper 4 composes all four layers and proves their joint necessity (this paper).

---

## Citation

```bibtex
@misc{fernandez2026comp,
  title   = {Irreducible Multi-Scale Governance: Composition and Limits
             of Atomic Admission Systems},
  author  = {Fernandez, Marcelo},
  year    = {2026},
  note    = {arXiv: [TBD]. DOI: [TBD]. Paper~4 of the Agent Governance Series.
             Source: https://github.com/chelof100/compositional-governance}
}
```

---

## Author

**Marcelo Fernandez** — TraslaIA — info@traslaia.com  
https://agentcontrolprotocol.xyz
