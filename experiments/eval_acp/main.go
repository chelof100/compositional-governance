// eval_acp — Paper 4 ACP evaluation harness for L0 and L1.
//
// Reads a JSON array of requests from stdin, evaluates each through the real
// ACP risk engine (L1) and optionally the atomic boundary check (L0),
// and writes a JSON array of results to stdout.
//
// Usage:
//   echo '[{...}]' | eval_acp.exe --subset=L0L1
//
// Subsets: stateless | L1 | L0L1
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/chelof100/acp-framework/acp-go/pkg/risk"
)

// ── I/O types ─────────────────────────────────────────────────────────────────

type Request struct {
	AgentID       string `json:"agent_id"`
	ActorID       string `json:"actor_id"`
	Capability    string `json:"capability"`
	Resource      string `json:"resource"`
	ResourceClass string `json:"resource_class"` // "public"|"sensitive"|"restricted"
	StepIndex     int    `json:"step_index"`
	Scenario      string `json:"scenario"`
}

type Result struct {
	StepIndex    int    `json:"step_index"`
	AgentID      string `json:"agent_id"`
	ActorID      string `json:"actor_id"`
	Decision     string `json:"decision"`
	RSFinal      int    `json:"rs_final"`
	DeniedReason string `json:"denied_reason,omitempty"`
	Subset       string `json:"subset"`
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	subset := flag.String("subset", "L0L1", "Layer subset: stateless|L1|L0L1")
	flag.Parse()

	var requests []Request
	if err := json.NewDecoder(os.Stdin).Decode(&requests); err != nil {
		fmt.Fprintf(os.Stderr, "decode error: %v\n", err)
		os.Exit(1)
	}

	ledger := risk.NewInMemoryQuerier()
	policy := risk.DefaultPolicyConfig()
	now := time.Now()
	results := make([]Result, 0, len(requests))

	for i, req := range requests {
		t := now.Add(time.Duration(i) * 100 * time.Millisecond)
		rc := parseRC(req.ResourceClass)

		evalReq := risk.EvalRequest{
			AgentID:       req.AgentID,
			Capability:    req.Capability,
			Resource:      req.Resource,
			ResourceClass: rc,
			Policy:        policy,
			Now:           t,
		}

		res := Result{
			StepIndex: req.StepIndex,
			AgentID:   req.AgentID,
			ActorID:   req.ActorID,
			Subset:    *subset,
		}

		switch *subset {
		case "stateless":
			res.Decision = "APPROVED"
			res.RSFinal = 0

		case "L1":
			r, err := risk.Evaluate(evalReq, ledger)
			if err != nil {
				res.Decision = "DENIED"
				res.DeniedReason = "EVAL_ERROR"
			} else {
				res.Decision = string(r.Decision)
				res.RSFinal = r.RSFinal
				res.DeniedReason = r.DeniedReason
				mutate(ledger, evalReq, r, t)
			}

		case "L0L1":
			// L0: atomic boundary pre-filter
			if !l0Check(req.Capability, rc) {
				res.Decision = "ESCALATED"
				res.RSFinal = -1
				res.DeniedReason = "L0_BOUNDARY"
				results = append(results, res)
				continue
			}
			// L1: stateful risk engine
			r, err := risk.Evaluate(evalReq, ledger)
			if err != nil {
				res.Decision = "DENIED"
				res.DeniedReason = "EVAL_ERROR"
			} else {
				res.Decision = string(r.Decision)
				res.RSFinal = r.RSFinal
				res.DeniedReason = r.DeniedReason
				mutate(ledger, evalReq, r, t)
			}
		}
		results = append(results, res)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(results)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func parseRC(s string) risk.ResourceClass {
	switch s {
	case "sensitive":
		return risk.ResourceSensitive
	case "restricted":
		return risk.ResourceRestricted
	default:
		return risk.ResourcePublic
	}
}

// l0Check: Paper 0 atomic boundary.
// Restricted resources require financial or admin capability prefix.
func l0Check(cap string, rc risk.ResourceClass) bool {
	if rc == risk.ResourceRestricted {
		return cap == "acp:cap:financial.transfer" ||
			cap == "acp:cap:admin.override" ||
			cap == "acp:cap:data.read"
	}
	return true
}

// mutate applies the evaluate-then-mutate contract (ACP §4).
func mutate(q *risk.InMemoryQuerier, req risk.EvalRequest, r *risk.EvalResult, now time.Time) {
	patKey := risk.PatternKey(req.AgentID, req.Capability, req.Resource)
	q.AddRequest(req.AgentID, now)
	q.AddPattern(patKey, now)
	if r.Decision == risk.DENIED && r.DeniedReason != "COOLDOWN_ACTIVE" {
		q.AddDenial(req.AgentID, now)
		if enter, _ := risk.ShouldEnterCooldown(req.AgentID, req.Policy, q, now); enter {
			q.SetCooldown(req.AgentID, now.Add(
				time.Duration(req.Policy.CooldownPeriodSeconds)*time.Second))
		}
	}
}
