package mcp

import (
	"context"
	"fmt"
	"strings"

	"github.com/dbrushchenko/delta-mem-go/internal/server"
	"github.com/dbrushchenko/delta-mem-go/internal/session"
)

// RegisterTools wires all δ-mem-go service methods as MCP tools.
func RegisterTools(h *Handler, svc *server.Service) {
	h.AddTool(Tool{
		Name: "dmem_store", Description: "Store a fact. Layers: embed → δ-mem → turbovec → self-model.",
		InputSchema: Schema(map[string]any{"key": StringProp("short label"), "content": StringProp("full text"), "owner": StringProp("owner name")}, "key", "content"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			norm, err := svc.Store(context.Background(), owner, str(args, "key"), str(args, "content"))
			if err != nil { return "", err }
			return fmt.Sprintf("stored: norm=%.4f", norm), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_store_deep", Description: "Store through ALL layers (Store + Learn). Full pipeline.",
		InputSchema: Schema(map[string]any{"key": StringProp("short label"), "content": StringProp("full text"), "owner": StringProp("owner name")}, "key", "content"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			norm, err := svc.StoreDeep(context.Background(), owner, str(args, "key"), str(args, "content"))
			if err != nil { return "", err }
			return fmt.Sprintf("stored-deep: norm=%.4f", norm), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_recall", Description: "Query δ-mem. Returns confidence. Triggers self-training.",
		InputSchema: Schema(map[string]any{"query": StringProp("search text"), "owner": StringProp("owner name")}, "query"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			_, conf, err := svc.Recall(context.Background(), owner, str(args, "query"))
			if err != nil { return "", err }
			return fmt.Sprintf("confidence=%.4f", conf), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_learn", Description: "Absorb a fact deeply. embed → δ-mem → turbogo → IBNN reinforce.",
		InputSchema: Schema(map[string]any{"fact": StringProp("fact to learn"), "owner": StringProp("owner name")}, "fact"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			return "learned", svc.Learn(context.Background(), owner, str(args, "fact"))
		},
	})

	h.AddTool(Tool{
		Name: "dmem_think", Description: "Full synthesis. ALL 12 layers including Gemma + Wander. Seeds: comma-separated.",
		InputSchema: Schema(map[string]any{"seeds": StringProp("comma-separated seed concepts"), "owner": StringProp("owner name")}, "seeds"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			seeds := strings.Split(str(args, "seeds"), ",")
			for i := range seeds { seeds[i] = strings.TrimSpace(seeds[i]) }
			thought, err := svc.Think(context.Background(), owner, seeds)
			if err != nil { return "", err }
			result := fmt.Sprintf("idea: %s\nconf=%.4f novelty=%.4f valid=%v", thought.Idea, thought.Confidence, thought.Novelty, thought.Valid)
			if len(thought.Neighbors) > 0 { result += fmt.Sprintf("\nneighbors: %s", strings.Join(thought.Neighbors, ", ")) }
			return result, nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_adapt", Description: "Correct a misconception. Suppresses wrong, strengthens right.",
		InputSchema: Schema(map[string]any{"wrong": StringProp("incorrect statement"), "right": StringProp("correct statement"), "owner": StringProp("owner name")}, "wrong", "right"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			impact, err := svc.Adapt(context.Background(), owner, str(args, "wrong"), str(args, "right"))
			if err != nil { return "", err }
			return fmt.Sprintf("adapted: impact=%.4f", impact), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_forget", Description: "Fade a memory from δ-mem state.",
		InputSchema: Schema(map[string]any{"what": StringProp("text to forget"), "owner": StringProp("owner name")}, "what"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			return "forgotten", svc.Forget(context.Background(), owner, str(args, "what"))
		},
	})

	h.AddTool(Tool{
		Name: "dmem_validate", Description: "Check statement against truth engine (axioms + NLI).",
		InputSchema: Schema(map[string]any{"statement": StringProp("statement to check"), "owner": StringProp("owner name")}, "statement"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			valid, grounding, coherence, reason, contradictions := svc.Validate(context.Background(), owner, str(args, "statement"))
			status := "✓ VALID"
			if !valid { status = "✗ REJECTED" }
			result := fmt.Sprintf("%s grounding=%.3f coherence=%.3f", status, grounding, coherence)
			if reason != "" { result += " reason=" + reason }
			if len(contradictions) > 0 { result += fmt.Sprintf(" contradicts=%v", contradictions) }
			return result, nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_search", Description: "Vector search. deep=true for turbogo (quantized), false for turbovec.",
		InputSchema: Schema(map[string]any{"query": StringProp("search text"), "deep": BoolProp("use turbogo", false), "owner": StringProp("owner name")}, "query"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			deep, _ := args["deep"].(bool)
			// Embed via IBNN
			emb, err := svc.IBNNForward(context.Background(), owner, str(args, "query"))
			if err != nil { return "", err }
			var ids []string; var scores []float32
			if deep {
				ids, scores, err = svc.TurbogoSearch(context.Background(), owner, emb, 5)
			} else {
				ids, scores, err = svc.TurboSearch(context.Background(), owner, emb, 5)
			}
			if err != nil { return "", err }
			var lines []string
			for i, id := range ids { lines = append(lines, fmt.Sprintf("[%d] %.4f  %s", i+1, scores[i], id)) }
			if len(lines) == 0 { return "(no results)", nil }
			return strings.Join(lines, "\n"), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_confident", Description: "Check self-model confidence on a topic.",
		InputSchema: Schema(map[string]any{"text": StringProp("topic to check"), "owner": StringProp("owner name")}, "text"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			level, raw := svc.AmIConfident(context.Background(), owner, str(args, "text"))
			levels := []string{"NeverSeen", "Uncertain", "Confident"}
			l := "Unknown"
			if level < len(levels) { l = levels[level] }
			return fmt.Sprintf("%s (raw=%.2f)", l, raw), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_temporal", Description: "Query recent temporal events.",
		InputSchema: Schema(map[string]any{"limit": IntProp("max events", 10), "owner": StringProp("owner name")}, ),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			limit := intVal(args, "limit", 10)
			events := svc.QueryTemporal(owner, limit)
			var lines []string
			for _, e := range events { lines = append(lines, fmt.Sprintf("[%s] %s — %s", e.When.Format("2006-01-02T15:04"), e.ID, e.Content)) }
			if len(lines) == 0 { return "(no events)", nil }
			return strings.Join(lines, "\n"), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_wander", Description: "Wander control. action: start, stop, harvest.",
		InputSchema: Schema(map[string]any{"action": StringProp("start|stop|harvest"), "owner": StringProp("owner name")}, "action"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			switch str(args, "action") {
			case "start": svc.StartWander(owner); return "wander started", nil
			case "stop": svc.StopWander(owner); return "wander stopped", nil
			default:
				thoughts := svc.HarvestWander(owner)
				if len(thoughts) == 0 { return "(no spontaneous thoughts)", nil }
				var lines []string
				for _, t := range thoughts { lines = append(lines, fmt.Sprintf("[%.3f|nov=%.3f] %s", t.Confidence, t.Novelty, t.Idea)) }
				return strings.Join(lines, "\n"), nil
			}
		},
	})

	h.AddTool(Tool{
		Name: "dmem_axiom", Description: "Set an immutable truth. Cannot be overridden.",
		InputSchema: Schema(map[string]any{"statement": StringProp("truth statement"), "domain": StringProp("optional domain")}, "statement"),
		Fn: func(args map[string]any) (string, error) {
			svc.AddAxiom(str(args, "statement"), str(args, "domain"))
			return fmt.Sprintf("axiom set: %s", str(args, "statement")), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_health", Description: "Server health: owners, stores, recalls, uptime.",
		InputSchema: Schema(map[string]any{}),
		Fn: func(args map[string]any) (string, error) {
			owners, _, stores, recalls, uptime := svc.Health()
			return fmt.Sprintf("owners=%d stores=%d recalls=%d uptime=%s", owners, stores, recalls, uptime), nil
		},
	})

	// Session-aware search and management
	h.AddTool(Tool{
		Name: "dmem_session_search", Description: "Session-aware search. Respects budget, dedup, staleness. Won't repeat results already shown this session.",
		InputSchema: Schema(map[string]any{"query": StringProp("search text"), "session_id": StringProp("session ID (auto-generated if empty)"), "session_type": StringProp("hook-light|hook-standard|hook-deep|agent-mcp|cli"), "max_chars": IntProp("label length", 200), "owner": StringProp("owner name")}, "query"),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			sid := str(args, "session_id")
			if sid == "" { sid = "mcp-" + owner }
			stStr := str(args, "session_type")
			if stStr == "" { stStr = "agent-mcp" }
			maxChars := intVal(args, "max_chars", 200)
			// Embed query
			emb, err := svc.IBNNForward(context.Background(), owner, str(args, "query"))
			if err != nil { return "", err }
			// Parse session type
			var st session.SessionType
			switch stStr {
			case "hook-light": st = session.HookLight
			case "hook-standard": st = session.HookStandard
			case "hook-deep": st = session.HookDeep
			case "agent-mcp": st = session.AgentMCP
			case "think": st = session.ThinkSession
			case "cli": st = session.CLISession
			default: st = session.AgentMCP
			}
			ids, scores, err := svc.SessionSearch(context.Background(), sid, owner, st, emb, 5)
			if err != nil { return "", err }
			var lines []string
			for i, id := range ids {
				label := id
				if len(label) > maxChars { label = label[:maxChars] }
				lines = append(lines, fmt.Sprintf("[%d] %.4f  %s", i+1, scores[i], label))
			}
			if len(lines) == 0 { return "(no new results — budget exhausted or all seen this session)", nil }
			return strings.Join(lines, "\n"), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_session_status", Description: "Session status: injected count, tokens used, budget remaining.",
		InputSchema: Schema(map[string]any{"session_id": StringProp("session ID"), "owner": StringProp("owner name")}),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			sid := str(args, "session_id")
			if sid == "" { sid = "mcp-" + owner }
			sess := svc.Sessions().GetOrCreate(sid, owner, session.AgentMCP)
			remaining := sess.Policy.TokenBudget - sess.TokensUsed
			return fmt.Sprintf("session=%s turns=%d injected=%d tokens_used=%d budget_remaining=%d",
				sid, sess.TurnCount, len(sess.Injected), sess.TokensUsed, remaining), nil
		},
	})

	h.AddTool(Tool{
		Name: "dmem_session_reset", Description: "Reset a session. Clears dedup and budget — all memories become eligible for re-injection. Tracking restarts from zero.",
		InputSchema: Schema(map[string]any{"session_id": StringProp("session ID to reset"), "owner": StringProp("owner name")}),
		Fn: func(args map[string]any) (string, error) {
			owner := str(args, "owner")
			sid := str(args, "session_id")
			if sid == "" { sid = "mcp-" + owner }
			summary := svc.Sessions().Close(sid)
			if summary != nil {
				return fmt.Sprintf("session closed (was: turns=%d injected=%d tokens=%d) — tracking restarted",
					summary.Turns, summary.Injected, summary.TokensUsed), nil
			}
			return "session reset — tracking restarted from zero", nil
		},
	})
}

func str(args map[string]any, key string) string {
	if v, ok := args[key].(string); ok { return v }
	return ""
}

func intVal(args map[string]any, key string, def int) int {
	if v, ok := args[key].(float64); ok { return int(v) }
	return def
}

