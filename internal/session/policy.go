package session

// Policy defines behavior per session type.
type Policy struct {
	MaxPerTurn      int     // max injections per turn
	TokenBudget     int     // total tokens allowed for enrichment this session
	MinScore        float32 // minimum similarity score to inject
	StaleAfterTurns int     // re-inject after this many turns
	MaxChars        int     // label truncation for results
}

// PolicyFor returns the default policy for a session type.
func PolicyFor(st SessionType) Policy {
	switch st {
	case HookLight:
		return Policy{
			MaxPerTurn:      2,
			TokenBudget:     500,
			MinScore:        0.30,
			StaleAfterTurns: 50, // basically never re-inject
			MaxChars:        200,
		}
	case HookStandard:
		return Policy{
			MaxPerTurn:      3,
			TokenBudget:     1000,
			MinScore:        0.25,
			StaleAfterTurns: 30,
			MaxChars:        200,
		}
	case HookDeep:
		return Policy{
			MaxPerTurn:      5,
			TokenBudget:     2000,
			MinScore:        0.20,
			StaleAfterTurns: 20,
			MaxChars:        300,
		}
	case AgentMCP:
		return Policy{
			MaxPerTurn:      10,
			TokenBudget:     5000,
			MinScore:        0.15,
			StaleAfterTurns: 10,
			MaxChars:        500, // agent can override per-call
		}
	case ThinkSession:
		return Policy{
			MaxPerTurn:      999, // unlimited
			TokenBudget:     999999,
			MinScore:        0.05,
			StaleAfterTurns: 1, // always fresh
			MaxChars:        1000,
		}
	case CLISession:
		return Policy{
			MaxPerTurn:      20,
			TokenBudget:     10000,
			MinScore:        0.10,
			StaleAfterTurns: 1,
			MaxChars:        500,
		}
	default:
		return Policy{MaxPerTurn: 3, TokenBudget: 1000, MinScore: 0.25, StaleAfterTurns: 30, MaxChars: 200}
	}
}
