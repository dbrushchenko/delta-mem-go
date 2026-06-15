package session

import (
	"sync"
	"time"
)

// SessionType determines policy for what gets surfaced and how much.
type SessionType int

const (
	HookLight    SessionType = iota // gRPC hook (fast, 1-3 results, strict budget)
	HookStandard                    // gRPC hook standard tier
	HookDeep                        // gRPC hook deep tier
	AgentMCP                        // MCP interactive (generous, 5-10 results)
	ThinkSession                    // Think synthesis (full access, no budget)
	CLISession                      // mem-cli manual use
)

func (t SessionType) String() string {
	return [...]string{"hook-light", "hook-standard", "hook-deep", "agent-mcp", "think", "cli"}[t]
}

// Manager tracks all active sessions and enforces policies.
type Manager struct {
	sessions map[string]*Session
	mu       sync.RWMutex
	ttl      time.Duration
}

func NewManager(ttl time.Duration) *Manager {
	m := &Manager{sessions: make(map[string]*Session), ttl: ttl}
	go m.reaper()
	return m
}

// GetOrCreate returns an existing session or creates one with the given type.
func (m *Manager) GetOrCreate(id, owner string, st SessionType) *Session {
	m.mu.RLock()
	if s, ok := m.sessions[id]; ok {
		s.LastActive = time.Now()
		s.TurnCount++
		m.mu.RUnlock()
		return s
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sessions[id]; ok {
		s.LastActive = time.Now()
		s.TurnCount++
		return s
	}
	s := &Session{
		ID:          id,
		Owner:       owner,
		Type:        st,
		Policy:      PolicyFor(st),
		Injected:    make(map[string]*Injection),
		Created:     time.Now(),
		LastActive:  time.Now(),
		TurnCount:   1,
		TokensUsed:  0,
	}
	m.sessions[id] = s
	return s
}

// Close ends a session and returns its summary for persistence.
func (m *Manager) Close(id string) *SessionSummary {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok { return nil }
	delete(m.sessions, id)
	return s.Summarize()
}

// ActiveCount returns the number of active sessions.
func (m *Manager) ActiveCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.sessions)
}

// reaper closes expired sessions periodically.
func (m *Manager) reaper() {
	ticker := time.NewTicker(m.ttl / 2)
	for range ticker.C {
		m.mu.Lock()
		now := time.Now()
		for id, s := range m.sessions {
			if now.Sub(s.LastActive) > m.ttl {
				delete(m.sessions, id)
			}
		}
		m.mu.Unlock()
	}
}
