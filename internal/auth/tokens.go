package auth

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
)

// TokenStore maps bearer tokens to owner names.
type TokenStore struct {
	tokens map[string]string // token → owner
	mu     sync.RWMutex
	path   string // persist to file
}

// NewTokenStore loads tokens from file or env.
func NewTokenStore(path string) *TokenStore {
	ts := &TokenStore{tokens: make(map[string]string), path: path}
	// Load from file
	if data, err := os.ReadFile(path); err == nil {
		json.Unmarshal(data, &ts.tokens)
	}
	// Load from env (comma-separated token:owner pairs)
	if env := os.Getenv("MEMGO_TOKENS"); env != "" {
		for _, pair := range strings.Split(env, ",") {
			parts := strings.SplitN(strings.TrimSpace(pair), ":", 2)
			if len(parts) == 2 { ts.tokens[parts[0]] = parts[1] }
		}
	}
	return ts
}

// Resolve returns the owner for a token, or empty string if invalid.
func (ts *TokenStore) Resolve(token string) string {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return ts.tokens[token]
}

// Enroll creates a new owner with a generated token. Returns the token.
func (ts *TokenStore) Enroll(owner string) string {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	// Check if owner already has a token
	for tok, o := range ts.tokens {
		if o == owner { return tok }
	}
	token := generateToken()
	ts.tokens[token] = owner
	ts.save()
	return token
}

// Enabled returns true if any tokens are configured.
func (ts *TokenStore) Enabled() bool {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return len(ts.tokens) > 0 || os.Getenv("MEMGO_AUTH") == "true"
}

func (ts *TokenStore) save() {
	if ts.path == "" { return }
	data, _ := json.MarshalIndent(ts.tokens, "", "  ")
	os.WriteFile(ts.path, data, 0600)
}

func generateToken() string {
	b := make([]byte, 24)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// Middleware validates bearer token and injects owner into request context.
// Skips auth for /health, /ready, /metrics, /enroll.
func Middleware(ts *TokenStore) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip auth for public endpoints
			if r.URL.Path == "/health" || r.URL.Path == "/ready" || r.URL.Path == "/metrics" || r.URL.Path == "/enroll" {
				next.ServeHTTP(w, r)
				return
			}
			if !ts.Enabled() {
				next.ServeHTTP(w, r)
				return
			}
			auth := r.Header.Get("Authorization")
			if !strings.HasPrefix(auth, "Bearer ") {
				http.Error(w, `{"error":"missing bearer token"}`, http.StatusUnauthorized)
				return
			}
			token := strings.TrimPrefix(auth, "Bearer ")
			owner := ts.Resolve(token)
			if owner == "" {
				http.Error(w, `{"error":"invalid token"}`, http.StatusForbidden)
				return
			}
			// Inject owner into header so downstream handlers use it
			r.Header.Set("X-Owner", owner)
			next.ServeHTTP(w, r)
		})
	}
}

// EnrollHandler handles POST /enroll — self-enrollment for new owners.
func EnrollHandler(ts *TokenStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct{ Owner string `json:"owner"` }
		json.NewDecoder(r.Body).Decode(&req)
		if req.Owner == "" {
			http.Error(w, `{"error":"owner required"}`, http.StatusBadRequest)
			return
		}
		token := ts.Enroll(req.Owner)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"token": token, "owner": req.Owner})
	}
}

// Privacy filter — scrubs secrets from content before storage.
var secretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(api[_-]?key|token|password|secret|bearer)\s*[:=]\s*\S+`),
	regexp.MustCompile(`ghp_[A-Za-z0-9]{36}`),
	regexp.MustCompile(`(?i)authorization:\s*bearer\s+\S+`),
	regexp.MustCompile(`[A-Za-z0-9+/]{40,}={0,2}`), // base64 blobs > 40 chars
}

// ScrubSecrets removes likely secrets from text before storage.
func ScrubSecrets(text string) string {
	for _, re := range secretPatterns {
		text = re.ReplaceAllString(text, "[REDACTED]")
	}
	return text
}
