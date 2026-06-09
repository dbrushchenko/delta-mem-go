package server

import (
	"encoding/json"
	"net/http"
)

func (s *Service) RegisterHTTP(mux *http.ServeMux) {
	mux.HandleFunc("POST /store", s.handleStore)
	mux.HandleFunc("POST /recall", s.handleRecall)
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /reset", s.handleReset)
	mux.HandleFunc("POST /ibnn-forward", s.handleIBNNForward)
	mux.HandleFunc("POST /ibnn-forward-hidden", s.handleIBNNForwardHidden)
	mux.HandleFunc("POST /turbo-add", s.handleTurboAdd)
	mux.HandleFunc("POST /turbo-search", s.handleTurboSearch)
	mux.HandleFunc("POST /generate", s.handleGenerate)
}

type storeReq struct{ Owner, Key, Content string }
type recallReq struct{ Owner, Query string }
type resetReq struct{ Owner string }
type ibnnReq struct{ Owner, Text string }
type ibnnHiddenReq struct{ Owner string; HiddenState []float32 `json:"hidden_state"` }
type turboAddReq struct{ Owner, ID string; Vector []float32 }
type turboSearchReq struct{ Owner string; Query []float32; K int }
type generateReq struct{ Owner, Prompt string }

func (s *Service) handleStore(w http.ResponseWriter, r *http.Request) {
	var req storeReq; json.NewDecoder(r.Body).Decode(&req)
	norm, err := s.Store(r.Context(), req.Owner, req.Key, req.Content)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"ok": true, "state_norm": norm})
}

func (s *Service) handleRecall(w http.ResponseWriter, r *http.Request) {
	var req recallReq; json.NewDecoder(r.Body).Decode(&req)
	corr, conf, err := s.Recall(r.Context(), req.Owner, req.Query)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"confidence": conf, "correction_dim": len(corr)})
}

func (s *Service) handleHealth(w http.ResponseWriter, r *http.Request) {
	owners, avgNorm, stores, recalls, uptime := s.Health()
	json.NewEncoder(w).Encode(map[string]any{"owners_active": owners, "avg_state_norm": avgNorm, "total_stores": stores, "total_recalls": recalls, "uptime": uptime})
}

func (s *Service) handleReset(w http.ResponseWriter, r *http.Request) {
	var req resetReq; json.NewDecoder(r.Body).Decode(&req)
	s.ResetState(r.Context(), req.Owner)
	json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func (s *Service) handleIBNNForward(w http.ResponseWriter, r *http.Request) {
	var req ibnnReq; json.NewDecoder(r.Body).Decode(&req)
	out, err := s.IBNNForward(r.Context(), req.Owner, req.Text)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"output": out, "dim": len(out)})
}

func (s *Service) handleIBNNForwardHidden(w http.ResponseWriter, r *http.Request) {
	var req ibnnHiddenReq; json.NewDecoder(r.Body).Decode(&req)
	out, err := s.IBNNForwardHidden(r.Context(), req.Owner, req.HiddenState)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"output": out, "dim": len(out)})
}

func (s *Service) handleTurboAdd(w http.ResponseWriter, r *http.Request) {
	var req turboAddReq; json.NewDecoder(r.Body).Decode(&req)
	if err := s.TurboAdd(r.Context(), req.Owner, req.ID, req.Vector); err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func (s *Service) handleTurboSearch(w http.ResponseWriter, r *http.Request) {
	var req turboSearchReq; json.NewDecoder(r.Body).Decode(&req)
	ids, scores, err := s.TurboSearch(r.Context(), req.Owner, req.Query, req.K)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"ids": ids, "scores": scores})
}

func (s *Service) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req generateReq; json.NewDecoder(r.Body).Decode(&req)
	resp, err := s.Generate(r.Context(), req.Owner, req.Prompt)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]any{"response": resp})
}
