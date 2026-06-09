package health

import "net/http"

type ReadinessChecker struct {
	EmbedderReady bool
	TurbovecReady bool
	GemmaReady    bool
}

func (c *ReadinessChecker) Check(w http.ResponseWriter, r *http.Request) {
	if !c.EmbedderReady || !c.TurbovecReady {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("NOT READY"))
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("READY"))
}
