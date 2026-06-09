package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{Name: "http_requests_total", Help: "Total HTTP requests"},
		[]string{"method", "path", "status"},
	)

	httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{Name: "http_request_duration_seconds", Help: "HTTP request latency"},
		[]string{"method", "path"},
	)

	embeddingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{Name: "embedding_latency_seconds", Help: "nomic embedding latency"},
	)

	ibnnForwardLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{Name: "ibnn_forward_latency_seconds", Help: "IBNN forward pass latency"},
	)

	turbovecSearchLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{Name: "turbovec_search_latency_seconds", Help: "turbovec ANN search latency"},
	)
	turbovecAddTotal = prometheus.NewCounter(
		prometheus.CounterOpts{Name: "turbovec_add_total", Help: "Total vectors added to turbovec"},
	)

	gemmaGenerateLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{Name: "gemma_generate_latency_seconds", Help: "Gemma 4 QAT generation latency"},
	)
)

func init() {
	prometheus.MustRegister(
		httpRequestsTotal,
		httpRequestDuration,
		embeddingLatency,
		ibnnForwardLatency,
		turbovecSearchLatency,
		turbovecAddTotal,
		gemmaGenerateLatency,
	)
}

// Middleware returns a Prometheus-instrumented HTTP handler.
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		duration := time.Since(start).Seconds()
		httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, http.StatusText(sw.status)).Inc()
		httpRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (sw *statusWriter) WriteHeader(code int) {
	sw.status = code
	sw.ResponseWriter.WriteHeader(code)
}

func RegisterMetricsHandler(mux *http.ServeMux) {
	mux.Handle("/metrics", promhttp.Handler())
}

func RecordEmbedding(duration time.Duration)     { embeddingLatency.Observe(duration.Seconds()) }
func RecordIBNNForward(duration time.Duration)   { ibnnForwardLatency.Observe(duration.Seconds()) }
func RecordTurbovecSearch(duration time.Duration) { turbovecSearchLatency.Observe(duration.Seconds()) }
func RecordGemmaGenerate(duration time.Duration)  { gemmaGenerateLatency.Observe(duration.Seconds()) }
func RecordTurbovecAdd()                          { turbovecAddTotal.Inc() }
