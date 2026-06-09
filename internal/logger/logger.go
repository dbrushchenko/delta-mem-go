package logger

import (
	"log/slog"
	"os"
)

func New(level string) *slog.Logger {
	var opts slog.HandlerOptions
	switch level {
	case "debug":
		opts.Level = slog.LevelDebug
	case "warn":
		opts.Level = slog.LevelWarn
	case "error":
		opts.Level = slog.LevelError
	default:
		opts.Level = slog.LevelInfo
	}
	handler := slog.NewJSONHandler(os.Stdout, &opts)
	return slog.New(handler).With(
		slog.String("service", "delta-mem-go"),
	)
}

func WithTrace(log *slog.Logger, owner, traceID string) *slog.Logger {
	return log.With(
		slog.String("owner", owner),
		slog.String("trace_id", traceID),
	)
}
