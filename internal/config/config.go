package config

import (
	"flag"
	"os"
	"strings"
)

// Config holds all production configuration.
type Config struct {
	HTTPPort      int
	GRPCPort      int
	DataDir       string
	ModelPath     string
	ORTLib        string
	EmbedDim      int
	APIKeys       map[string]bool
	RateLimit     int
	LogLevel      string
	GemmaModel    string
	GemmaModelPath string
	GemmaURL      string
	TurbovecURL   string
}

func Load() *Config {
	c := &Config{
		HTTPPort:    8080,
		GRPCPort:    9090,
		DataDir:     "./data/states",
		EmbedDim:    768,
		RateLimit:   1000,
		LogLevel:    "info",
		GemmaModel:  "gemma-4-e4b-it-q4",
		GemmaURL:    "http://localhost:11434",
		TurbovecURL: "http://localhost:8001",
		APIKeys:     make(map[string]bool),
	}

	flag.IntVar(&c.HTTPPort, "port", c.HTTPPort, "HTTP port")
	flag.IntVar(&c.GRPCPort, "grpc-port", c.GRPCPort, "gRPC port")
	flag.StringVar(&c.DataDir, "data", c.DataDir, "state directory")
	flag.StringVar(&c.ModelPath, "model", "", "nomic ONNX model path")
	flag.StringVar(&c.ORTLib, "ort-lib", "", "ONNX Runtime .so path")
	flag.IntVar(&c.EmbedDim, "embed-dim", c.EmbedDim, "Matryoshka dimension (64-768)")
	flag.IntVar(&c.RateLimit, "rate-limit", c.RateLimit, "requests/min per owner")
	flag.StringVar(&c.LogLevel, "log-level", c.LogLevel, "debug|info|warn|error")
	flag.StringVar(&c.GemmaModelPath, "gemma-model-path", "", "path to Gemma GGUF file")
	flag.Parse()

	if v := os.Getenv("DATA_DIR"); v != "" {
		c.DataDir = v
	}
	if v := os.Getenv("API_KEYS"); v != "" {
		for _, k := range strings.Split(v, ",") {
			if k = strings.TrimSpace(k); k != "" {
				c.APIKeys[k] = true
			}
		}
	}
	if v := os.Getenv("GEMMA_URL"); v != "" {
		c.GemmaURL = v
	}
	if v := os.Getenv("GEMMA_MODEL_PATH"); v != "" {
		c.GemmaModelPath = v
	}

	return c
}
