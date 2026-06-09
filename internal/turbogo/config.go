package turbogo

type Config struct {
	Dim      int
	BitWidth int
}

func DefaultConfig() Config {
	return Config{Dim: 768, BitWidth: 4}
}
