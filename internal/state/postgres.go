package state

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"time"

	_ "github.com/lib/pq"
)

// PGBackend stores state in PostgreSQL for durability beyond Redis 24h TTL.
type PGBackend struct {
	db *sql.DB
}

func NewPGBackend(connStr string) (*PGBackend, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil { return nil, err }
	if err := db.Ping(); err != nil { return nil, err }
	// Create table if not exists
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS mem_state (
		owner TEXT NOT NULL,
		key TEXT NOT NULL,
		data BYTEA NOT NULL,
		updated_at TIMESTAMPTZ DEFAULT NOW(),
		PRIMARY KEY (owner, key)
	)`)
	if err != nil { return nil, err }
	return &PGBackend{db: db}, nil
}

func (p *PGBackend) Save(owner, key string, data []byte) error {
	_, err := p.db.Exec(
		`INSERT INTO mem_state (owner, key, data, updated_at) VALUES ($1, $2, $3, $4)
		 ON CONFLICT (owner, key) DO UPDATE SET data = $3, updated_at = $4`,
		owner, key, data, time.Now())
	return err
}

func (p *PGBackend) Load(owner, key string) ([]byte, error) {
	var data []byte
	err := p.db.QueryRow(`SELECT data FROM mem_state WHERE owner=$1 AND key=$2`, owner, key).Scan(&data)
	if err == sql.ErrNoRows { return nil, os.ErrNotExist }
	return data, err
}

// HybridBackend uses Redis for fast access + PG for durability.
// Reads: Redis first, fallback to PG. Writes: both.
type HybridBackend struct {
	Redis *RedisBackend
	PG    *PGBackend
}

func NewHybridBackend(redisAddr, redisPass string, pgConnStr string) (*HybridBackend, error) {
	r := NewRedisBackend(redisAddr, redisPass, 0)
	if err := r.Ping(); err != nil {
		return nil, fmt.Errorf("redis: %w", err)
	}
	pg, err := NewPGBackend(pgConnStr)
	if err != nil {
		return nil, fmt.Errorf("postgres: %w", err)
	}
	return &HybridBackend{Redis: r, PG: pg}, nil
}

func (h *HybridBackend) Save(owner, key string, data []byte) error {
	// Write to both — Redis for speed, PG for durability
	if err := h.Redis.Save(owner, key, data); err != nil {
		return err
	}
	return h.PG.Save(owner, key, data)
}

func (h *HybridBackend) Load(owner, key string) ([]byte, error) {
	// Try Redis first (fast)
	data, err := h.Redis.Load(owner, key)
	if err == nil { return data, nil }
	// Fallback to PG (durable) and re-cache in Redis
	data, err = h.PG.Load(owner, key)
	if err == nil {
		h.Redis.Save(owner, key, data) // re-warm cache
	}
	return data, err
}

// FlushAll forces all PG state into Redis (cold start recovery).
func (h *HybridBackend) FlushAll(ctx context.Context) error {
	rows, err := h.PG.db.QueryContext(ctx, `SELECT owner, key, data FROM mem_state`)
	if err != nil { return err }
	defer rows.Close()
	count := 0
	for rows.Next() {
		var owner, key string
		var data []byte
		if err := rows.Scan(&owner, &key, &data); err != nil { continue }
		h.Redis.Save(owner, key, data)
		count++
	}
	fmt.Printf("[state] Flushed %d entries from PG to Redis\n", count)
	return nil
}
