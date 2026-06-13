package state

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/redis/go-redis/v9"
)

// Backend abstracts state persistence. Windows uses FileBackend, mesh uses RedisBackend.
type Backend interface {
	Save(owner, key string, data []byte) error
	Load(owner, key string) ([]byte, error)
}

// FileBackend persists to local gob files (Windows desktop mode).
type FileBackend struct {
	Dir string
}

func NewFileBackend(dir string) *FileBackend {
	os.MkdirAll(dir, 0755)
	return &FileBackend{Dir: dir}
}

func (f *FileBackend) Save(owner, key string, data []byte) error {
	path := filepath.Join(f.Dir, fmt.Sprintf("%s.%s", owner, key))
	return os.WriteFile(path, data, 0644)
}

func (f *FileBackend) Load(owner, key string) ([]byte, error) {
	path := filepath.Join(f.Dir, fmt.Sprintf("%s.%s", owner, key))
	return os.ReadFile(path)
}

// RedisBackend persists to Redis (mesh mode). TTL = 24h. Flush to PostgreSQL for durability.
type RedisBackend struct {
	rdb *redis.Client
	ttl time.Duration
}

func NewRedisBackend(addr, password string, db int) *RedisBackend {
	rdb := redis.NewClient(&redis.Options{Addr: addr, Password: password, DB: db})
	return &RedisBackend{rdb: rdb, ttl: 24 * time.Hour}
}

func (r *RedisBackend) Save(owner, key string, data []byte) error {
	ctx := context.Background()
	redisKey := fmt.Sprintf("memgo:%s:%s", owner, key)
	return r.rdb.Set(ctx, redisKey, data, r.ttl).Err()
}

func (r *RedisBackend) Load(owner, key string) ([]byte, error) {
	ctx := context.Background()
	redisKey := fmt.Sprintf("memgo:%s:%s", owner, key)
	val, err := r.rdb.Get(ctx, redisKey).Bytes()
	if err == redis.Nil { return nil, os.ErrNotExist }
	return val, err
}

// Ping checks Redis connectivity.
func (r *RedisBackend) Ping() error {
	return r.rdb.Ping(context.Background()).Err()
}
