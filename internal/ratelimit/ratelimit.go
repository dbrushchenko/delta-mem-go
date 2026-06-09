package ratelimit

import (
	"sync"

	"golang.org/x/time/rate"
)

type Limiter struct {
	limits map[string]*rate.Limiter
	mu     sync.RWMutex
	rps    int
}

func New(rps int) *Limiter {
	return &Limiter{limits: make(map[string]*rate.Limiter), rps: rps}
}

func (l *Limiter) Allow(owner string) bool {
	l.mu.RLock()
	lim, exists := l.limits[owner]
	l.mu.RUnlock()
	if !exists {
		l.mu.Lock()
		lim = rate.NewLimiter(rate.Limit(l.rps), l.rps*2)
		l.limits[owner] = lim
		l.mu.Unlock()
	}
	return lim.Allow()
}
