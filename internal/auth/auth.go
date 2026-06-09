package auth

import (
	"net/http"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"context"
)

func HTTPMiddleware(next http.Handler, validKeys map[string]bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(validKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}
		key := r.Header.Get("X-API-Key")
		if key == "" || !validKeys[key] {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func GRPCUnaryInterceptor(validKeys map[string]bool) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if len(validKeys) == 0 {
			return handler(ctx, req)
		}
		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "missing metadata")
		}
		keys := md.Get("x-api-key")
		if len(keys) == 0 || !validKeys[keys[0]] {
			return nil, status.Error(codes.Unauthenticated, "invalid API key")
		}
		return handler(ctx, req)
	}
}
