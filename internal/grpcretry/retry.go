package grpcretry

import (
	"context"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func RetryInterceptor() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		const maxRetries = 3
		backoff := 100 * time.Millisecond
		for attempt := 0; attempt <= maxRetries; attempt++ {
			err := invoker(ctx, method, req, reply, cc, opts...)
			if err == nil {
				return nil
			}
			st, ok := status.FromError(err)
			if !ok || (st.Code() != codes.Unavailable && st.Code() != codes.DeadlineExceeded) {
				return err
			}
			if attempt == maxRetries {
				return err
			}
			time.Sleep(backoff)
			backoff *= 2
		}
		return nil
	}
}
