FROM golang:1.23 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -tags cgo -o /delta-mem-go ./cmd/delta-mem-go

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /delta-mem-go /usr/local/bin/delta-mem-go
COPY models/ /models/
EXPOSE 8080 9090
VOLUME /data/states
CMD ["delta-mem-go", "--model", "/models/nomic-embed-text-v1.5.onnx", "--embed-dim", "768"]
