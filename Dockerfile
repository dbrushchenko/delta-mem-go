FROM golang:1.25 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -o /delta-mem-go ./cmd/delta-mem-go
RUN CGO_ENABLED=1 go build -o /mem-cli ./cmd/mem-cli

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y ca-certificates libgomp1 wget && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz && \
    tar xzf onnxruntime-linux-x64-1.26.0.tgz && \
    cp onnxruntime-linux-x64-1.26.0/lib/libonnxruntime.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-* /var/lib/apt/lists/*

COPY --from=builder /delta-mem-go /usr/local/bin/delta-mem-go
COPY --from=builder /mem-cli /usr/local/bin/mem-cli

EXPOSE 8080 9090
VOLUME /data
ENV ORT_LIB_DIR=/usr/local/lib
CMD ["delta-mem-go", "--model", "/models/nomic-embed-text-v1.5.onnx", "--data", "/data", "--port", "8080", "--grpc-port", "9090"]
