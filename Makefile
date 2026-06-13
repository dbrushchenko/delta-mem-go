.PHONY: all build test docker-up docker-down proto metrics self-train clean

BINARY ?= delta-mem-go
VERSION ?= $(shell git describe --tags --always --dirty)
IMAGE ?= your-registry/delta-mem-go:$(VERSION)
DATA_DIR ?= ./data/states

all: build

build:
	CGO_ENABLED=1 go build -tags cgo -ldflags "-X main.version=$(VERSION)" -o $(BINARY) ./cmd/delta-mem-go

installer: build
	cp $(BINARY) installer/delta-mem-go.exe
	go build -o delta-mem-go-setup.exe ./installer/

test:
	go test ./... -race -count=1 -v

proto:
	protoc --go_out=. --go-grpc_out=. \
		--go_opt=paths=source_relative \
		--go-grpc_opt=paths=source_relative \
		proto/deltamem.proto

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

self-train:
	@echo "=== Full self-training pipeline ==="
	python scripts/merge_text_files.py data/raw/ --output data/merged_training_data.txt
	python scripts/prepare_training_data.py data/merged_training_data.txt \
		--output data/train_synthetic.jsonl \
		--synthetic \
		--service-url http://localhost:8080/generate

clean:
	rm -f $(BINARY)
	rm -rf data/states/*
	rm -f data/*.jsonl data/*.txt

.DEFAULT_GOAL := build
