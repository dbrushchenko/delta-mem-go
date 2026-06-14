#!/usr/bin/env python3
"""Prepare plain-text data for delta-mem-go training. Chunks text into key-content JSONL pairs."""
import argparse, json, os
from pathlib import Path

def chunk_text(text, max_tokens=2048):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_tokens:
            if current: chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current: chunks.append(current.strip())
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to .txt file or directory")
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    input_path = Path(args.input)
    texts = []
    if input_path.is_file():
        texts.append(input_path.read_text(encoding="utf-8"))
    elif input_path.is_dir():
        for f in input_path.glob("**/*.txt"):
            texts.append(f.read_text(encoding="utf-8"))

    dataset = []
    for text in texts:
        for i, chunk in enumerate(chunk_text(text, args.max_tokens)):
            dataset.append({"key": f"chunk-{i}", "content": chunk})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Done! {len(dataset)} training examples -> {args.output}")

if __name__ == "__main__":
    main()
