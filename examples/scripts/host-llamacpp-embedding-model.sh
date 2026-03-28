#!/bin/bash

set -e

target_dir="llamacpp_files"

# Normalize working directory
if [ "$(basename "$(dirname "$PWD")")" == "$target_dir" ]; then
	cd ..
fi

if [ "$(basename "$PWD")" != "$target_dir" ]; then
	mkdir -p "$target_dir"
	cd "$target_dir" || exit 1
fi

# Check if llama-server exists globally
if command -v llama-server >/dev/null 2>&1; then
	echo "Using system-installed llama-server"
	LLAMA_SERVER_BIN=$(command -v llama-server)
else
	echo "llama-server not found. Installing locally..."

	llama_cpp_url="https://github.com/ggml-org/llama.cpp/releases/download/b7180/llama-b7180-bin-ubuntu-x64.zip"
	llama_cpp_zip="llama-b7180-bin-ubuntu-x64.zip"

	if [ ! -f "build/bin/llama-server" ]; then
		echo "Downloading llama.cpp binaries..."
		curl -L -o "$llama_cpp_zip" "$llama_cpp_url"
		unzip -o "$llama_cpp_zip"
		chmod +x build/bin/llama-server
	fi

	LLAMA_SERVER_BIN="./build/bin/llama-server"
fi

# Model setup
gguf_file="all-MiniLM-L6-v2.Q4_K_M.gguf"
gguf_url="https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf?download=true"
# Download model if not present
if [ ! -f "$gguf_file" ]; then
	echo "Downloading $gguf_file..."
	curl -L -o "$gguf_file" "$gguf_url"
fi

# Run server
"$LLAMA_SERVER_BIN" --embeddings -m "$gguf_file" --port 1234
