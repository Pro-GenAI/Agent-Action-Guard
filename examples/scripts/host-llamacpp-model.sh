#!/bin/bash

# This file enables running a local llama.cpp server hosting a small model.

target_dir="llamacpp_files"

# if current dir is a folder under target_dir, cd up one level
if [ "$(basename "$(dirname "$PWD")")" == "$target_dir" ]; then
	cd ..
fi

# if current dir is not target_dir, create and cd into it
if [ "$(basename "$PWD")" != "$target_dir" ]; then
	mkdir -p "$target_dir"
	cd "$target_dir" || exit 1
fi


llama_cpp_url="https://github.com/ggml-org/llama.cpp/releases/download/b7180/llama-b7180-bin-ubuntu-x64.zip"
llama_cpp_zip="llama-b7180-bin-ubuntu-x64.zip"
# if llama-server does not exist, download and unzip it
if [ ! -f "build/bin/llama-server" ]; then
	echo "Downloading llama.cpp binaries..."
	curl -L -o "$llama_cpp_zip" "$llama_cpp_url"
	unzip "$llama_cpp_zip"
	chmod +x build/bin/llama-server
fi


gguf_file="all-MiniLM-L12-v2.Q4_K_M.gguf"
gguf_url="https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q4_K_M.gguf?download=true"
# if file does not exist, download it
if [ ! -f "$gguf_file" ]; then
	echo "Downloading $gguf_file..."
	curl -L -o "$gguf_file" "$gguf_url"
fi

./build/bin/llama-server --embeddings -m "$gguf_file" --port 1234

# ./llamacpp_files/build/bin/llama-server --embeddings -m ./llamacpp_files/all-MiniLM-L12-v2.Q4_K_M.gguf --port 1234

