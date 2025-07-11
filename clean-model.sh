#!/bin/bash
set -e

MODEL_DIR="model"

if [ ! -d "$MODEL_DIR" ]; then
  echo "No model directory to clean."
  exit 0
fi

echo "Cleaning model directory: $MODEL_DIR"
find "$MODEL_DIR" -mindepth 1 -not -name '.gitignore' -not -name 'README.md' -exec rm -rf {} +

echo "Model directory cleaned (except .gitignore and README.md)."