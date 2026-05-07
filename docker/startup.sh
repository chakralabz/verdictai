#!/bin/sh

set -e

MODELS_DIR=${DOCLING_ARTIFACTS_PATH:-/models/docling}

echo "Using models path: $MODELS_DIR"

mkdir -p "$MODELS_DIR"

if [ ! -f "$MODELS_DIR/.download_complete" ]; then
    echo "Models not found. Downloading..."

    uv run docling-tools models download -o "$MODELS_DIR"

    touch "$MODELS_DIR/.download_complete"

    echo "Models downloaded."
else
    echo "Models already exist. Skipping download."
fi

echo "Starting application..."

exec uv run python main.py