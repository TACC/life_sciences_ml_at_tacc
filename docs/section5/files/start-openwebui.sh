#!/bin/sh

OPEN_WEBUI_SIF="$SCRATCH/open-webui_v0.6.7-ollama.sif"
OPEN_WEBUI_DIR="$SCRATCH/openwebui-data"
OLLAMA_DATA="$OPEN_WEBUI_DIR/ollama"
OPEN_WEBUI_DATA="$OPEN_WEBUI_DIR/open-webui"
OPEN_WEBUI_STATIC="$OPEN_WEBUI_DIR/static"
SECRET_KEY="$OPEN_WEBUI_DIR/.webui_secret_key"

if command -v module &>/dev/null; then
    module unload xalt
    module load tacc-apptainer
    module list
fi

if [[ ! -s "$OPEN_WEBUI_SIF" ]]; then
    echo "Pulling open-webui image..."
    apptainer pull docker://ghcr.io/open-webui/open-webui:v0.6.7-ollama
fi

echo "Creating data directories, if needed..."
mkdir -p $OPEN_WEBUI_DIR
mkdir -p $OLLAMA_DATA
mkdir -p $OPEN_WEBUI_DATA
mkdir -p $OPEN_WEBUI_STATIC
if [[ ! -s "$SECRET_KEY" ]]; then
    echo "Creating secret key..."
    echo $(head -c 12 /dev/random | base64) > $SECRET_KEY
else
    echo "Key already exists..."
fi

WEBUI_SECRET_KEY=$(cat "$SECRET_KEY")

echo "Starting open-webui instance..."
apptainer instance start  --nv \
    --bind $OLLAMA_DATA:$HOME/.ollama \
    --bind $OPEN_WEBUI_DATA:/app/backend/data \
    --bind $OPEN_WEBUI_STATIC:/app/backend/static \
    --env "WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY" \
    $OPEN_WEBUI_SIF \
    openwebui1 \
    > $SCRATCH/openwebui1.log 2>&1

sleep 5

echo "Running start script..."
apptainer exec --env "WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY" \
    --env "CONFIG_LOG_LEVEL=DEBUG" \
    --env "OLLAMA_LOG_LEVEL=DEBUG" \
    --env "STATIC_DIR=/app/backend/static" \
    instance://openwebui1 \
    /app/backend/start.sh >> $SCRATCH/openwebui1.log 2>&1 &

sleep 10

echo "To stop open-webui, type 'apptainer instance stop openwebui1'" 
