#!/bin/bash

# Usage message
if [ $# -ne 1 ]; then
    echo "Usage: $0 /full/path/to/container.sif"
    exit 1
fi

SIF_PATH="$1"

# Validate container path
if [ ! -f "$SIF_PATH" ]; then
    echo "Error: SIF file not found at $SIF_PATH"
    exit 2
fi

# Create kernel directory
KERNEL_NAME="tf-cuda122"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
mkdir -p "$KERNEL_DIR"

# Generate kernel.json dynamically
cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "display_name": "tf-cuda122",
  "language": "python",
  "argv": [
    "/opt/apps/tacc-apptainer/1.3.3/bin/apptainer",
    "exec",
    "--nv",
    "--bind",
    "/run/user:/run/user",
    "$SIF_PATH",
    "python3",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ]
}
EOF

echo "Kernel 'tf-cuda122' installed to $KERNEL_DIR"
echo "Using container: $SIF_PATH"
