#!/bin/bash

SRC_DIR="/work/03762/eriksf/share/life_sci_ml_institute_2025"
IMAGE_TF="tensorflow-ml-container_0.1.sif"
IMAGE_PT="pytorch-ml-container_0.4.sif"

# Copy the images to SCRATCH
for img in $IMAGE_TF $IMAGE_PT
do
    echo "Copying $img to $SCRATCH..."
    cp $SRC_DIR/$img $SCRATCH
    chmod 644 $SCRATCH/$img
done

# Install the tensorflow kernel

# Define the kernel directory name
KERNEL_DIR="Day4-tf-217" 

# Create the directory
mkdir -p ~/.local/share/jupyter/kernels/$KERNEL_DIR

# Write the JSON content to the kernel directory
cat <<EOL > ~/.local/share/jupyter/kernels/$KERNEL_DIR/kernel.json
{
  "argv": [
    "/opt/apps/tacc-apptainer/1.3.3/bin/apptainer",
    "exec",
    "--nv",
    "--bind",
    "/run/user:/run/user",
    "--env",
    "TF_USE_LEGACY_KERAS=1",
    "$SCRATCH/$IMAGE_TF",
    "python3",
    "-m",
    "ipykernel_launcher",
    "--debug",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$KERNEL_DIR",
  "language": "python"
}
EOL

echo "Kernel directory created at ~/.local/share/jupyter/kernels/$KERNEL_DIR and kernel.json has been added."

# Install the pytorch kernel

# Define the kernel directory name
KERNEL_DIR="Day4-pt-251" 

# Create the directory
mkdir -p ~/.local/share/jupyter/kernels/$KERNEL_DIR

# Write the JSON content to the kernel directory
cat <<EOL > ~/.local/share/jupyter/kernels/$KERNEL_DIR/kernel.json
{
  "argv": [
    "/opt/apps/tacc-apptainer/1.3.3/bin/apptainer",
    "exec",
    "--nv",
    "--bind",
    "/run/user:/run/user",
    "$SCRATCH/$IMAGE_PT",
    "python3",
    "-m",
    "ipykernel_launcher",
    "--debug",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$KERNEL_DIR",
  "language": "python"
}
EOL

echo "Kernel directory created at ~/.local/share/jupyter/kernels/$KERNEL_DIR and kernel.json has been added."

