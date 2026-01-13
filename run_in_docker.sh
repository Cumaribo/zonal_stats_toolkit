#!/bin/bash
set -e

# This script builds the docker image for the zonal stats toolkit
# and then runs the runner.py script inside the container.
# It passes all arguments to the runner.py script.
#
# Example usage:
# ./run_in_docker.sh custom_ncp_analysis.ini

IMAGE_NAME="zonal_stats_toolkit:latest"
WORKDIR="/usr/local/wwf_es_beneficiaries"

# Build the docker image
echo "Building docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# Run the script in the container
echo "Running script in docker container..."
docker run --rm -it \
    -v "$(pwd)":"$WORKDIR" \
    --user "$(id -u):$(id -g)" \
    "$IMAGE_NAME" \
    python runner.py "$@"
