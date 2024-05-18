#!/bin/sh
docker run --name pytorch-container --gpus all -it --rm -v $(pwd):/app pytorch-gpu
