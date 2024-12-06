#!/bin/bash

LOCAL_DIR="."
REMOTE_USER="nils"
REMOTE_HOST="192.168.1.36"
REMOTE_DIR="~/Documents/Classes/dlbs"

rsync -av --exclude '/output' --exclude '/models' --exclude '/wandb' --exclude '/.venv' "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "Upload completed! Local directory '$LOCAL_DIR' uploaded to '$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR'."
