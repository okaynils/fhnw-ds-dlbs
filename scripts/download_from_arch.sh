#!/bin/bash

LOCAL_DIR="."
REMOTE_USER="nils"
REMOTE_HOST="192.168.1.36"
REMOTE_DIR="~/Documents/Classes/dlbs"

rsync -av \
    --include='/models/***' \
    --exclude='*' \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR"

echo "Download completed! Remote directory '$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR' synced to local directory '$LOCAL_DIR'."
