#!/bin/bash

REMOTE_USER="n.fahrni"
REMOTE_HOST="slurmlogin.cs.technik.fhnw.ch"
REMOTE_DIR="~/classes/dlbs"
LOCAL_DIR="."

rsync -av --progress \
    --include='models/' --include='models/**' \
    --exclude='*' \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR"

echo "Download completed! Remote directory '$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR' downloaded to '$LOCAL_DIR'."
