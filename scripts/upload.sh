#!/bin/bash
LOCAL_DIR="."
REMOTE_USER="n.fahrni"
REMOTE_HOST="slurmlogin.cs.technik.fhnw.ch"
REMOTE_DIR="~/classes/dlbs"
rsync -av --exclude '/output' --exclude '/.git' --exclude '/models' --exclude '/wandb' --exclude '/.venv' "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo "Upload completed! Local directory '$LOCAL_DIR' uploaded to '$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR'."