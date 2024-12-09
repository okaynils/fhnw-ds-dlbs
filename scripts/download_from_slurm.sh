#!/bin/bash

REMOTE_USER="n.fahrni"
REMOTE_HOST="slurmlogin.cs.technik.fhnw.ch"
REMOTE_DIR="~/classes/dlbs/models"
LOCAL_DIR="."

rsync -av --include='/output/***' --include='/models/***' --include='/wandb/***' --include='/logs/***' --exclude='*' "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"

echo "Download completed! Remote directory '$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR' downloaded to '$LOCAL_DIR'."
