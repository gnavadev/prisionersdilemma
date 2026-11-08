#!/usr/bin/env bash
# Start/continue training (Ctrl+C to stop and save)
cd "$(dirname "$0")" && exec .venv/bin/python arena.py "$@"
