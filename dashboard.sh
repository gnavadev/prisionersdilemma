#!/usr/bin/env bash
# Open the interactive dashboard in your browser (http://localhost:8501)
cd "$(dirname "$0")" && exec .venv/bin/streamlit run dashboard.py "$@"
