#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
    printf '%s\n' 'Create .venv and install requirements first; see README.md.'
    exit 1
fi
exec .venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false "$@"
