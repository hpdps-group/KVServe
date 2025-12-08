#!/bin/bash
# Run test script with proper Python path

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 test/test_pd_separation.py "$@"



