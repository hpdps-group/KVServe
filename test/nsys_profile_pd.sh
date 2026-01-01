#!/bin/bash
# 使用 nsys launch 来 profile Ray workers

nsys launch \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o pd_test \
  python test/test_pd_separation.py
