#!/bin/bash
# Script to reset CUDA/GPU state
# Run with: bash reset_cuda.sh

echo "Killing all GPU processes..."
pkill -9 -f "python.*kvserve"
pkill -9 -f "ray"
sleep 2

echo "Stopping Ray..."
ray stop -f
sleep 2

echo "Checking GPU status..."
nvidia-smi

echo ""
echo "==========================================="
echo "CUDA state has been cleared as much as possible"
echo "If you still see errors, you need to:"
echo "1. Reboot the system (recommended)"
echo "2. Or contact system admin to reload nvidia kernel modules"
echo "==========================================="

