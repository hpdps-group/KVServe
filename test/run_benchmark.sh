#!/bin/bash
# Multi-Stream Performance Benchmark Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Multi-Stream Benchmark Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python
if [ -f "/root/lzd/torch2.7.1/bin/python" ]; then
    PYTHON="/root/lzd/torch2.7.1/bin/python"
    echo -e "${GREEN}✅ Using Python: $PYTHON${NC}"
else
    echo -e "${RED}❌ Python not found at /root/lzd/torch2.7.1/bin/python${NC}"
    exit 1
fi

# Check if model exists
MODEL_PATH="/root/lzd/model/qwen2.5-VL"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Warning: Model not found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}   Please update model_path in benchmark_multistream.py${NC}"
fi

# Stop any existing Ray instances
echo -e "\n${YELLOW}🛑 Stopping existing Ray instances...${NC}"
ray stop 2>/dev/null || true

# Set environment
export PYTHONPATH="/root/lzd/vllm-0.10.1:$PROJECT_ROOT:$PYTHONPATH"

# Run benchmark
echo -e "\n${GREEN}🏃 Running benchmark...${NC}"
echo -e "${BLUE}========================================${NC}\n"

cd "$PROJECT_ROOT"
$PYTHON test/benchmark_multistream.py 2>&1 | tee benchmark_output.log

# Check if results file was created
if [ -f "benchmark_results.json" ]; then
    echo -e "\n${GREEN}✅ Benchmark completed successfully!${NC}"
    echo -e "${GREEN}📊 Results saved to: benchmark_results.json${NC}"
    echo -e "${GREEN}📝 Full log saved to: benchmark_output.log${NC}"
else
    echo -e "\n${RED}❌ Benchmark failed or incomplete${NC}"
    exit 1
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 Done!${NC}"
echo -e "${BLUE}========================================${NC}\n"

