#!/bin/bash

# QWEN3 vLLM 快速启动脚本
# 使用方法: ./quick_start.sh

echo "🚀 QWEN3 vLLM 服务快速启动脚本"
echo "=================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3，请先安装Python 3.8+"
    exit 1
fi

# 检查CUDA环境
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: 未找到nvidia-smi，请检查CUDA环境"
    exit 1
fi

echo "✅ 环境检查通过"

# 显示GPU信息
echo "📊 GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=, read -r name total free; do
    echo "  GPU: $name, 总显存: ${total}MB, 可用显存: ${free}MB"
done

# 安装依赖
echo "📦 安装依赖..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo "✅ 依赖安装完成"

# 测试vLLM服务
echo "🧪 测试vLLM服务..."
python3 start_vllm_service.py --test

if [ $? -ne 0 ]; then
    echo "❌ vLLM服务测试失败"
    exit 1
fi

echo "✅ vLLM服务测试通过"

# 启动Flask服务
echo "🌐 启动Flask服务..."
echo "服务将在 http://127.0.0.1:5000 启动"
echo "按 Ctrl+C 停止服务"
echo ""

python3 qwen3_caption_service.py 