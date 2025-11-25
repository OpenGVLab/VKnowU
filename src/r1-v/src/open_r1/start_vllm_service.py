#!/usr/bin/env python3
"""
vLLM服务启动脚本
用于启动QWEN3模型的vLLM推理服务
"""

import os
import sys
import argparse
from vllm import LLM, SamplingParams

def start_vllm_service():
    """启动vLLM服务"""
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    print("正在启动vLLM服务...")
    print("模型路径: /fs-computility/video/shared/hf_weight/Qwen3-32B")
    
    try:
        # 初始化vLLM模型
        llm = LLM(
            model="/fs-computility/video/shared/hf_weight/Qwen3-32B",
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            enforce_eager=True,
        )
        
        print("✅ vLLM模型加载成功!")
        print(f"模型设备: {llm.llm_engine.model_executor.driver_worker.model_runner.device}")
        print(f"最大序列长度: {llm.llm_engine.model_executor.driver_worker.model_runner.max_model_len}")
        
        # 测试推理
        print("\n正在测试推理...")
        test_prompt = "Hello, how are you?"
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate([test_prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        print(f"测试输入: {test_prompt}")
        print(f"测试输出: {response}")
        print("✅ 推理测试成功!")
        
        return llm
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动vLLM服务")
    parser.add_argument("--test", action="store_true", help="仅测试模型加载")
    args = parser.parse_args()
    
    if args.test:
        llm = start_vllm_service()
        if llm:
            print("\n🎉 vLLM服务测试完成，可以启动Flask应用了!")
        else:
            print("\n❌ vLLM服务测试失败，请检查配置!")
            sys.exit(1)
    else:
        print("使用方法:")
        print("python start_vllm_service.py --test  # 测试vLLM服务")
        print("python qwen3_caption_service.py      # 启动Flask应用") 