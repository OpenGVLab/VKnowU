#!/usr/bin/env python3
"""
vLLM服务配置文件
用于管理vLLM服务的各种配置参数
"""

import os
from typing import Dict, Any

class VLLMConfig:
    """vLLM服务配置类"""
    
    # 服务配置
    SERVICE_HOST = "127.0.0.1"
    SERVICE_PORT = 5000
    SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"
    
    # 模型配置
    MODEL_PATH = "/fs-computility/video/shared/hf_weight/Qwen3-32B"
    MODEL_CONFIG = {
        "trust_remote_code": True,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "enforce_eager": True,
    }
    
    # 推理配置
    SAMPLING_CONFIG = {
        "temperature": 0.0,
        "max_tokens": 64,
        "stop": ["<|im_end|>", "<|endoftext|>", "\n\n"]
    }
    
    # 请求配置
    REQUEST_CONFIG = {
        "timeout": 30,
        "retry_times": 3,
        "retry_delay": 1.0
    }
    
    # 环境变量
    ENV_VARS = {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_TRITON_KERNEL": "1",  # 启用Triton内核优化
    }
    
    @classmethod
    def get_service_url(cls) -> str:
        """获取服务URL"""
        return cls.SERVICE_URL
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """获取模型配置"""
        return cls.MODEL_CONFIG.copy()
    
    @classmethod
    def get_sampling_config(cls) -> Dict[str, Any]:
        """获取采样配置"""
        return cls.SAMPLING_CONFIG.copy()
    
    @classmethod
    def get_request_config(cls) -> Dict[str, Any]:
        """获取请求配置"""
        return cls.REQUEST_CONFIG.copy()
    
    @classmethod
    def setup_environment(cls):
        """设置环境变量"""
        for key, value in cls.ENV_VARS.items():
            os.environ[key] = value
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否有效"""
        try:
            # 检查模型路径是否存在
            if not os.path.exists(cls.MODEL_PATH):
                print(f"❌ 模型路径不存在: {cls.MODEL_PATH}")
                return False
            
            # 检查端口是否在有效范围内
            if not (1 <= cls.SERVICE_PORT <= 65535):
                print(f"❌ 端口号无效: {cls.SERVICE_PORT}")
                return False
            
            # 检查GPU内存使用率
            if not (0.1 <= cls.MODEL_CONFIG["gpu_memory_utilization"] <= 1.0):
                print(f"❌ GPU内存使用率无效: {cls.MODEL_CONFIG['gpu_memory_utilization']}")
                return False
            
            print("✅ 配置验证通过")
            return True
            
        except Exception as e:
            print(f"❌ 配置验证失败: {e}")
            return False

# 默认配置实例
config = VLLMConfig()

if __name__ == "__main__":
    # 测试配置
    print("🔧 vLLM配置测试")
    print("=" * 40)
    
    print(f"服务URL: {config.get_service_url()}")
    print(f"模型路径: {config.MODEL_PATH}")
    print(f"GPU内存使用率: {config.MODEL_CONFIG['gpu_memory_utilization']}")
    print(f"最大序列长度: {config.MODEL_CONFIG['max_model_len']}")
    
    # 验证配置
    if config.validate_config():
        print("\n🎉 配置测试通过!")
    else:
        print("\n❌ 配置测试失败!") 