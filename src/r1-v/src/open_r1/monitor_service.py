#!/usr/bin/env python3
"""
vLLM服务监控脚本
监控服务状态、性能指标和资源使用情况
"""

import requests
import time
import psutil
import GPUtil
import json
from datetime import datetime
from typing import Dict, Any, Optional

class VLLMServiceMonitor:
    """vLLM服务监控器"""
    
    def __init__(self, service_url: str = "http://127.0.0.1:5000"):
        self.service_url = service_url
        self.monitoring = False
        
    def check_service_health(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.service_url}/", timeout=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 404:  # Flask默认返回404，说明服务在运行
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": None
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "down",
                "response_time": None,
                "timestamp": datetime.now().isoformat(),
                "error": "Connection refused"
            }
        except Exception as e:
            return {
                "status": "error",
                "response_time": None,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_system_resources(self) -> Dict[str, Any]:
        """获取系统资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # GPU使用情况
            gpu_info = {}
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_info[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                        "load": gpu.load * 100
                    }
            except Exception as e:
                gpu_info = {"error": str(e)}
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "gpu": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_inference_performance(self) -> Dict[str, Any]:
        """测试推理性能"""
        test_data = {
            "content": "这是一张测试图片，显示了一个红色的苹果。",
            "sol": "<answer>苹果</answer>",
            "problem_type": "caption",
            "problem": "图片中有什么水果？"
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.service_url}/predict",
                json=test_data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "output": result.get("output"),
                    "timestamp": datetime.now().isoformat(),
                    "error": None
                }
            else:
                return {
                    "status": "failed",
                    "response_time": end_time - start_time,
                    "output": None,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "response_time": None,
                "output": None,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        health = self.check_service_health()
        resources = self.get_system_resources()
        performance = self.test_inference_performance()
        
        return {
            "health": health,
            "resources": resources,
            "performance": performance,
            "summary": {
                "service_status": health["status"],
                "cpu_usage": resources.get("cpu_percent", 0),
                "memory_usage": resources.get("memory", {}).get("percent", 0),
                "avg_response_time": performance.get("response_time", 0),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def print_report(self, report: Dict[str, Any]):
        """打印监控报告"""
        print(f"\n📊 vLLM服务监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 服务状态
        health = report["health"]
        print(f"🔍 服务状态: {health['status']}")
        if health['response_time']:
            print(f"   响应时间: {health['response_time']:.3f}s")
        if health['error']:
            print(f"   错误信息: {health['error']}")
        
        # 系统资源
        resources = report["resources"]
        if "error" not in resources:
            print(f"\n💻 系统资源:")
            print(f"   CPU使用率: {resources['cpu_percent']:.1f}%")
            
            memory = resources['memory']
            print(f"   内存使用率: {memory['percent']:.1f}%")
            print(f"   内存使用: {memory['used'] // (1024**3):.1f}GB / {memory['total'] // (1024**3):.1f}GB")
            
            # GPU信息
            if "error" not in resources['gpu']:
                print(f"\n🎮 GPU状态:")
                for gpu_id, gpu in resources['gpu'].items():
                    print(f"   {gpu_id}: {gpu['name']}")
                    print(f"     显存使用: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_percent']:.1f}%)")
                    print(f"     温度: {gpu['temperature']}°C")
                    print(f"     负载: {gpu['load']:.1f}%")
        
        # 性能测试
        performance = report["performance"]
        print(f"\n⚡ 性能测试:")
        print(f"   状态: {performance['status']}")
        if performance['response_time']:
            print(f"   响应时间: {performance['response_time']:.3f}s")
        if performance['output'] is not None:
            print(f"   输出: {performance['output']}")
        if performance['error']:
            print(f"   错误: {performance['error']}")
        
        # 总结
        summary = report["summary"]
        print(f"\n📈 总结:")
        print(f"   服务状态: {summary['service_status']}")
        print(f"   CPU使用率: {summary['cpu_usage']:.1f}%")
        print(f"   内存使用率: {summary['memory_usage']:.1f}%")
        print(f"   平均响应时间: {summary['avg_response_time']:.3f}s")
    
    def start_monitoring(self, interval: int = 30):
        """开始持续监控"""
        print(f"🚀 开始监控vLLM服务 (间隔: {interval}秒)")
        print(f"服务地址: {self.service_url}")
        print("按 Ctrl+C 停止监控")
        
        self.monitoring = True
        
        try:
            while self.monitoring:
                report = self.generate_report()
                self.print_report(report)
                
                if interval > 0:
                    print(f"\n⏰ {interval}秒后重新检查...")
                    time.sleep(interval)
                else:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⏹️ 监控已停止")
            self.monitoring = False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM服务监控工具")
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="服务URL")
    parser.add_argument("--once", action="store_true", help="只检查一次")
    parser.add_argument("--interval", type=int, default=30, help="监控间隔(秒)")
    
    args = parser.parse_args()
    
    monitor = VLLMServiceMonitor(args.url)
    
    if args.once:
        # 只检查一次
        report = monitor.generate_report()
        monitor.print_report(report)
    else:
        # 持续监控
        monitor.start_monitoring(args.interval)

if __name__ == "__main__":
    main() 