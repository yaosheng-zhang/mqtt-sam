import psutil
import torch

class SystemMonitor:
    """系统资源监控类"""

    @staticmethod
    def get_cpu_usage():
        """
        获取CPU使用率

        Returns:
            float: CPU使用率 (0.0-1.0)
        """
        try:
            # 获取CPU使用率百分比，interval=1表示采样1秒
            cpu_percent = psutil.cpu_percent(interval=0.5) / 100.0
            return round(cpu_percent, 3)
        except Exception as e:
            print(f"[监控] 获取CPU使用率失败: {e}")
            return 0.0

    @staticmethod
    def get_memory_usage():
        """
        获取内存使用率

        Returns:
            float: 内存使用率 (0.0-1.0)
        """
        try:
            mem = psutil.virtual_memory()
            return round(mem.percent / 100.0, 3)
        except Exception as e:
            print(f"[监控] 获取内存使用率失败: {e}")
            return 0.0

    @staticmethod
    def get_gpu_usage():
        """
        获取GPU使用率

        Returns:
            float: GPU使用率 (0.0-1.0)，如果没有GPU则返回0.0
        """
        try:
            if not torch.cuda.is_available():
                return 0.0

            # 获取GPU内存使用情况
            gpu_memory_allocated = torch.cuda.memory_allocated(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory

            if gpu_memory_total == 0:
                return 0.0

            gpu_usage = gpu_memory_allocated / gpu_memory_total
            return round(gpu_usage, 3)
        except Exception as e:
            print(f"[监控] 获取GPU使用率失败: {e}")
            return 0.0

    @staticmethod
    def get_system_status():
        """
        获取完整的系统状态信息

        Returns:
            dict: 包含CPU、内存、GPU使用率的字典
        """
        return {
            "cpu_usage": SystemMonitor.get_cpu_usage(),
            "memory_usage": SystemMonitor.get_memory_usage(),
            "gpu_usage": SystemMonitor.get_gpu_usage()
        }


# 测试代码
if __name__ == "__main__":
    monitor = SystemMonitor()
    status = monitor.get_system_status()
    print(f"CPU使用率: {status['cpu_usage']*100:.1f}%")
    print(f"内存使用率: {status['memory_usage']*100:.1f}%")
    print(f"GPU使用率: {status['gpu_usage']*100:.1f}%")
