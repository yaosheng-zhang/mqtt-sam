# main.py
from mqtt import MQTTClient
from handlers.label_handler.auto_label_handler import AutoLabelHandler
from handlers.geo_handler.geo_label_handler import GeoLabelHandler
from config_loader import get_config
from ai_engine import SAM3InferenceEngine
import subprocess
import sys
import time
import os


def start_sharp_api_service(config):
    """
    启动 Sharp API 服务（后台进程）

    Args:
        config: 配置对象

    Returns:
        subprocess.Popen: API 服务进程对象
    """
    try:
        # 检查是否配置了 Sharp Python 路径
        sharp_python_path = config.gs_sharp_python_path
        if not os.path.exists(sharp_python_path):
            print(f"[警告] Sharp Python 路径不存在: {sharp_python_path}")
            print(f"[警告] GS 服务将不可用")
            return None

        print(f"[GS] 正在启动 Sharp API 服务...")
        print(f"   地址: http://{config.gs_api_host}:{config.gs_api_port}")

        # 启动 Sharp API 服务作为子进程
        api_process = subprocess.Popen(
            [sys.executable, "sharp_api_service.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 等待 API 服务启动
        time.sleep(2)

        # 检查进程是否成功启动
        if api_process.poll() is not None:
            # 进程已退出
            stdout, stderr = api_process.communicate()
            print(f"[错误] Sharp API 服务启动失败")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None

        print(f"[GS] Sharp API 服务已启动 (PID: {api_process.pid})")
        return api_process

    except Exception as e:
        print(f"[警告] Sharp API 服务启动失败: {e}")
        return None

def main():
    config = get_config()

    print("="*70)
    print("SAM3 多功能AI边缘推理节点 启动中")
    print("="*70)

    # 检查是否启用变化检测服务
    enable_change_detection = False
    pe_model_path = None

    try:
        pe_model_path = config.change_detection_pe_model_path
        enable_change_detection = True
        print(f"[配置] 检测到变化检测服务配置")
    except Exception:
        print(f"[配置] 未配置变化检测服务,将仅加载 SAM3 模型")

    # 检查是否启用 GS 服务
    enable_gs_service = False
    sharp_api_process = None

    try:
        sharp_python_path = config.gs_sharp_python_path
        enable_gs_service = True
        print(f"[配置] 检测到 GS 服务配置")

        # 启动 Sharp API 服务
        sharp_api_process = start_sharp_api_service(config)
        if sharp_api_process is None:
            print(f"[警告] Sharp API 服务启动失败，GS 服务将不可用")
            enable_gs_service = False
    except Exception:
        print(f"[配置] 未配置 GS 服务")

    # 全局唯一模型引擎
    engine = SAM3InferenceEngine(
        model_path=config.ai_model_path,
        device=config.ai_device,
        image_path=config.ai_image_path or "",
        threshold=config.ai_threshold,
        mask_threshold=config.ai_mask_threshold,
        enable_pe=enable_change_detection,
        pe_model_path=pe_model_path
    )

    client = MQTTClient(config)

    # 注册基础服务
    client.register_handler(AutoLabelHandler(engine, config))
    client.register_handler(GeoLabelHandler(engine, config))

    # 注册变化检测服务 (如果 PE 模型加载成功)
    if enable_change_detection and engine.pe_model is not None:
        try:
            from handlers.change_detection_handler import ChangeDetectionHandler
            client.register_handler(ChangeDetectionHandler(engine, config))
            print(f"[服务] 变化检测服务已启用")
        except Exception as e:
            print(f"[警告] 变化检测服务注册失败: {e}")

    # 注册 GS 服务 (如果 Sharp API 启动成功)
    if enable_gs_service and sharp_api_process is not None:
        try:
            from handlers.gs_handler import GSHandler
            client.register_handler(GSHandler(engine, config))
            print(f"[服务] Gaussian Splatting 服务已启用")
        except Exception as e:
            print(f"[警告] GS 服务注册失败: {e}")

    print(f"\n设备ID: {config.mqtt_dev_id}")
    print(f"自动标注 → {config.mqtt_subscribe_topic}")
    print(f"地理标注 → {config.geo_subscribe_topic}")
    if enable_change_detection and engine.pe_model is not None:
        print(f"变化检测 → {config.change_detection_subscribe_topic}")
    if enable_gs_service and sharp_api_process is not None:
        print(f"GS 重建   → {config.gs_subscribe_topic}")
    print(f"推理设备: {config.ai_device.upper()}")
    print("="*70)

    try:
        client.start()
    except KeyboardInterrupt:
        print("\n[系统] 正在关闭...")
    finally:
        client.stop()

        # 停止 Sharp API 服务
        if sharp_api_process is not None:
            print("[GS] 正在停止 Sharp API 服务...")
            sharp_api_process.terminate()
            try:
                sharp_api_process.wait(timeout=5)
                print("[GS] Sharp API 服务已停止")
            except subprocess.TimeoutExpired:
                print("[GS] Sharp API 服务未响应，强制终止")
                sharp_api_process.kill()

if __name__ == "__main__":
    main()