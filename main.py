# main.py
from mqtt import MQTTClient
from handlers.label_handler.auto_label_handler import AutoLabelHandler
from handlers.geo_handler.geo_label_handler import GeoLabelHandler
from config_loader import get_config
from ai_engine import SAM3InferenceEngine

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

    print(f"\n设备ID: {config.mqtt_dev_id}")
    print(f"自动标注 → {config.mqtt_subscribe_topic}")
    print(f"地理标注 → {config.geo_subscribe_topic}")
    if enable_change_detection and engine.pe_model is not None:
        print(f"变化检测 → {config.change_detection_subscribe_topic}")
    print(f"推理设备: {config.ai_device.upper()}")
    print("="*70)

    try:
        client.start()
    except KeyboardInterrupt:
        print("\n[系统] 正在关闭...")
    finally:
        client.stop()

if __name__ == "__main__":
    main()