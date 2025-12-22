# config_loader.py
import os
import yaml
import torch
from pathlib import Path


class Config:
    """统一配置管理类（单例模式）"""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._resolve_device()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"配置文件不存在: {os.path.abspath(self.config_path)}\n"
                "请确保 config.yaml 在项目根目录，或指定正确路径"
            )
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _resolve_device(self):
        device = self.get('ai.device', 'auto')
        if device == 'auto':
            resolved = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._config.setdefault('ai', {})['device'] = resolved
            print(f"[配置] 自动检测设备: {resolved.upper()}")

    def get(self, key, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_required(self, key):
        value = self.get(key)
        if value is None:
            raise ValueError(f"缺少必需配置项: {key}")
        return value

    # ==================== MQTT 配置 ====================
    @property
    def mqtt_broker(self):
        return self.get_required('mqtt.broker_address')

    @property
    def mqtt_port(self):
        return self.get('mqtt.broker_port', 1883)

    @property
    def mqtt_keep_alive(self):
        return self.get('mqtt.keep_alive', 60)

    @property
    def mqtt_username(self):
        return self.get('mqtt.username')

    @property
    def mqtt_password(self):
        return self.get('mqtt.password')

    @property
    def mqtt_dev_id(self):
        return self.get_required('mqtt.dev_id')

    @property
    def mqtt_heartbeat_interval(self):
        return self.get('mqtt.heartbeat_interval', 10)

    @property
    def mqtt_heartbeat_topic(self):
        return self.get('mqtt.heartbeat_topic', '/ai/devstate')

    @property
    def mqtt_topic_prefix(self):
        return self.get('mqtt.topic_prefix', '/ai/sam3')

    # AutoLabel 服务主题
    @property
    def mqtt_subscribe_topic(self):
        prefix = self.mqtt_topic_prefix
        dev_id = self.mqtt_dev_id
        suffix = self.get('mqtt.topics.label_service.subscribe_suffix', '/AutoLabel')
        return f"{prefix}/{dev_id}{suffix}"

    def get_mqtt_response_topic(self, dev_id):
        prefix = self.mqtt_topic_prefix
        suffix = self.get('mqtt.topics.label_service.publish_suffix', '/AutoLabel_Res')
        return f"{prefix}/{dev_id}{suffix}"

    # GeoLabel 服务主题
    @property
    def geo_subscribe_topic(self):
        prefix = self.mqtt_topic_prefix
        dev_id = self.mqtt_dev_id
        suffix = self.get('mqtt.topics.geo_service.subscribe_suffix', '/GeoLabel')
        return f"{prefix}/{dev_id}{suffix}"

    def get_geo_response_topic(self, dev_id):
        prefix = self.mqtt_topic_prefix
        suffix = self.get('mqtt.topics.geo_service.publish_suffix', '/GeoLabel_Res')
        return f"{prefix}/{dev_id}{suffix}"

    # ChangeDetection 服务主题 (新增)
    @property
    def change_detection_subscribe_topic(self):
        prefix = self.mqtt_topic_prefix
        dev_id = self.mqtt_dev_id
        suffix = self.get('mqtt.topics.change_detection_service.subscribe_suffix', '/ChangeDetect')
        return f"{prefix}/{dev_id}{suffix}"

    def get_change_detection_response_topic(self, dev_id):
        prefix = self.mqtt_topic_prefix
        suffix = self.get('mqtt.topics.change_detection_service.publish_suffix', '/ChangeDetect_Res')
        return f"{prefix}/{dev_id}{suffix}"

    # ==================== AI 公共配置 ====================
    @property
    def ai_model_path(self):
        return self.get_required('ai.model_path')

    @property
    def ai_device(self):
        return self.get_required('ai.device')

    @property
    def ai_image_path(self):
        return self.get('ai.image_path', '')

    # AutoLabel 专用配置
    @property
    def ai_threshold(self):
        return self.get('ai.label_service.inference.threshold', 0.25)

    @property
    def ai_mask_threshold(self):
        return self.get('ai.label_service.inference.mask_threshold', 0.3)

    @property
    def ai_default_class_id(self):
        return self.get('ai.label_service.inference.default_class_id', 0)

    # GeoLabel 专用配置
    @property
    def geo_ai_threshold(self):
        return self.get('ai.geo_service.inference.threshold', 0.25)

    @property
    def geo_ai_mask_threshold(self):
        return self.get('ai.geo_service.inference.mask_threshold', 0.3)

    # ==================== ChangeDetection 服务配置 (新增) ====================
    @property
    def change_detection_pe_model_path(self):
        return self.get_required('ai.change_detection_service.pe_model_path')

    @property
    def change_detection_output_dir(self):
        return self.get('ai.change_detection_service.output_dir', './results_change_detection')

    @property
    def change_detection_pred_iou_thresh(self):
        return self.get('ai.change_detection_service.pred_iou_thresh', 0.3)

    @property
    def change_detection_sem_dist_thresh(self):
        return self.get('ai.change_detection_service.sem_dist_thresh', 0.20)

    @property
    def change_detection_overlap_filter_thresh(self):
        return self.get('ai.change_detection_service.overlap_filter_thresh', 0.7)

    @property
    def change_detection_concepts(self):
        return self.get('ai.change_detection_service.concepts', ["building", "road"])

    @property
    def change_detection_return_base64(self):
        return self.get('ai.change_detection_service.return_base64', False)

    # ==================== 日志配置 ====================
    @property
    def log_level(self):
        return self.get('logging.level', 'INFO')

    @property
    def show_payload(self):
        return self.get('logging.show_payload', False)


# ==================== 单例全局实例 ====================
_global_config = None


def get_config(config_path="config.yaml"):
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


# ==================== 测试代码 ====================
if __name__ == "__main__":
    config = get_config()
    print("配置加载成功！部分配置如下：")
    print(f"Broker: {config.mqtt_broker}:{config.mqtt_port}")
    print(f"设备ID: {config.mqtt_dev_id}")
    print(f"AutoLabel 订阅: {config.mqtt_subscribe_topic}")
    print(f"GeoLabel 订阅: {config.geo_subscribe_topic}")
    print(f"模型路径: {config.ai_model_path}")
    print(f"推理设备: {config.ai_device}")