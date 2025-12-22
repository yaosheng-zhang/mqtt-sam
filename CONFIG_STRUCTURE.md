# 配置文件结构说明

## 概览

项目现在支持两个独立的服务，每个服务有自己的配置，同时共享一些公共配置。

## 配置结构

### 1. MQTT 配置 (`mqtt`)

#### 公共部分
- `broker_address`: MQTT Broker 地址
- `broker_port`: MQTT Broker 端口
- `keep_alive`: 保持连接时间
- `username`, `password`: 认证信息
- `dev_id`: 设备ID
- `heartbeat_interval`: 心跳间隔（秒）
- `heartbeat_topic`: 心跳主题
- `topic_prefix`: 主题前缀（默认 `/ai/sam3`）

#### 服务主题配置 (`topics`)
```yaml
topics:
  label_service:        # AutoLabel 服务
    subscribe_suffix: "/AutoLabel"
    publish_suffix: "/AutoLabel_Res"

  geo_service:          # GeoLabel 服务
    subscribe_suffix: "/GeoLabel"
    publish_suffix: "/GeoLabel_Res"
```

**实际主题格式**:
- 订阅: `{topic_prefix}/{dev_id}{subscribe_suffix}`
- 发布: `{topic_prefix}/{dev_id}{publish_suffix}`

例如:
- AutoLabel 订阅: `/ai/sam3/sam3-device-001/AutoLabel`
- GeoLabel 订阅: `/ai/sam3/sam3-device-001/GeoLabel`

---

### 2. AI 配置 (`ai`)

#### 公共配置
所有服务共享:
- `model_path`: AI 模型路径
- `device`: 推理设备 (`cuda`, `cpu`, 或 `auto`)

#### AutoLabel 服务配置 (`label_service`)
```yaml
label_service:
  image_path: "https://djicloudapiyy.oss-cn-chengdu.aliyuncs.com/"
  inference:
    threshold: 0.25
    mask_threshold: 0.3
    default_class_id: 0
```

#### GeoLabel 服务配置 (`geo_service`)
```yaml
geo_service:
  inference:
    threshold: 0.25
    mask_threshold: 0.3
```

**注意**: GeoLabel 服务不需要 `image_path` 前缀，因为它直接接收完整的图片路径。

---

### 3. 日志配置 (`logging`)
- `level`: 日志级别 (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `show_payload`: 是否显示完整的消息内容

---

## Config Loader API

### MQTT 相关

#### AutoLabel 服务
```python
config.mqtt_subscribe_topic        # /ai/sam3/{dev_id}/AutoLabel
config.get_mqtt_response_topic(dev_id)  # /ai/sam3/{dev_id}/AutoLabel_Res
```

#### GeoLabel 服务
```python
config.geo_subscribe_topic         # /ai/sam3/{dev_id}/GeoLabel
config.get_geo_response_topic(dev_id)   # /ai/sam3/{dev_id}/GeoLabel_Res
```

#### 公共
```python
config.mqtt_broker
config.mqtt_port
config.mqtt_username
config.mqtt_password
config.mqtt_dev_id
config.mqtt_heartbeat_topic
config.mqtt_heartbeat_interval
```

---

### AI 相关

#### 公共配置
```python
config.ai_model_path    # 所有服务共享
config.ai_device        # 所有服务共享
```

#### AutoLabel 服务
```python
config.ai_image_path           # 图片URL前缀
config.ai_threshold            # 推理阈值
config.ai_mask_threshold       # 掩码阈值
config.ai_default_class_id     # 默认类别ID
```

#### GeoLabel 服务
```python
config.geo_ai_threshold        # 推理阈值
config.geo_ai_mask_threshold   # 掩码阈值
```

---

### 日志相关
```python
config.log_level
config.show_payload
```

---

## 使用示例

### 启动 AutoLabel 服务
```python
from config_loader import get_config
from ai_engine import SAM3InferenceEngine
from mqtt_service import MQTTServiceWrapper

config = get_config()
engine = SAM3InferenceEngine(
    model_path=config.ai_model_path,      # 公共
    device=config.ai_device,              # 公共
    image_path=config.ai_image_path,      # label_service 专属
    threshold=config.ai_threshold,        # label_service 专属
    mask_threshold=config.ai_mask_threshold  # label_service 专属
)
service = MQTTServiceWrapper(config, engine)
service.start()
```

### 启动 GeoLabel 服务
```python
from config_loader import get_config
from ai_engine import SAM3InferenceEngine
from geo_label_service import GeoLabelService

config = get_config()
engine = SAM3InferenceEngine(
    model_path=config.ai_model_path,         # 公共
    device=config.ai_device,                 # 公共
    image_path="",                           # geo_service 不需要前缀
    threshold=config.geo_ai_threshold,       # geo_service 专属
    mask_threshold=config.geo_ai_mask_threshold  # geo_service 专属
)
service = GeoLabelService(config, engine)
service.start()
```

### 同时启动两个服务（推荐）
```bash
python service_manager.py
```

服务管理器会自动读取配置并启动两个服务，每个服务使用自己对应的配置。

---

## 配置隔离优势

1. **清晰的职责分离**: 每个服务有自己的配置部分
2. **共享公共资源**: 模型路径和设备配置共享，避免重复
3. **独立调优**: 每个服务可以独立调整推理参数
4. **易于扩展**: 添加新服务只需在配置中新增对应部分
5. **进程隔离**: 通过 service_manager 实现进程级隔离，一个服务崩溃不影响另一个
