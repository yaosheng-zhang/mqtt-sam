# SAM3 多功能AI边缘推理节点

基于 MQTT 的分布式 AI 推理服务,支持自动标注和地理定位两大核心功能,采用 SAM3 (Segment Anything Model 3) 进行高精度图像分割。

## 项目概述

本项目是一个轻量级的边缘 AI 推理节点,通过 MQTT 协议接收任务请求,使用 SAM3 模型进行实时图像分析,并返回结构化结果。特别适用于无人机影像处理、遥感标注等场景。

### 核心特性

- **双服务架构**: 自动标注(AutoLabel) + 地理定位(GeoLabel)
- **异步任务队列**: 多线程并发处理,支持高并发请求
- **智能负载管理**: 实时心跳监控,自动上报设备状态
- **灵活配置**: YAML 配置文件,支持本地/远程图片处理
- **地理精准定位**: 基于 EXIF 元数据和针孔相机模型的 GPS 坐标计算

## 系统架构

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────┐
│  MQTT Broker│◄─────►│  MQTT-SAM3 Node  │◄─────►│ SAM3 Model  │
│  (消息中心) │       │   (推理节点)      │       │  (AI引擎)   │
└─────────────┘       └──────────────────┘       └─────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼────────┐  ┌───────▼────────┐
            │  AutoLabel     │  │   GeoLabel     │
            │  自动标注服务   │  │   地理标注服务  │
            └────────────────┘  └────────────────┘
```

## 目录结构

```
mqtt-sam3-/
├── config.yaml                 # 主配置文件
├── main.py                     # 程序入口
├── mqtt.py                     # MQTT 客户端实现
├── ai_engine.py                # SAM3 推理引擎
├── task_queue.py               # 异步任务队列
├── config_loader.py            # 配置加载器
├── system_monitor.py           # 系统监控模块
├── handlers/
│   ├── base_handler.py         # 抽象处理器基类
│   ├── label_handler/
│   │   └── auto_label_handler.py   # 自动标注处理器
│   └── geo_handler/
│       ├── geo_label_handler.py    # 地理标注处理器
│       └── geo_utils.py            # 地理坐标计算工具
└── README.md                   # 本文档
```

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.x+ (GPU加速,可选)
- ExifTool (地理定位功能必需)

### 安装依赖

```bash
pip install torch transformers paho-mqtt pillow pyyaml requests psutil GPUtil
```

安装 ExifTool (Windows):
```bash
# 下载后添加到系统 PATH
https://exiftool.org/
```

### 配置文件

编辑 [config.yaml](config.yaml) 文件:

```yaml
mqtt:
  broker_address: "your.mqtt.broker"
  broker_port: 1883
  dev_id: "sam3-device-001"

ai:
  model_path: "path/to/sam3/model"
  device: "auto"  # cuda/cpu/auto
  image_path: "https://your-image-server/"
```

### 启动服务

```bash
python main.py
```

启动成功后会显示:

```
======================================================================
SAM3 多功能AI边缘推理节点 启动中
======================================================================
[AI核心] 正在加载 SAM3 模型 (cuda)...
[AI核心] 模型加载完成!
设备ID: sam3-device-001
自动标注 → /ai/sam3/sam3-device-001/AutoLabel
地理标注 → /ai/sam3/sam3-device-001/GeoLabel
推理设备: CUDA
======================================================================
```

## 功能详解

### 1. 自动标注服务 (AutoLabel)

将 AI 检测结果转换为 YOLO 格式标注文件。

**订阅主题**: `/ai/sam3/{dev_id}/AutoLabel`

**请求格式**:
```json
{
  "prompt": "roof",
  "files": [
    "relative/path/to/image1.jpg",
    "relative/path/to/image2.jpg"
  ],
  "label": 0
}
```

**响应主题**: `/ai/sam3/{dev_id}/AutoLabel_Res`

**响应格式**:
```json
{
  "prompt": "roof",
  "res": [{
    "imgpath": "relative/path/to/image1.jpg",
    "labels": [
      "0 0.512345 0.345678 0.123456 0.234567"
    ]
  }],
  "label": 0,
  "progress": {
    "current": 1,
    "total": 2
  },
  "sequence_id": 1,
  "timestamp": "2025-12-05T10:30:45.123456"
}
```

**YOLO 标签格式**: `class_id center_x center_y width height` (归一化坐标)

**核心逻辑** ([auto_label_handler.py:72-106](handlers/label_handler/auto_label_handler.py#L72-L106)):
- 边界框转中心点坐标
- 归一化到 [0, 1] 范围
- Clamping 防止越界
- 保留 6 位小数精度

### 2. 地理标注服务 (GeoLabel)

基于 EXIF 元数据计算检测目标的精确 GPS 坐标。

**订阅主题**: `/ai/sam3/{dev_id}/GeoLabel`

**请求格式**:
```json
{
  "prompt": "solar panel",
  "files": [
    "drone_images/DJI_0123.JPG"
  ]
}
```

**响应主题**: `/ai/sam3/{dev_id}/GeoLabel_Res`

**响应格式**:
```json
{
  "image_path": "drone_images/DJI_0123.JPG",
  "prompt": "solar panel",
  "objects": [
    {
      "id": 1,
      "confidence": 0.95,
      "bbox": [123.5, 456.7, 234.5, 567.8],
      "pixel_center": [179.0, 512.2],
      "geolocation": {
        "latitude": 30.12345678,
        "longitude": 104.87654321,
        "altitude_m": 75.3,
        "is_mask_precise": true
      }
    }
  ],
  "total_objects": 1,
  "is_geo_valid": true,
  "timestamp": "2025-12-05T10:31:20.456789"
}
```

**地理计算原理** ([geo_utils.py:160-178](handlers/geo_handler/geo_utils.py#L160-L178)):

1. **提取 EXIF 元数据**:
   - GPS 坐标 (GPSLatitude, GPSLongitude)
   - 相对高度 (RelativeAltitude)
   - 云台偏航角 (GimbalYawDegree)
   - 焦距 (FocalLength)
   - 图像尺寸 (ImageWidth, ImageHeight)

2. **针孔相机模型**:
   ```
   GSD = (高度 × 传感器宽度) / (焦距 × 图像宽度)
   ```

3. **像素到GPS转换**:
   - 计算像素偏移量
   - 应用偏航角旋转矩阵
   - WGS84 椭球坐标变换

4. **支持远程图片**: 自动下载 HTTP/HTTPS 图片并清理临时文件

### 3. 设备心跳监控

**发布主题**: `/ai/devstate/{dev_id}`

**心跳数据**:
```json
{
  "cmd": "devstate",
  "gateway": "sam3-device-001",
  "data": {
    "cpuUsage": 45.2,
    "memoryUsage": 67.8,
    "gpuUsage": 89.3,
    "devicestatus": 1
  },
  "timestamp": 1733456789123
}
```

**设备状态码**:
- `1`: 空闲 (Idle)
- `2`: 忙碌 (Busy)

## 核心组件说明

### AI 推理引擎 ([ai_engine.py](ai_engine.py))

**SAM3InferenceEngine 类**:
- 支持自动设备选择 (CUDA/CPU)
- 远程图片自动下载
- 可选 mask 返回 (节省内存)
- 实例分割后处理

**关键方法**:
```python
result = engine.analyze_image(
    relative_path="path/to/image.jpg",
    prompt="roof",
    threshold=0.25,
    mask_threshold=0.3,
    return_masks=False  # AutoLabel不需要mask
)
```

### 任务队列 ([task_queue.py](task_queue.py))

**设计特点**:
- 单例模式,全局共享
- 4 线程并发推理 (避免 GPU 过载)
- 队列容量 200,防止内存溢出
- 异常隔离,单个任务失败不影响全局

**性能优化** ([mqtt.py:74-96](mqtt.py#L74-L96)):
- `on_message` < 5ms 立即返回
- 解析和推理完全异步
- 忙碌状态自动管理

### MQTT 客户端 ([mqtt.py](mqtt.py))

**特性**:
- MQTTv5 协议
- 动态主题注册
- 消息 QoS 1 可靠传输
- NumPy 类型自动转换

**消息处理流程**:
```
接收消息 → JSON解析 → 提取dev_id → 匹配Handler → 入队 → 异步推理 → 发布结果
```

## 配置参数详解

| 参数路径 | 说明 | 默认值 |
|---------|------|--------|
| `mqtt.broker_address` | MQTT Broker 地址 | - |
| `mqtt.broker_port` | Broker 端口 | 1883 |
| `mqtt.dev_id` | 设备唯一标识 | sam3-device-001 |
| `mqtt.heartbeat_interval` | 心跳间隔(秒) | 10 |
| `ai.model_path` | SAM3 模型路径 | - |
| `ai.device` | 推理设备 | auto |
| `ai.image_path` | 图片基础路径/URL | - |
| `ai.label_service.inference.threshold` | 检测置信度阈值 | 0.25 |
| `ai.label_service.inference.mask_threshold` | 掩码二值化阈值 | 0.3 |

完整配置结构参考 [CONFIG_STRUCTURE.md](CONFIG_STRUCTURE.md)

## 使用场景

### 场景1: 无人机屋顶检测

```python
# 客户端发送
{
  "prompt": "roof",
  "files": ["drone/area_01/DJI_0456.JPG"],
  "label": 0
}

# 获得 YOLO 标注 + GPS 坐标
# 可直接用于训练或 GIS 系统
```

### 场景2: 批量影像标注

```python
# 一次性提交 100 张图片
{
  "prompt": "solar panel",
  "files": ["img001.jpg", "img002.jpg", ..., "img100.jpg"]
}

# 实时接收进度回调
# progress: {current: 23, total: 100}
```

### 场景3: 多节点分布式推理

```yaml
# 节点1: 专注屋顶检测
dev_id: "sam3-roof-001"

# 节点2: 专注车辆检测
dev_id: "sam3-vehicle-002"

# 统一调度,负载均衡
```

## 性能优化建议

1. **GPU 显存管理**:
   - 单张图片推理: ~2-4GB 显存
   - 建议 RTX 3060 (12GB) 以上

2. **并发线程数调整**:
   ```python
   # task_queue.py:19
   for i in range(4):  # 根据 GPU 性能调整
   ```

3. **网络图片缓存**:
   - 高频访问图片建议本地缓存
   - 减少重复下载开销

4. **日志等级**:
   ```yaml
   logging:
     level: "WARNING"  # 生产环境降低日志
     show_payload: false
   ```

## 常见问题

### Q1: 提示 "找不到模型文件夹"
**A**: 检查 [config.yaml](config.yaml) 中 `ai.model_path` 路径是否正确,确保包含 `config.json` 和 `pytorch_model.bin`。

### Q2: 地理定位返回 `null`
**A**:
1. 确认图片包含 GPS EXIF 信息 (使用 `exiftool image.jpg` 验证)
2. 检查 ExifTool 是否正确安装
3. 查看日志中的 `is_geo_valid` 字段

### Q3: GPU 利用率低
**A**:
- 增加 `task_queue.py` 中的线程数
- 检查是否有 CPU 瓶颈 (数据加载/预处理)

### Q4: MQTT 连接失败
**A**:
- 验证 Broker 地址和端口
- 检查防火墙设置
- 确认认证信息 (username/password)

## 开发扩展

### 添加新的处理服务

1. 继承 `MessageHandler` 基类:
```python
# handlers/custom_handler.py
from handlers.base_handler import MessageHandler

class CustomHandler(MessageHandler):
    @property
    def subscribe_topic(self):
        return "/ai/sam3/+/CustomService"

    def get_name(self):
        return "自定义服务"

    def on_message(self, dev_id, data, publish):
        # 实现你的逻辑
        result = self.ai_engine.analyze_image(...)
        publish(f"/ai/sam3/{dev_id}/Custom_Res", result)
```

2. 在 [main.py](main.py) 注册:
```python
client.register_handler(CustomHandler(engine, config))
```

### 自定义坐标转换

修改 [geo_utils.py:160](handlers/geo_handler/geo_utils.py#L160) 中的 `pixel_to_gps` 方法,实现自定义投影算法。

## 技术栈

- **深度学习**: PyTorch, Transformers (HuggingFace)
- **通信协议**: MQTT (Paho)
- **图像处理**: Pillow, NumPy
- **元数据解析**: ExifTool
- **配置管理**: PyYAML

## 许可证

本项目代码仅供学习参考,使用的 SAM3 模型需遵循其原始许可协议。

## 更新日志

- **2025-12**:
  - 优化地理计算,支持云端图片
  - 增加心跳监控和设备状态上报
  - 实现 NumPy 类型自动序列化

## 联系方式

如有问题或建议,请查看代码中的注释或修改配置文件进行实验。

---

**生成时间**: 2025-12-05
**代码版本**: 基于当前工作目录分析生成
