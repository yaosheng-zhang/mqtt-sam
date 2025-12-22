# handlers/auto_label_handler.py
from ..base_handler import MessageHandler
from datetime import datetime

class AutoLabelHandler(MessageHandler):
    @property
    def subscribe_topic(self): 
        return self.config.mqtt_subscribe_topic
    
    def get_name(self): 
        return "自动标注服务"

    def on_message(self, dev_id, data, publish):
        # 1. 获取输入参数
        prompt = data.get("prompt", "")
        files = data.get("files", [])
        # label_flag: 原始的标签标识（可能是int也可能是str），用于回传
        label_flag = data.get("label", self.config.ai_default_class_id)
        # label_id: 用于写入txt内容的整数ID
        try:
            label_id = int(label_flag)
        except ValueError:
            label_id = 0

        if not files:
            print("[AutoLabel] 收到空文件列表")
            return

        total_files = len(files)
        print(f"[AutoLabel] 开始处理任务: {len(files)} 张图片, 提示词: {prompt}")

        # 2. 遍历处理
        for index, path in enumerate(files, 1):
            yolo_labels = []
            
            # 调用 AI 引擎
            result = self.ai_engine.analyze_image(
                relative_path=path,
                prompt=prompt,
                threshold=self.config.ai_threshold,
                mask_threshold=self.config.ai_mask_threshold,
                return_masks=False  # 不需要掩码，提高速度
            )

            # 如果推理成功，进行格式转换
            if result.get("success", False):
                yolo_labels = self.to_yolo_format(
                    boxes=result["boxes"],
                    img_w=result["image_width"],
                    img_h=result["image_height"],
                    label_id=label_id
                )
            else:
                print(f"[AutoLabel] 图片处理失败: {path} - {result.get('error')}")
                # 即使失败也建议发送空标签，保证前端进度条能走完
                yolo_labels = []

            # 3. 构造符合要求的响应数据格式
            response_data = {
                "prompt": prompt,
                "res": [{"imgpath": path, "labels": yolo_labels}],
                "label": label_flag,
                "progress": {"current": index, "total": total_files},
                "sequence_id": index,               # 当前序号
                "timestamp": datetime.now().isoformat()
            }

            # 4. 发布消息
            publish(self.config.get_mqtt_response_topic(dev_id), response_data)

    @staticmethod
    def to_yolo_format(boxes, img_w, img_h, label_id):
        """
        将 [x0, y0, x1, y1] 转换为 YOLO 格式 [class_id cx cy w h]
        并包含边界限制逻辑
        """
        if not boxes:
            return []
            
        labels = []
        for box in boxes:
            x0, y0, x1, y1 = box
            
            # 计算中心点和宽高
            box_w = x1 - x0
            box_h = y1 - y0
            center_x = x0 + box_w / 2
            center_y = y0 + box_h / 2
            
            # 归一化
            norm_cx = center_x / img_w
            norm_cy = center_y / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h
            
            # --- 数值截断 (Clamping) ---
            # 确保坐标严格在 [0, 1] 范围内，防止 AI 输出越界导致训练报错
            norm_cx = max(0.0, min(1.0, norm_cx))
            norm_cy = max(0.0, min(1.0, norm_cy))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            # 格式化: 类别 cx cy w h (保留6位小数)
            labels.append(f"{label_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        return labels