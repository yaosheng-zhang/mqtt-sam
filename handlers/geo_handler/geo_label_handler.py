# handlers/geo_label_handler.py
from ..base_handler import MessageHandler
from .geo_utils import GeoCalculator
from datetime import datetime
import numpy as np

class GeoLabelHandler(MessageHandler):
    @property
    def subscribe_topic(self):
        return self.config.geo_subscribe_topic

    def get_name(self):
        return "地理标注服务"

    def on_message(self, dev_id, data, publish):
        prompt = data.get("prompt", "")
        files = data.get("files", [])
        if not files:
            return

        # 确保基础路径不为 None
        base_path = self.config.ai_image_path or ""

        for i, path in enumerate(files, 1):
            # 1. AI 推理
            result = self.ai_engine.analyze_image(
                relative_path=path,
                prompt=prompt,
                threshold=self.config.geo_ai_threshold,
                mask_threshold=self.config.geo_ai_mask_threshold,
                return_masks=True 
            )

            if not result.get("success", False):
                print(f"[GeoLabel] 推理失败: {path}")
                continue

            # 2. 构造地理计算器 (传入基础路径和相对路径)
            geo = GeoCalculator(base_path, path)
            objects = []

            # 3. 遍历结果
            masks = result.get("masks")
            if masks is None:
                masks = [None] * len(result["boxes"])

            for idx, (box, score, mask) in enumerate(zip(result["boxes"], result["scores"], masks), 1):
                x0, y0, x1, y1 = box

                # --- 坐标选点策略优化 ---
                if mask is not None and np.any(mask):
                    ys, xs = np.where(mask)
                    # 策略A (俯视): 取质心/重心 (推荐用于房顶、车辆)
                    foot_x = np.mean(xs)
                    foot_y = np.mean(ys)
                    
                    # 策略B (侧视): 取底部中心 (如果你拍的是站立的人，用这行)
                    # foot_y = ys.max()
                    # foot_x = xs[ys == foot_y].mean()
                else:
                    # 退化为 BBox 中心
                    foot_x = (x0 + x1) / 2
                    foot_y = (y0 + y1) / 2

                # 4. 计算 GPS
                lat, lon = geo.pixel_to_gps(foot_x, foot_y)

                # 5. 构造返回对象
                obj_data = {
                    "id": idx,
                    "confidence": round(float(score), 3),
                    "bbox": [round(float(x), 1) for x in box],
                    "pixel_center": [round(foot_x, 1), round(foot_y, 1)], # 方便调试看有没有偏
                    "geolocation": None  # 默认空
                }

                # 只有当计算成功时才填入数据
                if lat is not None and lon is not None:
                    obj_data["geolocation"] = {
                        "latitude": round(lat, 8),
                        "longitude": round(lon, 8),
                        "altitude_m": round(geo.meta.get("alt", 0), 1),
                        "is_mask_precise": (mask is not None)
                    }

                objects.append(obj_data)

            # 6. 发布结果
            publish(
                topic=self.config.get_geo_response_topic(dev_id),
                data={
                    "image_path": path,
                    "prompt": prompt,
                    "objects": objects,
                    "total_objects": len(objects),
                    "is_geo_valid": geo.valid, # 告诉前端这张图有没有坐标
                    "timestamp": datetime.now().isoformat()
                }
            )