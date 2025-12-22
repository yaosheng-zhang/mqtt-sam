# handlers/change_detection_handler/change_detection_handler.py
"""
变化检测服务 Handler
"""
from ..base_handler import MessageHandler
from .change_detector import ChangeDetector
from .sam3_mask_generator import Sam3MaskGenerator
from datetime import datetime
import cv2
import os
import base64
import tempfile
import numpy as np


class ChangeDetectionHandler(MessageHandler):
    """变化检测服务处理器"""

    def __init__(self, ai_engine, config):
        super().__init__(ai_engine, config)

        # 初始化 SAM3 Mask 生成器
        self.sam3_generator = Sam3MaskGenerator(
            sam3_model=ai_engine.sam3_model,
            sam3_processor=ai_engine.sam3_processor,
            device=ai_engine.device,
            pred_iou_thresh=config.change_detection_pred_iou_thresh,
            concepts=config.change_detection_concepts
        )

        # 初始化变化检测器
        self.change_detector = ChangeDetector(
            pe_model=ai_engine.pe_model,
            pe_preprocess=ai_engine.pe_preprocess,
            sam3_generator=self.sam3_generator,
            device=ai_engine.device
        )

        # 输出目录
        self.output_dir = config.change_detection_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[ChangeDetection] 初始化完成")
        print(f"   检测概念: {config.change_detection_concepts}")
        print(f"   输出目录: {self.output_dir}")

    @property
    def subscribe_topic(self):
        return self.config.change_detection_subscribe_topic

    def get_name(self):
        return "变化检测服务"

    def on_message(self, dev_id, data, publish):
        """
        处理变化检测请求

        消息格式:
        {
            "image_a": "path/to/image_a.jpg",
            "image_b": "path/to/image_b.jpg",
            "output_name": "optional_output_name.jpg"  # 可选
        }
        """
        try:
            # 1. 解析输入
            image_a_path = data.get("image_a")
            image_b_path = data.get("image_b")
            output_name = data.get("output_name")

            if not image_a_path or not image_b_path:
                error_msg = "缺少必需参数: image_a 和 image_b"
                print(f"[ChangeDetection] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return

            print(f"[ChangeDetection] 开始处理变化检测:")
            print(f"   图片A: {image_a_path}")
            print(f"   图片B: {image_b_path}")

            # 2. 加载图片
            base_path = self.config.ai_image_path or ""
            full_path_a = os.path.join(base_path, image_a_path) if not image_a_path.startswith(('http://', 'https://')) else image_a_path
            full_path_b = os.path.join(base_path, image_b_path) if not image_b_path.startswith(('http://', 'https://')) else image_b_path
            print(full_path_a, full_path_b)
            img1 = self._load_image(full_path_a)
            img2 = self._load_image(full_path_b)
        
            if img1 is None or img2 is None:
                error_msg = "图片加载失败"
                print(f"[ChangeDetection] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return

            # 3. 执行变化检测
            result = self.change_detector.detect_changes(
                img1,
                img2,
                sem_dist_thresh=self.config.change_detection_sem_dist_thresh,
            )

            # 4. 生成可视化结果
            if output_name is None:
                # 生成默认文件名
                basename_a = os.path.splitext(os.path.basename(image_a_path))[0]
                output_name = f"{basename_a}_change_detection.jpg"

            output_path = os.path.join(self.output_dir, output_name)
            self.change_detector.visualize(img1, img2, result, output_path)

            print(f"[ChangeDetection] 检测完成:")
            print(f"   匹配对象: {len(result['matches'])}")
            print(f"   消失对象: {len(result['unmatched_1'])}")
            print(f"   新增对象: {len(result['unmatched_2'])}")
            print(f"   总变化数: {result['changed_count']}")
            print(f"   结果保存: {output_path}")

            # 5. 构造响应 (可选: 返回 base64 编码的图片)
            response_data = {
                "success": True,
                "image_a": image_a_path,
                "image_b": image_b_path,
                "output_path": output_path,
                "statistics": {
                    "matched": len(result['matches']),
                    "removed": len(result['unmatched_1']),
                    "new": len(result['unmatched_2']),
                    "changed": result['changed_count'],
                    "total_objects_a": result['total_objects_1'],
                    "total_objects_b": result['total_objects_2']
                },
                "timestamp": datetime.now().isoformat()
            }

            # 可选: 如果需要返回图片 base64
            if self.config.change_detection_return_base64:
                # 读取并压缩图片以减小大小
                img_result = cv2.imread(output_path)
                # 调整分辨率 (最大宽度 1920)
                h, w = img_result.shape[:2]
                if w > 1920:
                    scale = 1920 / w
                    new_w, new_h = 1920, int(h * scale)
                    img_result = cv2.resize(img_result, (new_w, new_h))

                # JPEG 压缩 (质量 85)
                _, buffer = cv2.imencode('.jpg', img_result, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                # 检查大小 (限制 1MB)
                size_mb = len(img_base64) / 1024 / 1024
                if size_mb > 1.0:
                    print(f"[警告] Base64 图片过大 ({size_mb:.2f}MB)，已跳过")
                else:
                    response_data["result_image_base64"] = img_base64
                    print(f"[ChangeDetection] 返回 base64 图片 ({size_mb:.2f}MB)")

            # 6. 发布结果
            publish(
                topic=self.config.get_change_detection_response_topic(dev_id),
                data=response_data
            )

        except Exception as e:
            error_msg = f"变化检测失败: {str(e)}"
            print(f"[ChangeDetection] {error_msg}")
            import traceback
            traceback.print_exc()
            self._publish_error(dev_id, error_msg, publish)

    def _load_image(self, path):
        """加载图片 (支持本地和远程)"""
        try:
            if path.startswith(('http://', 'https://')):
                # 远程图片
                import requests
                from io import BytesIO
                resp = requests.get(path, timeout=30)
                resp.raise_for_status()
                img_array = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
            else:
                # 本地图片
                if not os.path.exists(path):
                    print(f"[ChangeDetection] 文件不存在: {path}")
                    return None
                img = cv2.imread(path)
                return img
        except Exception as e:
            print(f"[ChangeDetection] 图片加载失败: {path} - {e}")
            return None

    def _publish_error(self, dev_id, error_msg, publish):
        """发布错误消息"""
        publish(
            topic=self.config.get_change_detection_response_topic(dev_id),
            data={
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        )
