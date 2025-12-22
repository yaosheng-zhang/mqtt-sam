import os
import sys
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from transformers import Sam3Processor, Sam3Model
# 引用同目录下的 geo_utils
from handlers.geo_handler.geo_utils import GeoCalculator

# 尝试导入 PE 模型相关模块 (可选,用于变化检测)
try:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
    PE_AVAILABLE = True
except ImportError:
    PE_AVAILABLE = False
    print("[警告] PE 模块未找到,变化检测服务将不可用")
    print("       如需使用变化检测,请将 core 目录添加到项目根目录")

class SAM3InferenceEngine:
    def __init__(self, model_path, device, image_path, threshold=0.25, mask_threshold=0.3, enable_pe=False, pe_model_path=None):
        print(f"[AI核心] 正在加载 SAM3 模型 ({device})...", flush=True)

        # --- SAM3 模型加载 ---
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件夹: {os.path.abspath(model_path)}")

        try:
            # 加载 SAM3 模型
            self.sam3_model = Sam3Model.from_pretrained(model_path).to(device)
            self.sam3_processor = Sam3Processor.from_pretrained(model_path)
            self.device = device
            self.image_path = image_path
            self.threshold = threshold
            self.mask_threshold = mask_threshold
            print("[AI核心] SAM3 模型加载完成!", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[严重错误] SAM3 模型加载失败: {e}", flush=True)
            raise e

        # --- PE 模型加载 (可选,用于变化检测) ---
        self.pe_model = None
        self.pe_preprocess = None

        if enable_pe:
            if not PE_AVAILABLE:
                print("[警告] PE 模块不可用,跳过 PE 模型加载")
            elif pe_model_path is None:
                print("[警告] 未指定 PE 模型路径,跳过 PE 模型加载")
            elif not os.path.exists(pe_model_path):
                print(f"[警告] PE 模型文件不存在: {pe_model_path}")
            else:
                try:
                    print(f"[AI核心] 正在加载 PE 模型...", flush=True)
                    self.pe_model = pe.VisionTransformer.from_config(
                        "PE-Core-L14-336",
                        pretrained=False,
                        checkpoint_path=pe_model_path
                    ).to(device).eval()
                    self.pe_preprocess = transforms.get_image_transform(224)
                    print("[AI核心] PE 模型加载完成!", flush=True)
                except Exception as e:
                    print(f"[警告] PE 模型加载失败: {e}")
                    import traceback
                    traceback.print_exc()

    # 保持原有的 model/processor 属性以兼容旧代码
    @property
    def model(self):
        return self.sam3_model

    @property
    def processor(self):
        return self.sam3_processor 

    def analyze_image(self, relative_path, prompt, 
                  threshold=None, mask_threshold=None,
                  return_masks=True):
        """
        万能、干净、可控的推理接口
        
        参数:
            return_masks (bool): 是否返回掩码（GeoLabel需要，AutoLabel不需要）
        """
        threshold = threshold or self.threshold
        mask_threshold = mask_threshold or self.mask_threshold
    
        image_path = os.path.join(self.image_path, relative_path)
    
        # === 1. 加载图片 ===
        try:
            if image_path.startswith(('http://', 'https://')):
                resp = requests.get(image_path, timeout=30)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                if not os.path.exists(image_path):
                    return {"success": False, "error": "file not found"}
                img = Image.open(image_path).convert("RGB")
            w, h = img.size
        except Exception as e:
            return {"success": False, "error": str(e)}
        # === 2. 推理 ===
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
    
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(h, w)]
        )[0]
 
        boxes = results["boxes"].cpu().numpy()      # (N, 4)
        scores = results["scores"].cpu().numpy()    # (N,)
    
        # === 3. 按需返回 masks ===
        masks = None
        if return_masks and "masks" in results:
            masks = (results["masks"].cpu().numpy() > 0.5).astype(bool)  # List of (H,W) bool
    
        return {
            "success": True,
            "image_width": w,
            "image_height": h,
            "boxes": boxes.tolist(),           # List[List[float]]
            "scores": scores.tolist(),        # List[float]
            "masks": masks.tolist() if masks is not None else None,  # 可选返回
        }
        
       

    def analyze_images_batch(self, image_paths, prompt, label_id=0, callback=None):
        """
        批量处理图片，每处理完一张就通过回调返回结果

        Args:
            image_paths: 图片路径或URL列表
            prompt: 提示词
            label_id: YOLO格式中的类别ID
            callback: 回调函数，参数为 (index, image_path, yolo_labels)
                     index: 当前处理的图片索引
                     image_path: 图片路径
                     yolo_labels: YOLO格式标签列表

        Returns:
            list: 所有图片的处理结果列表
        """
        results = []
        total = len(image_paths)

        print(f"[AI核心] 开始批量处理 {total} 张图片")

        for idx, image_path in enumerate(image_paths):
            print(f"[AI核心] 处理进度: {idx + 1}/{total} - {image_path}")

            # 处理单张图片
            yolo_labels = self.analyze_image(image_path, prompt, label_id)

            result = {
                "imgpath": image_path,
                "labels": yolo_labels
            }
            results.append(result)

            # 如果有回调函数，立即调用
            if callback:
                callback(idx, image_path, yolo_labels, result)

        print(f"[AI核心] 批量处理完成，共处理 {total} 张图片")
        return results