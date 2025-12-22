# handlers/change_detection_handler/sam3_mask_generator.py
"""
SAM3 Mask 生成器 - 用于变化检测
"""
import numpy as np
import cv2
import torch
from PIL import Image


class Sam3MaskGenerator:
    """SAM3 实例分割 Mask 生成器"""

    def __init__(self, sam3_model, sam3_processor, device, pred_iou_thresh=0.3, concepts=None):
        """
        Args:
            sam3_model: SAM3 模型实例
            sam3_processor: SAM3 处理器
            device: 计算设备
            pred_iou_thresh: 预测置信度阈值
            concepts: 检测概念列表 (如 ["building", "road"])
        """
        self.sam3_model = sam3_model
        self.sam3_processor = sam3_processor
        self.device = device
        self.pred_iou_thresh = pred_iou_thresh
        self.concepts = concepts or ["building", "road"]

    @torch.no_grad()
    def generate(self, image_cv: np.ndarray):
        """
        生成图像中的所有对象 Mask

        Args:
            image_cv: OpenCV BGR 格式图像

        Returns:
            np.ndarray: shape (N, H, W), bool 类型的 mask 数组
        """
        pil_img = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        H, W = image_cv.shape[:2]
        img_area = H * W

        all_masks = []
        all_scores = []

        # 对每个概念进行检测
        for concept in self.concepts:
            try:
                inputs = self.sam3_processor(
                    images=pil_img,
                    text=concept,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.sam3_model(**inputs)

                results = self.sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.pred_iou_thresh,
                    mask_threshold=self.pred_iou_thresh,
                    target_sizes=[(H, W)]
                )[0]

                masks = results["masks"].cpu().numpy()
                scores = results["scores"].cpu().numpy()

                if len(masks) > 0:
                    valid_indices = []
                    for k in range(len(masks)):
                        m = masks[k]
                        y_idxs, x_idxs = np.where(m)
                        if len(y_idxs) == 0:
                            continue

                        mask_area = len(y_idxs)

                        # 1. 面积过滤 (过小或过大)
                        if mask_area < 80 or (mask_area / img_area) > 0.75:
                            continue

                        # 2. 边框过滤 (几乎覆盖整张图的去掉)
                        h_box = y_idxs.max() - y_idxs.min()
                        w_box = x_idxs.max() - x_idxs.min()
                        if h_box > H * 0.98 and w_box > W * 0.98:
                            continue

                        if scores[k] > self.pred_iou_thresh:
                            valid_indices.append(k)

                    if valid_indices:
                        all_masks.extend(masks[valid_indices])
                        all_scores.extend(scores[valid_indices])

            except Exception as e:
                print(f"[MaskGenerator] 检测 '{concept}' 时出错: {e}")
                continue

        if len(all_masks) == 0:
            return np.array([], dtype=bool).reshape(0, H, W)

        masks_np = np.stack(all_masks)
        scores_np = np.array(all_scores)

        # NMS 去重
        return self._nms(masks_np, scores_np)

    def _nms(self, masks, scores):
        """非极大值抑制 (NMS) - 去除重叠的 mask"""
        order = np.argsort(-scores)
        keep = []
        areas = np.sum(masks, axis=(1, 2))

        for i in order:
            curr_mask = masks[i]
            is_redundant = False

            for k in keep:
                kept_mask = masks[k]
                intersection = np.sum(curr_mask & kept_mask)
                if intersection == 0:
                    continue

                union = areas[i] + areas[k] - intersection
                iou = intersection / (union + 1e-6)
                ioa_curr = intersection / (areas[i] + 1e-6)
                ioa_kept = intersection / (areas[k] + 1e-6)

                # 去重条件
                if iou > 0.6 or ioa_curr > 0.85 or ioa_kept > 0.90:
                    is_redundant = True
                    break

            if not is_redundant:
                keep.append(i)

        return masks[keep].astype(bool)
