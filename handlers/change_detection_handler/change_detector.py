# handlers/change_detection_handler/change_detector.py
"""
变化检测核心工具类
基于 PE (Vision Encoder) 和 SAM3 的图像变化检测
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment


class ChangeDetector:
    """变化检测器 - 负责双时相图像的变化分析"""

    def __init__(self, pe_model, pe_preprocess, sam3_generator, device):
        """
        Args:
            pe_model: PE Vision Encoder 模型
            pe_preprocess: PE 预处理函数
            sam3_generator: SAM3 Mask 生成器
            device: 计算设备 (cuda/cpu)
        """
        self.pe_model = pe_model
        self.pe_preprocess = pe_preprocess
        self.sam3_generator = sam3_generator
        self.device = device

        # 调试：输出设备信息
        device_str = str(device)
        print(f"[ChangeDetector] 初始化完成, 设备: {device_str} (类型: {type(device).__name__})")

    def detect_changes(self, img1_cv, img2_cv, sem_dist_thresh=0.20):
        """
        检测两张图片的变化

        Args:
            img1_cv: 第一张图片 (OpenCV BGR 格式)
            img2_cv: 第二张图片 (OpenCV BGR 格式)
            sem_dist_thresh: 语义距离阈值

        Returns:
            dict: {
                'matches': [(i, j), ...],
                'unmatched_1': [i, ...],
                'unmatched_2': [j, ...],
                'change_info': {(i, j): {'changed': bool, 'sem': float, 'col': float}},
                'masks_1': np.ndarray,
                'masks_2': np.ndarray,
                'changed_count': int
            }
        """
        print(f"[变化检测调试] sem_dist_thresh = {sem_dist_thresh}")
        H, W = img1_cv.shape[:2]

        # 1. 生成 Mask
        print(f"[变化检测调试] 开始生成 Mask...")
        masks1 = self.sam3_generator.generate(img1_cv)
        masks2 = self.sam3_generator.generate(img2_cv)
        print(f"[变化检测调试] Mask 生成完成: img1={len(masks1)} masks, img2={len(masks2)} masks")

        # 2. 提取特征
        print(f"[变化检测调试] 开始提取特征...")
        vecs1 = self._batch_extract_features(img1_cv, masks1)
        vecs2 = self._batch_extract_features(img2_cv, masks2)
        print(f"[变化检测调试] 特征提取完成: vecs1={vecs1.shape}, vecs2={vecs2.shape}")

        # 3. 匹配
        print(f"[变化检测调试] 开始对象匹配...")
        matches, matched_1, matched_2 = self._match_objects(
            masks1, masks2, vecs1, vecs2, img1_cv.shape
        )
        print(f"[变化检测调试] 匹配完成: {len(matches)} 对匹配")

        # 4. 分析变化
        change_info = {}
        for (i, j) in matches:
            # 1. 语义特征距离
            sem_dist = 1.0 - torch.dot(vecs1[i], vecs2[j]).item()

            # 2. 颜色特征变化 (HSV 鲁棒分析)
            is_color_changed, col_diff_val = self._check_color_change_hsv(
                img1_cv, masks1[i], img2_cv, masks2[j]
            )

            # 3. 综合判定: 语义剧变 OR 颜色明显变化
            is_changed = (sem_dist > sem_dist_thresh) or is_color_changed

            change_info[(i, j)] = {
                'changed': is_changed,
                'sem': sem_dist,
                'col': col_diff_val
            }

            # 调试输出
            if is_changed:
                reason = "语义变化" if sem_dist > sem_dist_thresh else "颜色变化"
                print(f"[变化检测调试] 匹配对 ({i}->{j}) 检测到变化: {reason}, sem={sem_dist:.3f}, col={col_diff_val:.1f}")

        # 5. 统计未匹配的
        set_indices_1 = set(range(len(masks1)))
        set_indices_2 = set(range(len(masks2)))
        unmatched_1 = list(set_indices_1 - matched_1)
        unmatched_2 = list(set_indices_2 - matched_2)

        changed_count = sum(1 for v in change_info.values() if v['changed'])
        changed_count += len(unmatched_1) + len(unmatched_2)

        return {
            'matches': matches,
            'unmatched_1': unmatched_1,
            'unmatched_2': unmatched_2,
            'change_info': change_info,
            'masks_1': masks1,
            'masks_2': masks2,
            'changed_count': changed_count,
            'total_objects_1': len(masks1),
            'total_objects_2': len(masks2)
        }

    @torch.no_grad()
    def _get_object_embedding_via_crop(self, img_cv, mask):
        """提取单个对象的特征向量"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return torch.zeros(1024).to(self.device)

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        h, w = y_max - y_min, x_max - x_min

        # 裁剪 ROI
        roi = img_cv[y_min:y_max+1, x_min:x_max+1].copy()
        roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
        roi[~roi_mask] = 0

        # 缩放到 224x224
        target_size = 224
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        scale = min(target_size / (h+1e-6), target_size / (w+1e-6))
        new_h, new_w = int(h * scale), int(w * scale)

        if new_h <= 0 or new_w <= 0:
            return torch.zeros(1024).to(self.device)

        roi_resized = cv2.resize(roi, (new_w, new_h))
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = roi_resized

        # 预处理并提取特征
        pil_crop = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        inp = self.pe_preprocess(pil_crop).unsqueeze(0).to(self.device)

        # 判断设备类型
        if isinstance(self.device, str):
            device_type = "cuda" if "cuda" in self.device else "cpu"
        else:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"

        with torch.autocast(device_type):
            # 使用倒数第2层特征 (layer_idx=-2)，与独立脚本一致
            features = self.pe_model.forward_features(inp, layer_idx=-2)
            cls_token = features[:, 0, :]

        return F.normalize(cls_token.squeeze(0), dim=0)

    def _batch_extract_features(self, img, masks):
        """批量提取特征"""
        if len(masks) == 0:
            return torch.empty((0, 1024)).to(self.device)
        feats = []
        for m in masks:
            feats.append(self._get_object_embedding_via_crop(img, m))
        return torch.stack(feats)

    def _match_objects(self, masks1, masks2, vecs1, vecs2, img_shape):
        """
        匹配两张图中的对象
        使用语义相似度、IoU和质心距离的综合得分
        """
        matched_1 = set()
        matched_2 = set()
        matches = []

        if len(vecs1) == 0 or len(vecs2) == 0:
            return matches, matched_1, matched_2

        H, W = img_shape[:2]

        # 计算语义相似度矩阵
        sim = torch.mm(vecs1, vecs2.t()).cpu().numpy()

        n1, n2 = len(masks1), len(masks2)
        scores = np.zeros((n1, n2))

        def get_centroid(m):
            """计算质心"""
            M = cv2.moments(m.astype(np.uint8))
            if M["m00"] == 0:
                return np.array([0.0, 0.0])
            return np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])

        # 计算所有质心
        c1 = [get_centroid(m) for m in masks1]
        c2 = [get_centroid(m) for m in masks2]
        diag = np.sqrt(H**2 + W**2)

        # 计算综合匹配得分
        for i in range(n1):
            for j in range(n2):
                s_sem = sim[i, j]  # 语义相似度

                # IoU
                inter = np.sum(masks1[i] & masks2[j])
                iou = inter / (np.sum(masks1[i] | masks2[j]) + 1e-6)

                # 归一化的质心距离
                dist = np.linalg.norm(c1[i] - c2[j]) / diag

                # 综合得分计算
                if iou > 0:
                    # 有重叠：语义相似度 30% + IoU 70%
                    score = 0.3 * s_sem + 0.7 * iou
                else:
                    # 无重叠：根据距离调整语义得分
                    if dist > 0.15:
                        score = -1  # 距离太远，不匹配
                    else:
                        # 距离越近，得分越高
                        score = s_sem * (1.0 - dist/0.15)

                scores[i, j] = score

        # 使用匈牙利算法进行最优匹配
        rows, cols = linear_sum_assignment(scores, maximize=True)

        # 过滤低分匹配
        for r, c in zip(rows, cols):
            if scores[r, c] > 0.40:  # 匹配阈值
                matches.append((r, c))
                matched_1.add(r)
                matched_2.add(c)

        return matches, matched_1, matched_2

    def _check_color_change_hsv(self, img1, mask1, img2, mask2):
        """
        基于 HSV 空间的鲁棒颜色变化检测
        只比较 H(色相) 和 S(饱和度)，忽略 V(明度) 以抗光照干扰
        """
        def get_dominant_hsv(img, mask):
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pixels = hsv_img[mask > 0]
            if len(pixels) == 0:
                return np.array([0, 0, 0])
            return np.median(pixels, axis=0)

        hsv1 = get_dominant_hsv(img1, mask1)
        hsv2 = get_dominant_hsv(img2, mask2)

        h1, s1, _ = hsv1
        h2, s2, _ = hsv2

        # 阈值参数
        S_THRESH = 20      # 饱和度低于此值视为灰色
        H_DIFF_THRESH = 20    # 色相差异阈值

        is_gray1 = s1 < S_THRESH
        is_gray2 = s2 < S_THRESH

        # 1. 灰度 vs 彩色
        if is_gray1 != is_gray2:
            col_diff_val = abs(s1 - s2)
            # 与独立脚本保持一致：阈值为 20
            if col_diff_val > 20:
                return True, col_diff_val
            return False, col_diff_val

        # 2. 都是灰色
        if is_gray1 and is_gray2:
            return False, 0.0

        # 3. 都是彩色 - 比较色相
        diff = abs(h1 - h2)
        hue_diff = min(diff, 180 - diff)

        if hue_diff > H_DIFF_THRESH:
            return True, hue_diff

        return False, hue_diff

    def visualize(self, img1, img2, result, save_path):
        """
        可视化变化检测结果
        以变化后的图片(img2)为底图，用半透明蒙版标记变化区域：
        - 新增对象：绿色
        - 消失对象：红色
        - 变化对象：黄色

        Args:
            img1, img2: 原始图片
            result: detect_changes() 返回的结果
            save_path: 保存路径
        """
        H, W = img2.shape[:2]

        # 以 img2（变化后的图片）作为底图
        base_img = img2.copy()

        # 1. 先给底图添加一层浅色半透明蒙版（增加对比度）
        # 创建白色半透明遮罩，让底图稍微变亮/变淡
        overlay = np.ones((H, W, 3), dtype=np.uint8) * 230  # 浅灰白色
        base_img_dimmed = cv2.addWeighted(base_img, 0.6, overlay, 0.4, 0)

        # 2. 创建变化区域的蒙版图层
        mask_layer = np.zeros((H, W, 3), dtype=np.uint8)

        # 定义颜色 (BGR 格式，使用更鲜艳的颜色以增强对比)
        COLOR_REMOVED = (100, 100, 255)    # 红色（消失的）- 更鲜艳
        COLOR_NEW = (100, 255, 100)        # 绿色（新增的）- 更鲜艳
        COLOR_CHANGED = (100, 255, 255)    # 黄色（变化的）- 更鲜艳

        masks1 = result['masks_1']
        masks2 = result['masks_2']
        matches = result['matches']
        unmatched_1 = result['unmatched_1']
        unmatched_2 = result['unmatched_2']
        change_info = result['change_info']

        # 调试：统计变化情况
        changed_matches = sum(1 for (i, j) in matches if change_info.get((i, j), {}).get('changed', False))
        unchanged_matches = len(matches) - changed_matches
        print(f"[可视化调试] 匹配对象: {len(matches)}, 其中变化: {changed_matches}, 未变化: {unchanged_matches}")
        print(f"[可视化调试] 新增对象: {len(unmatched_2)}, 消失对象: {len(unmatched_1)}")

        # 3. 绘制消失的对象（在 img1 中存在，但在 img2 中不存在）
        for i in unmatched_1:
            mask_layer[masks1[i]] = COLOR_REMOVED

        # 4. 绘制新增的对象（在 img2 中存在，但在 img1 中不存在）
        for j in unmatched_2:
            mask_layer[masks2[j]] = COLOR_NEW

        # 5. 绘制发生变化的对象（匹配上但是内容变化了）
        for (i, j) in matches:
            info = change_info.get((i, j), {'changed': False, 'sem': 0, 'col': 0})
            if info['changed']:
                mask_layer[masks2[j]] = COLOR_CHANGED
                print(f"[可视化调试] 变化对象 ({i}->{j}): sem={info['sem']:.3f}, col={info['col']:.1f}")

        # 6. 混合底图和变化蒙版层
        # 使用变淡后的底图，让变化区域更突出
        visual_img = cv2.addWeighted(base_img_dimmed, 0.5, mask_layer, 0.5, 0)

        # 7. 添加轮廓和标签，增强可读性
        def draw_label(img, text, pos, bg_color):
            """绘制带背景的文字标签"""
            (w_txt, h_txt), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x, y = pos
            # 绘制半透明背景
            overlay = img.copy()
            cv2.rectangle(overlay, (x-w_txt//2-5, y-h_txt-5), (x+w_txt//2+5, y+5), bg_color, -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            # 绘制文字
            cv2.putText(img, text, (x-w_txt//2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 8. 绘制消失对象的轮廓和标签
        for i in unmatched_1:
            contours = cv2.findContours(masks1[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if contours:
                cv2.drawContours(visual_img, contours, -1, COLOR_REMOVED, 2)
                M = cv2.moments(masks1[i].astype(np.uint8))
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    draw_label(visual_img, "Removed", (cx, cy), (0, 0, 180))

        # 9. 绘制新增对象的轮廓和标签
        for j in unmatched_2:
            contours = cv2.findContours(masks2[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if contours:
                cv2.drawContours(visual_img, contours, -1, COLOR_NEW, 2)
                M = cv2.moments(masks2[j].astype(np.uint8))
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    draw_label(visual_img, "New", (cx, cy), (0, 180, 0))

        # 10. 绘制变化对象的轮廓和标签
        for (i, j) in matches:
            info = change_info.get((i, j), {'changed': False})
            if info['changed']:
                contours = cv2.findContours(masks2[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if contours:
                    cv2.drawContours(visual_img, contours, -1, COLOR_CHANGED, 2)
                    M = cv2.moments(masks2[j].astype(np.uint8))
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        draw_label(visual_img, "Changed", (cx, cy), (0, 180, 180))

        # 保存结果
        cv2.imwrite(save_path, visual_img)
        return visual_img
