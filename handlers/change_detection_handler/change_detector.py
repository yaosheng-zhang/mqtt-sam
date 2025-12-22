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
        H, W = img1_cv.shape[:2]

        # 1. 生成 Mask
        masks1 = self.sam3_generator.generate(img1_cv)
        masks2 = self.sam3_generator.generate(img2_cv)

        # 2. 提取特征
        vecs1 = self._batch_extract_features(img1_cv, masks1)
        vecs2 = self._batch_extract_features(img2_cv, masks2)

        # 3. 匹配
        matches, matched_1, matched_2 = self._match_objects(
            masks1, masks2, vecs1, vecs2, img1_cv.shape
        )

        # 4. 分析变化
        change_info = {}
        for (i, j) in matches:
            sem_dist = 1.0 - torch.dot(vecs1[i], vecs2[j]).item()
            is_color_changed, col_diff_val = self._check_color_change_hsv(
                img1_cv, masks1[i], img2_cv, masks2[j]
            )
            is_changed = (sem_dist > sem_dist_thresh) or is_color_changed
            change_info[(i, j)] = {
                'changed': is_changed,
                'sem': sem_dist,
                'col': col_diff_val
            }

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

        # 判断设备类型 (self.device 是字符串)
        device_type = "cuda" if isinstance(self.device, str) and "cuda" in self.device else "cpu"
        with torch.autocast(device_type):
            features = self.pe_model.forward_features(inp, layer_idx=-3)
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
        """匹配两张图中的对象"""
        matched_1 = set()
        matched_2 = set()
        matches = []

        if len(vecs1) == 0 or len(vecs2) == 0:
            return matches, matched_1, matched_2

        H, W = img_shape[:2]
        sim = torch.mm(vecs1, vecs2.t()).cpu().numpy()

        n1, n2 = len(masks1), len(masks2)
        scores = np.zeros((n1, n2))

        def get_centroid(m):
            M = cv2.moments(m.astype(np.uint8))
            if M["m00"] == 0:
                return np.array([0.0, 0.0])
            return np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])

        c1 = [get_centroid(m) for m in masks1]
        c2 = [get_centroid(m) for m in masks2]
        diag = np.sqrt(H**2 + W**2)

        # 计算匹配得分
        for i in range(n1):
            for j in range(n2):
                s_sem = sim[i, j]
                inter = np.sum(masks1[i] & masks2[j])
                iou = inter / (np.sum(masks1[i] | masks2[j]) + 1e-6)
                dist = np.linalg.norm(c1[i] - c2[j]) / diag

                if iou > 0:
                    score = 0.3 * s_sem + 0.7 * iou
                else:
                    if dist > 0.15:
                        score = -1
                    else:
                        score = s_sem * (1.0 - dist/0.15)
                scores[i, j] = score

        # 匈牙利算法最优匹配
        rows, cols = linear_sum_assignment(scores, maximize=True)
        for r, c in zip(rows, cols):
            if scores[r, c] > 0.40:
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
            if col_diff_val > 30:
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

        Args:
            img1, img2: 原始图片
            result: detect_changes() 返回的结果
            save_path: 保存路径
        """
        H, W = img1.shape[:2]

        # 创建画布
        base_canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
        base_canvas[:, :W, :] = img1
        base_canvas[:, W:, :] = img2

        mask_layer = base_canvas.copy()

        COLOR_CHANGED = (0, 0, 255)      # Red
        COLOR_UNCHANGED = (0, 255, 0)    # Green

        masks1 = result['masks_1']
        masks2 = result['masks_2']
        matches = result['matches']
        unmatched_1 = result['unmatched_1']
        unmatched_2 = result['unmatched_2']
        change_info = result['change_info']

        # 绘制匹配的对象
        for (i, j) in matches:
            info = change_info.get((i, j), {'changed': False})
            color = COLOR_CHANGED if info['changed'] else COLOR_UNCHANGED
            mask_layer[:H, :W][masks1[i]] = color
            mask_layer[:H, W:][masks2[j]] = color

        # 绘制未匹配的对象
        for i in unmatched_1:
            mask_layer[:H, :W][masks1[i]] = COLOR_CHANGED

        for j in unmatched_2:
            mask_layer[:H, W:][masks2[j]] = COLOR_CHANGED

        # 混合图层
        visual_img = cv2.addWeighted(base_canvas, 0.6, mask_layer, 0.4, 0)

        def draw_label(img, text, pos):
            (w_txt, h_txt), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            x, y = pos
            cv2.rectangle(img, (x-w_txt//2-2, y-h_txt-2), (x+w_txt//2+2, y+4), (0,0,0), -1)
            cv2.putText(img, text, (x-w_txt//2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # 绘制连线和标签
        for (i, j) in matches:
            info = change_info.get((i, j), {'changed': False, 'sem': 0, 'col': 0})
            color = COLOR_CHANGED if info['changed'] else COLOR_UNCHANGED

            cnt1 = cv2.findContours(masks1[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            cnt2 = cv2.findContours(masks2[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            cnt2_shifted = cnt2 + np.array([W, 0])

            cv2.drawContours(visual_img, [cnt1], -1, color, 2)
            cv2.drawContours(visual_img, [cnt2_shifted], -1, color, 2)

            M1 = cv2.moments(masks1[i].astype(np.uint8))
            c1 = (int(M1["m10"]/(M1["m00"]+1e-6)), int(M1["m01"]/(M1["m00"]+1e-6)))

            M2 = cv2.moments(masks2[j].astype(np.uint8))
            c2 = (int(M2["m10"]/(M2["m00"]+1e-6)), int(M2["m01"]/(M2["m00"]+1e-6)))
            p2 = (c2[0]+W, c2[1])

            cv2.line(visual_img, c1, p2, color, 1)
            mid = ((c1[0]+p2[0])//2, (c1[1]+p2[1])//2)

            col_txt = ""
            if info['changed'] and info['col'] > 0:
                col_txt = f" C:{int(info['col'])}"
            draw_label(visual_img, f"S:{info['sem']:.2f}{col_txt}", mid)

        # 绘制 Removed 标签
        for i in unmatched_1:
            cnt = cv2.findContours(masks1[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            cv2.drawContours(visual_img, [cnt], -1, COLOR_CHANGED, 2)
            M = cv2.moments(masks1[i].astype(np.uint8))
            c = (int(M["m10"]/(M["m00"]+1e-6)), int(M["m01"]/(M["m00"]+1e-6)))
            draw_label(visual_img, "Removed", c)

        # 绘制 New 标签
        for j in unmatched_2:
            cnt = cv2.findContours(masks2[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            cnt_shifted = cnt + np.array([W, 0])
            cv2.drawContours(visual_img, [cnt_shifted], -1, COLOR_CHANGED, 2)
            M = cv2.moments(masks2[j].astype(np.uint8))
            c = (int(M["m10"]/(M["m00"]+1e-6)) + W, int(M["m01"]/(M["m00"]+1e-6)))
            draw_label(visual_img, "New", c)

        # 保存结果
        cv2.imwrite(save_path, visual_img)
        return visual_img
