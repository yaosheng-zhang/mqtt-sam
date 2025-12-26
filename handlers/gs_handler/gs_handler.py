"""
GS (Gaussian Splatting) Handler
处理 3D 高斯点云重建请求
"""
from ..base_handler import MessageHandler
from datetime import datetime
from utils import create_oss_uploader_from_config, convert_directory_ply_to_sog
import requests
import os
import glob


class GSHandler(MessageHandler):
    """Gaussian Splatting 服务处理器"""

    def __init__(self, ai_engine, config):
        super().__init__(ai_engine, config)

        # Sharp API 配置
        self.api_url = f"http://{config.gs_api_host}:{config.gs_api_port}"

        # 输出目录
        self.output_dir = config.gs_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化 OSS 上传器（使用公共工具）
        self.oss_uploader = create_oss_uploader_from_config(config, service_type="gs")

        print(f"[GS] 初始化完成")
        print(f"   Sharp API: {self.api_url}")
        print(f"   输出目录: {self.output_dir}")

    @property
    def subscribe_topic(self):
        return self.config.gs_subscribe_topic

    def get_name(self):
        return "Gaussian Splatting 服务"

    def on_message(self, dev_id, data, publish):
        """
        处理 Gaussian Splatting 请求

        消息格式:
        {
            "input_images": ["relative/path/image1.jpg", "relative/path/image2.jpg", ...],
            "output_name": "optional_output_name"  # 可选
        }
        """
        try:
            # 1. 解析输入
            input_images = data.get("input_images", [])
            output_name = data.get("output_name")

            if not input_images or not isinstance(input_images, list):
                error_msg = "缺少必需参数: input_images（图片相对路径列表）"
                print(f"[GS] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return

            print(f"[GS] 开始处理 Gaussian Splatting:")
            print(f"   输入图片数量: {len(input_images)}")

            # 2. 处理图片路径（相对路径 + base_path）
            base_path = self.config.ai_image_path or ""
            processed_images = []

            for img_path in input_images:
                # 拼接完整路径（与其他服务保持一致）
                if img_path.startswith(('http://', 'https://')):
                    # 如果是远程 URL，直接拼接
                    full_path = os.path.join(base_path, img_path) if base_path and not base_path.endswith('/') else base_path + img_path
                else:
                    # 如果是相对路径，拼接为完整路径
                    full_path = os.path.join(base_path, img_path) if base_path else img_path

                print(f"   图片: {img_path} -> {full_path}")

                # 验证本地文件是否存在（远程 URL 跳过）
                if not full_path.startswith(('http://', 'https://')):
                    if not os.path.exists(full_path):
                        error_msg = f"图片文件不存在: {full_path}"
                        print(f"[GS] {error_msg}")
                        self._publish_error(dev_id, error_msg, publish)
                        return

                processed_images.append(full_path)

            # 3. 调用 Sharp API 进行预测
            print(f"[GS] 调用 Sharp API: {self.api_url}/predict")

            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"input_images": processed_images},
                    timeout=self.config.gs_api_timeout
                )
                response.raise_for_status()
                api_result = response.json()
            except requests.exceptions.ConnectionError:
                error_msg = f"无法连接到 Sharp API 服务 ({self.api_url})，请确保服务已启动"
                print(f"[GS] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return
            except requests.exceptions.Timeout:
                error_msg = f"Sharp API 调用超时（超过 {self.config.gs_api_timeout} 秒）"
                print(f"[GS] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return
            except Exception as e:
                error_msg = f"Sharp API 调用失败: {str(e)}"
                print(f"[GS] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return

            # 4. 检查 API 返回结果
            if not api_result.get("success"):
                error_msg = api_result.get("error", "Sharp API 预测失败")
                print(f"[GS] {error_msg}")
                self._publish_error(dev_id, error_msg, publish)
                return

            output_path = api_result.get("output_path")
            task_id = api_result.get("task_id")

            print(f"[GS] Sharp 预测成功")
            print(f"   Task ID: {task_id}")
            print(f"   输出目录: {output_path}")

            # 5. 将 PLY 文件转换为 SOG 文件
            print(f"[GS] 开始转换 PLY 到 SOG...")

            try:
                sog_files = convert_directory_ply_to_sog(
                    input_dir=output_path,
                    output_dir=output_path  # 输出到同一目录
                )
                print(f"[GS] PLY 转换完成，生成 {len(sog_files)} 个 SOG 文件")
            except Exception as e:
                print(f"[GS] PLY 转换失败: {e}")
                import traceback
                traceback.print_exc()
                # 转换失败不影响流程，继续上传原始 PLY 文件
                sog_files = []

            # 6. 收集所有需要上传的文件（仅 SOG）
            all_files = []

            # 只添加 SOG 文件（不上传 PLY 文件）
            all_files.extend(sog_files)

            # 获取 PLY 文件数量（用于统计）
            ply_files = glob.glob(os.path.join(output_path, "*.ply"))

            print(f"[GS] 找到 {len(all_files)} 个 SOG 文件待上传")
            for f in all_files:
                print(f"   - {os.path.basename(f)}")

            # 7. 上传到 OSS（路径：/appleaigs/文件名）
            oss_paths = []

            if self.oss_uploader.is_available():
                for file_path in all_files:
                    if os.path.isfile(file_path):
                        filename = os.path.basename(file_path)
                        # 生成 OSS 路径: appleaigs/filename（直接使用文件名，不包含日期和task_id子目录）
                        oss_relative_path = f"appleaigs/{filename}"

                        oss_path = self.oss_uploader.upload_file(file_path, oss_relative_path)
                        if oss_path:
                            oss_paths.append(oss_path)
            else:
                print(f"[GS] OSS 上传器不可用，跳过上传")

            # 8. 构造响应
            response_data = {
                "success": True,
                "task_id": task_id,
                "input_images": input_images,  # 返回原始相对路径
                "output_path": output_path,  # 本地路径
                "oss_paths": oss_paths,  # OSS 路径列表
                "file_count": len(all_files),
                "sog_count": len(sog_files),
                "ply_count": len(ply_files),
                "timestamp": datetime.now().isoformat()
            }

            # 9. 发布结果
            publish(
                topic=self.config.get_gs_response_topic(dev_id),
                data=response_data
            )

            print(f"[GS] 处理完成，结果已发布")
            print(f"   PLY 文件: {len(ply_files)}")
            print(f"   SOG 文件: {len(sog_files)}")
            print(f"   OSS 上传: {len(oss_paths)}")

        except Exception as e:
            error_msg = f"Gaussian Splatting 处理失败: {str(e)}"
            print(f"[GS] {error_msg}")
            import traceback
            traceback.print_exc()
            self._publish_error(dev_id, error_msg, publish)

    def _publish_error(self, dev_id, error_msg, publish):
        """发布错误消息"""
        publish(
            topic=self.config.get_gs_response_topic(dev_id),
            data={
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        )
