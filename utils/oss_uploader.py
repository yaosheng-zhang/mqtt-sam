"""
公共 OSS 工具类
用于统一管理阿里云 OSS 上传功能
"""
import oss2
import os


class OSSUploader:
    """阿里云 OSS 上传工具类"""

    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name):
        """
        初始化 OSS 上传器

        Args:
            access_key_id: 阿里云 AccessKey ID
            access_key_secret: 阿里云 AccessKey Secret
            endpoint: OSS Endpoint
            bucket_name: OSS Bucket 名称
        """
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.bucket = None

        # 初始化 OSS 客户端
        self._init_oss_client()

    def _init_oss_client(self):
        """初始化 OSS 客户端"""
        try:
            if not self.access_key_id or not self.access_key_secret:
                print(f"[OSS] AccessKey 未配置，OSS 上传功能将不可用")
                return

            auth = oss2.Auth(self.access_key_id, self.access_key_secret)
            self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

            print(f"[OSS] 客户端初始化成功")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Endpoint: {self.endpoint}")

        except Exception as e:
            print(f"[OSS] 客户端初始化失败: {e}")
            self.bucket = None

    def upload_file(self, local_file_path, oss_relative_path):
        """
        上传文件到阿里云 OSS

        Args:
            local_file_path: 本地文件路径
            oss_relative_path: OSS 中的相对路径 (例如: "change_detection/2025-12-25/result.jpg")

        Returns:
            str: 成功时返回 OSS 相对路径，失败时返回 None
        """
        if self.bucket is None:
            print(f"[OSS] 客户端未初始化，跳过上传")
            return None

        if not os.path.exists(local_file_path):
            print(f"[OSS] 本地文件不存在: {local_file_path}")
            return None

        try:
            # 上传文件到 OSS
            with open(local_file_path, 'rb') as f:
                result = self.bucket.put_object(oss_relative_path, f)

            # 检查上传状态
            if result.status == 200:
                print(f"[OSS] 文件上传成功")
                print(f"   本地路径: {local_file_path}")
                print(f"   OSS路径: {oss_relative_path}")
                return oss_relative_path
            else:
                print(f"[OSS] 文件上传失败，状态码: {result.status}")
                return None

        except Exception as e:
            print(f"[OSS] 上传异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def upload_files(self, file_list, oss_base_path=""):
        """
        批量上传文件

        Args:
            file_list: 本地文件路径列表
            oss_base_path: OSS 基础路径前缀

        Returns:
            list: 成功上传的 OSS 路径列表
        """
        oss_paths = []

        for local_path in file_list:
            if not os.path.isfile(local_path):
                continue

            filename = os.path.basename(local_path)
            oss_relative_path = os.path.join(oss_base_path, filename).replace("\\", "/")

            oss_path = self.upload_file(local_path, oss_relative_path)
            if oss_path:
                oss_paths.append(oss_path)

        return oss_paths

    def is_available(self):
        """
        检查 OSS 客户端是否可用

        Returns:
            bool: 可用返回 True，否则返回 False
        """
        return self.bucket is not None

    def download_file(self, oss_url, local_file_path):
        """
        从 OSS 下载文件到本地

        Args:
            oss_url: OSS 文件 URL (可以是完整 URL 或相对路径)
            local_file_path: 本地文件保存路径

        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if self.bucket is None:
            print(f"[OSS] 客户端未初始化，无法下载")
            return False

        try:
            # 从完整 URL 中提取相对路径
            # URL 格式: https://{bucket}.{endpoint}/{path}
            if oss_url.startswith('http://') or oss_url.startswith('https://'):
                # 提取路径部分
                url_parts = oss_url.split('/', 3)
                if len(url_parts) >= 4:
                    oss_relative_path = url_parts[3]
                else:
                    print(f"[OSS] URL 格式不正确: {oss_url}")
                    return False
            else:
                # 已经是相对路径
                oss_relative_path = oss_url

            # 确保本地目录存在
            local_dir = os.path.dirname(local_file_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            # 下载文件
            self.bucket.get_object_to_file(oss_relative_path, local_file_path)

            if os.path.exists(local_file_path):
                print(f"[OSS] 文件下载成功")
                print(f"   OSS路径: {oss_relative_path}")
                print(f"   本地路径: {local_file_path}")
                return True
            else:
                print(f"[OSS] 文件下载失败: 本地文件不存在")
                return False

        except oss2.exceptions.NoSuchKey:
            print(f"[OSS] 文件不存在: {oss_url}")
            return False
        except Exception as e:
            print(f"[OSS] 下载异常: {e}")
            import traceback
            traceback.print_exc()
            return False


def create_oss_uploader_from_config(config, service_type="change_detection"):
    """
    从配置创建 OSS 上传器

    Args:
        config: 配置对象
        service_type: 服务类型 ("change_detection" 或 "gs")

    Returns:
        OSSUploader: OSS 上传器实例
    """
    if service_type == "change_detection":
        return OSSUploader(
            access_key_id=config.change_detection_oss_access_key_id,
            access_key_secret=config.change_detection_oss_access_key_secret,
            endpoint=config.change_detection_oss_endpoint,
            bucket_name=config.change_detection_oss_bucket_name
        )
    elif service_type == "gs":
        return OSSUploader(
            access_key_id=config.gs_oss_access_key_id,
            access_key_secret=config.gs_oss_access_key_secret,
            endpoint=config.gs_oss_endpoint,
            bucket_name=config.gs_oss_bucket_name
        )
    else:
        raise ValueError(f"不支持的服务类型: {service_type}")
