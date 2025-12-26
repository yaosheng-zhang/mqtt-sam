"""
通用图片下载工具
用于从远程 URL 下载图片到本地
"""
import os
import requests
import tempfile
import shutil
from typing import List, Tuple, Optional


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """
    从 URL 下载单张图片到本地

    Args:
        url: 图片 URL
        save_path: 保存路径
        timeout: 请求超时时间（秒）

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 下载图片
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        # 保存到文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if os.path.exists(save_path):
            print(f"[ImageDownloader] 下载成功: {os.path.basename(save_path)}")
            return True
        else:
            print(f"[ImageDownloader] 下载失败: 文件未保存")
            return False

    except requests.exceptions.Timeout:
        print(f"[ImageDownloader] 下载超时: {url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ImageDownloader] 下载失败: {url} - {e}")
        return False
    except Exception as e:
        print(f"[ImageDownloader] 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_images_batch(
    image_paths: List[str],
    base_url: str = "",
    output_dir: Optional[str] = None
) -> Tuple[List[str], Optional[str]]:
    """
    批量下载图片，支持混合本地路径和远程 URL

    Args:
        image_paths: 图片路径列表（可以是相对路径或完整 URL）
        base_url: 基础 URL，用于拼接相对路径
        output_dir: 输出目录，如果为 None 则创建临时目录

    Returns:
        Tuple[List[str], Optional[str]]:
            - 本地文件路径列表
            - 临时目录路径（如果创建了临时目录）
    """
    # 创建输出目录
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="image_downloads_")
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    local_paths = []

    for img_path in image_paths:
        # 拼接完整 URL
        if img_path.startswith(('http://', 'https://')):
            full_url = img_path
        else:
            # 相对路径拼接为完整 URL
            if base_url:
                if base_url.endswith('/'):
                    full_url = base_url + img_path
                else:
                    full_url = os.path.join(base_url, img_path).replace('\\', '/')
            else:
                full_url = img_path

        # 判断是远程 URL 还是本地文件
        if full_url.startswith(('http://', 'https://')):
            # 远程 URL - 下载到本地
            filename = os.path.basename(img_path)
            local_path = os.path.join(output_dir, filename)

            success = download_image(full_url, local_path)
            if not success:
                # 下载失败，清理临时目录
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError(f"图片下载失败: {full_url}")

            local_paths.append(local_path)

        else:
            # 本地文件路径
            if not os.path.exists(full_url):
                # 文件不存在，清理临时目录
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise FileNotFoundError(f"本地图片不存在: {full_url}")

            local_paths.append(full_url)

    return local_paths, temp_dir


def cleanup_temp_dir(temp_dir: Optional[str]):
    """
    清理临时目录

    Args:
        temp_dir: 临时目录路径
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[ImageDownloader] 临时目录已清理: {temp_dir}")
