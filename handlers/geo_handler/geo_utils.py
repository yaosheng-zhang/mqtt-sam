import os
import json
import math
import subprocess
import shutil
import tempfile
import requests  # 需要 pip install requests
from pathlib import Path

class GeoCalculator:
    def __init__(self, base_dir, relative_path):
        """
        支持本地路径 或 HTTP/HTTPS URL
        """
        self.valid = False
        self.meta = {}
        self.temp_file_path = None  # 用于记录临时文件路径以便清理

        # 1. 判断是 URL 还是 本地路径
        # 如果 relative_path 是 http 开头，忽略 base_dir
        image_path = os.path.join(base_dir, relative_path)
        if str(base_dir).startswith(('http://', 'https://')):
            self.image_source =  image_path
            self.is_url = True
            self.local_path = None # 稍后下载生成
        else:
            self.image_source = image_path
            self.is_url = False
            self.local_path = Path(self.image_source)

        # 2. 初始化元数据
        try:
            self._load_metadata()
            
            # 3. 如果元数据有效，初始化针孔模型参数
            if self.valid:
                self._init_pinhole_params()
            else:
                print(f"[GeoCalculator] 警告: 图片无法进行地理计算 -> {self.image_source}")
        finally:
            # 4. 无论成功失败，最后都要清理临时文件
            self._cleanup()

    def _download_image(self, url):
        """下载图片到临时文件"""
        try:
            # 创建一个带后缀的临时文件 (exiftool 需要后缀来识别文件类型)
            # delete=False 是因为我们要关闭文件后给 exiftool 用，最后手动删
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(temp_fd) # 关闭底层文件描述符
            
            # 流式下载，防止大文件爆内存
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return Path(temp_path)
        except Exception as e:
            print(f"[GeoCalculator] 图片下载失败: {e}")
            return None

    def _cleanup(self):
        """清理临时文件"""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.remove(self.temp_file_path)
                # print(f"[GeoCalculator] 临时文件已清理: {self.temp_file_path}")
            except Exception as e:
                print(f"[GeoCalculator] 临时文件清理失败: {e}")

    def _load_metadata(self):
        """调用 ExifTool 提取元数据 (支持 URL)"""
        
        # A. 准备文件路径
        target_path = None
        
        if self.is_url:
            # 如果是 URL，先下载
            print(f"[GeoCalculator] 正在下载云端图片: {self.image_source} ...")
            target_path = self._download_image(self.image_source)
            if not target_path:
                return # 下载失败
            self.temp_file_path = target_path # 记录以便 cleanup 删除
        else:
            # 如果是本地文件
            if not self.local_path.exists():
                print(f"[GeoCalculator] 错误: 本地文件不存在 -> {self.local_path}")
                return
            target_path = self.local_path

        # B. 检查 ExifTool 是否存在
        if not shutil.which("exiftool"):
            print("[GeoCalculator] 错误: 系统未安装 'exiftool'")
            return

        # Windows下隐藏控制台窗口
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # C. 执行命令 (针对 target_path)
        cmd = ['exiftool', '-json', '-n', str(target_path)]
        try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, startupinfo=startupinfo)
            data_list = json.loads(result.decode('utf-8'))
            
            if not data_list:
                print("[GeoCalculator] 错误: ExifTool 返回数据为空")
                return
                
            data = data_list[0]
            
            # D. 提取并验证关键数据
            lat = float(data.get('GPSLatitude', 0))
            lon = float(data.get('GPSLongitude', 0))
            
            if lat == 0 and lon == 0:
                print("[GeoCalculator] 错误: 未发现有效 GPS 信息")
                return

            self.meta = {
                'lat': lat,
                'lon': lon,
                'alt': float(data.get('RelativeAltitude', 75.0)), 
                'yaw': float(data.get('GimbalYawDegree', data.get('FlightYawDegree', 0))),
                'w': int(data.get('ImageWidth', 0)),
                'h': int(data.get('ImageHeight', 0)),
                'f': float(data.get('FocalLength', 6.72)) 
            }
            self.valid = True

        except subprocess.CalledProcessError as e:
            print(f"[GeoCalculator] ExifTool 调用失败: {e.output.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"[GeoCalculator] 元数据解析异常: {e}")

    def _init_pinhole_params(self):
        """初始化针孔相机参数"""
        m = self.meta
        try:
            # 传感器估算 (基于35mm等效24mm)
            self.sensor_w = 36.0 * (m['f'] / 24.0)
            
            # GSD 计算
            if m['f'] * m['w'] > 0:
                self.gsd = (m['alt'] * self.sensor_w) / (m['f'] * m['w'])
            else:
                self.gsd = 0
                self.valid = False

            self.cx = m['w'] / 2
            self.cy = m['h'] / 2
            self.R_EARTH = 6378137.0
        except Exception as e:
            print(f"[GeoCalculator] 参数初始化失败: {e}")
            self.valid = False

    def pixel_to_gps(self, u, v):
        """输入像素坐标 (u,v)，返回 (lat, lon)"""
        if not self.valid:
            return None, None
            
        m = self.meta
        
        dx_img = (u - self.cx) * self.gsd
        dy_img = (self.cy - v) * self.gsd 
        
        theta = math.radians(m['yaw'])
        d_east = dx_img * math.cos(theta) + dy_img * math.sin(theta)
        d_north = -dx_img * math.sin(theta) + dy_img * math.cos(theta)
        
        rad_lat = math.radians(m['lat'])
        delta_lat = (d_north / self.R_EARTH) * (180.0 / math.pi)
        delta_lon = (d_east / (self.R_EARTH * math.cos(rad_lat))) * (180.0 / math.pi)
        
        return m['lat'] + delta_lat, m['lon'] + delta_lon