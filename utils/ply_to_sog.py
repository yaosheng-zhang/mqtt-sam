"""
PLY 到 SOG (Scene Object Graph) 转换工具
"""
import os
import json
import struct
import numpy as np


class PLYToSOGConverter:
    """PLY 文件到 SOG 文件转换器"""

    def __init__(self):
        pass

    def convert(self, ply_file_path, output_sog_path=None):
        """
        将 PLY 文件转换为 SOG 文件

        Args:
            ply_file_path: PLY 文件路径
            output_sog_path: 输出 SOG 文件路径（可选，默认与 PLY 同目录）

        Returns:
            str: 生成的 SOG 文件路径
        """
        if not os.path.exists(ply_file_path):
            raise FileNotFoundError(f"PLY 文件不存在: {ply_file_path}")

        # 如果未指定输出路径，使用与 PLY 相同的目录和文件名
        if output_sog_path is None:
            base_name = os.path.splitext(ply_file_path)[0]
            output_sog_path = f"{base_name}.sog"

        print(f"[PLY2SOG] 开始转换: {ply_file_path} -> {output_sog_path}")

        # 读取 PLY 文件
        ply_data = self._read_ply(ply_file_path)

        # 转换为 SOG 格式
        sog_data = self._convert_to_sog(ply_data)

        # 保存 SOG 文件
        self._write_sog(sog_data, output_sog_path)

        print(f"[PLY2SOG] 转换完成: {output_sog_path}")

        return output_sog_path

    def _read_ply(self, ply_file_path):
        """
        读取 PLY 文件

        Args:
            ply_file_path: PLY 文件路径

        Returns:
            dict: PLY 数据（包含顶点、法线、颜色等）
        """
        try:
            # 尝试使用 plyfile 库读取（如果已安装）
            try:
                from plyfile import PlyData
                plydata = PlyData.read(ply_file_path)

                # 提取顶点数据
                vertices = plydata['vertex'].data
                vertex_count = len(vertices)

                # 提取坐标
                positions = np.column_stack([
                    vertices['x'],
                    vertices['y'],
                    vertices['z']
                ])

                # 尝试提取法线（如果有）
                normals = None
                if 'nx' in vertices.dtype.names:
                    normals = np.column_stack([
                        vertices['nx'],
                        vertices['ny'],
                        vertices['nz']
                    ])

                # 尝试提取颜色（如果有）
                colors = None
                if 'red' in vertices.dtype.names:
                    colors = np.column_stack([
                        vertices['red'],
                        vertices['green'],
                        vertices['blue']
                    ])

                return {
                    'positions': positions,
                    'normals': normals,
                    'colors': colors,
                    'vertex_count': vertex_count
                }

            except ImportError:
                # 如果没有安装 plyfile，使用简单的解析器
                print("[PLY2SOG] plyfile 库未安装，使用简化解析")
                return self._read_ply_simple(ply_file_path)

        except Exception as e:
            print(f"[PLY2SOG] 读取 PLY 文件失败: {e}")
            raise

    def _read_ply_simple(self, ply_file_path):
        """
        简单的 PLY 文件读取器（不依赖 plyfile 库）

        Args:
            ply_file_path: PLY 文件路径

        Returns:
            dict: PLY 数据
        """
        # 这是一个简化版本，仅支持基本的 ASCII PLY 格式
        # 如果需要支持二进制 PLY，建议安装 plyfile 库

        positions = []
        normals = []
        colors = []

        with open(ply_file_path, 'r') as f:
            # 跳过头部
            line = f.readline()
            while line.strip() != "end_header":
                line = f.readline()

            # 读取顶点数据
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    positions.append([float(parts[0]), float(parts[1]), float(parts[2])])

                    # 如果有法线
                    if len(parts) >= 6:
                        normals.append([float(parts[3]), float(parts[4]), float(parts[5])])

                    # 如果有颜色
                    if len(parts) >= 9:
                        colors.append([int(parts[6]), int(parts[7]), int(parts[8])])

        return {
            'positions': np.array(positions),
            'normals': np.array(normals) if normals else None,
            'colors': np.array(colors) if colors else None,
            'vertex_count': len(positions)
        }

    def _convert_to_sog(self, ply_data):
        """
        将 PLY 数据转换为 SOG 格式

        Args:
            ply_data: PLY 数据字典

        Returns:
            dict: SOG 数据
        """
        # SOG 格式定义（根据实际需求调整）
        sog_data = {
            "version": "1.0",
            "type": "gaussian_splatting",
            "metadata": {
                "vertex_count": ply_data['vertex_count'],
                "has_normals": ply_data['normals'] is not None,
                "has_colors": ply_data['colors'] is not None
            },
            "data": {
                "positions": ply_data['positions'].tolist(),
            }
        }

        # 添加法线（如果有）
        if ply_data['normals'] is not None:
            sog_data["data"]["normals"] = ply_data['normals'].tolist()

        # 添加颜色（如果有）
        if ply_data['colors'] is not None:
            sog_data["data"]["colors"] = ply_data['colors'].tolist()

        return sog_data

    def _write_sog(self, sog_data, output_path):
        """
        写入 SOG 文件

        Args:
            sog_data: SOG 数据
            output_path: 输出文件路径
        """
        # 将 SOG 数据保存为 JSON 格式
        # 如果需要二进制格式，可以修改这部分
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sog_data, f, indent=2)


def convert_ply_to_sog(ply_file_path, output_sog_path=None):
    """
    便捷函数：将 PLY 文件转换为 SOG 文件

    Args:
        ply_file_path: PLY 文件路径
        output_sog_path: 输出 SOG 文件路径（可选）

    Returns:
        str: 生成的 SOG 文件路径
    """
    converter = PLYToSOGConverter()
    return converter.convert(ply_file_path, output_sog_path)


def convert_directory_ply_to_sog(input_dir, output_dir=None):
    """
    批量转换目录下的所有 PLY 文件为 SOG 文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选，默认与输入目录相同）

    Returns:
        list: 生成的 SOG 文件路径列表
    """
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    converter = PLYToSOGConverter()
    sog_files = []

    # 查找所有 PLY 文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.ply'):
            ply_path = os.path.join(input_dir, filename)
            sog_filename = os.path.splitext(filename)[0] + '.sog'
            sog_path = os.path.join(output_dir, sog_filename)

            try:
                converter.convert(ply_path, sog_path)
                sog_files.append(sog_path)
            except Exception as e:
                print(f"[PLY2SOG] 转换失败 {filename}: {e}")

    return sog_files
