# -*- mode: python ; coding: utf-8 -*-
"""
SAM3 MQTT 推理节点 - PyInstaller 打包配置
优化版本,解决深度学习模型打包问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all
import sys
import os

# ==================== 数据收集配置 ====================

datas = []
binaries = []
hiddenimports = []

# 1. 包含配置文件 (运行必需)
datas += [
    ('config.yaml', '.'),  # 配置文件放到根目录
]

# 2. Transformers 模型和数据文件
# 注意: SAM3 模型文件通常很大(>1GB),不建议打包进exe
# 推荐方式: 模型文件放在外部目录,通过 config.yaml 配置路径
datas += collect_data_files('transformers', include_py_files=True)
hiddenimports += collect_submodules('transformers')
hiddenimports += [
    'transformers.models.sam3',
    'transformers.models.sam3.modeling_sam3',
    'transformers.models.sam3.processing_sam3',
    'transformers.models.sam3.configuration_sam3',
]

# 3. PyTorch 相关
hiddenimports += [
    'torch',
    'torch.nn',
    'torch.cuda',
    'torch._C',
    'torch.jit',
    'torch.jit._script',
]

# 4. MQTT 客户端
hiddenimports += [
    'paho.mqtt.client',
]

# 5. 图像处理
hiddenimports += [
    'PIL',
    'PIL.Image',
    'PIL.ImageFile',
]

# 6. 系统监控
hiddenimports += [
    'psutil',
    'psutil._pswindows',  # Windows 专用
]

# 7. 其他依赖
hiddenimports += [
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',  # NumPy 关键模块
    'yaml',
    'requests',
    'queue',
    'threading',
    'json',
    'datetime',
    'scipy',                          # 变化检测需要 (linear_sum_assignment)
    'scipy.optimize',
]

# 8. NumPy 2.x 兼容性修复 (临时方案,建议降级到 1.26.x)
# 如果必须使用 NumPy 2.x,取消下面的注释
# hiddenimports += [
#     'numpy._core',
#     'numpy._core.multiarray',
#     'numpy._core.umath',
#     'numpy._core._multiarray_umath',
# ]

# ==================== 排除不需要的模块 (减小体积) ====================
excludes = [
    'matplotlib',      # 不需要绘图
    # 'scipy',         # 变化检测需要,不能排除
    'pandas',          # 不需要数据分析
    'jupyter',         # 不需要notebook
    'IPython',
    'tkinter',         # 不需要GUI
    'PySide6',         # 你的项目不用Qt
    'PyQt5',
    'PyQt6',
    'unittest',        # 不需要测试框架
    'pytest',
]

# ==================== 分析配置 ====================
a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],  # 当前目录
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=2,  # 优化级别 (0-2)
)

pyz = PYZ(a.pure)

# ==================== 打包模式选择 ====================
# 模式1: 单文件模式 (推荐用于分发)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='mqtt-sam3',
    debug=False,              # 发布版本关闭调试
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                 # 开启UPX压缩 (可选)
    upx_exclude=[
        'vcruntime140.dll',   # Windows运行时不要压缩
        'python*.dll',
    ],
    runtime_tmpdir=None,
    console=True,             # 保留控制台 (方便看日志)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,  # 可选图标
)

# ==================== 模式2: 文件夹模式 (调试用,启动更快) ====================
# 取消下面的注释,并注释掉上面的 EXE,即可切换为文件夹模式
"""
exe = EXE(
    pyz,
    a.scripts,
    [],  # 不打包 binaries 和 datas
    exclude_binaries=True,
    name='mqtt-sam3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mqtt-sam3',
)
"""
