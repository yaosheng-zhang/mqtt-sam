"""
Sharp ML 3D Gaussian Splatting REST API 服务
从配置文件读取模型信息，只负责运行预测并返回结果路径
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import shutil
import tempfile
import uuid
from typing import List, Optional
from datetime import datetime
import logging
from config_loader import get_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sharp ML API",
    description="3D Gaussian Splatting API 服务",
    version="1.0.0"
)


class PredictRequest(BaseModel):
    """预测请求模型"""
    input_images: List[str]  # 输入图片路径列表


class PredictResponse(BaseModel):
    """预测响应模型"""
    success: bool
    task_id: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    timestamp: str


# 全局配置
config = None


def load_config():
    """加载配置"""
    global config
    if config is None:
        config = get_config()
    return config


@app.on_event("startup")
async def startup_event():
    """服务启动时加载配置"""
    cfg = load_config()
    logger.info("="*70)
    logger.info("Sharp ML API 服务启动")
    logger.info(f"Sharp 虚拟环境: {cfg.gs_sharp_python_path}")
    logger.info(f"输出目录: {cfg.gs_output_dir}")
    logger.info("="*70)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "Sharp ML API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict_gaussian_splatting(request: PredictRequest):
    """
    执行 3D Gaussian Splatting 预测

    Args:
        request: 包含输入图片路径列表的请求

    Returns:
        预测结果，包括输出路径
    """
    cfg = load_config()
    task_id = str(uuid.uuid4())[:8]  # 短 ID
    timestamp = datetime.now().isoformat()

    logger.info(f"[Task {task_id}] 收到预测请求，图片数量: {len(request.input_images)}")

    try:
        # 验证输入图片是否存在
        missing_images = []
        for img_path in request.input_images:
            if not os.path.exists(img_path):
                missing_images.append(img_path)

        if missing_images:
            error_msg = f"以下图片不存在: {', '.join(missing_images)}"
            logger.error(f"[Task {task_id}] {error_msg}")
            return PredictResponse(
                success=False,
                task_id=task_id,
                error=error_msg,
                timestamp=timestamp
            )

        # 创建临时输入目录
        temp_input_dir = tempfile.mkdtemp(prefix=f"sharp_input_{task_id}_")
        logger.info(f"[Task {task_id}] 临时输入目录: {temp_input_dir}")

        # 复制输入图片到临时目录
        for img_path in request.input_images:
            shutil.copy2(img_path, temp_input_dir)
            logger.info(f"[Task {task_id}] 复制图片: {os.path.basename(img_path)}")

        # 创建输出目录（使用配置中的目录 + task_id）
        output_dir = os.path.join(cfg.gs_output_dir, f"task_{task_id}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"[Task {task_id}] 输出目录: {output_dir}")

        # 构建 sharp predict 命令
        sharp_cmd = [
            cfg.gs_sharp_python_path,
            "-m", "sharp",
            "predict",
            "-i", temp_input_dir,
            "-o", output_dir
        ]

        logger.info(f"[Task {task_id}] 执行命令: {' '.join(sharp_cmd)}")

        # 执行 sharp predict 命令
        result = subprocess.run(
            sharp_cmd,
            capture_output=True,
            text=True,
            timeout=cfg.gs_predict_timeout  # 从配置读取超时时间
        )

        # 清理临时输入目录
        shutil.rmtree(temp_input_dir, ignore_errors=True)

        # 检查执行结果
        if result.returncode != 0:
            logger.error(f"[Task {task_id}] Sharp 命令执行失败")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")

            return PredictResponse(
                success=False,
                task_id=task_id,
                error=f"Sharp 预测失败: {result.stderr}",
                timestamp=timestamp
            )

        logger.info(f"[Task {task_id}] Sharp 预测成功")
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")

        # 返回成功响应
        return PredictResponse(
            success=True,
            task_id=task_id,
            output_path=output_dir,
            timestamp=timestamp
        )

    except subprocess.TimeoutExpired:
        logger.error(f"[Task {task_id}] Sharp 命令执行超时")
        return PredictResponse(
            success=False,
            task_id=task_id,
            error=f"预测超时（超过 {cfg.gs_predict_timeout} 秒）",
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"[Task {task_id}] 预测失败: {str(e)}")
        import traceback
        traceback.print_exc()

        return PredictResponse(
            success=False,
            task_id=task_id,
            error=f"预测失败: {str(e)}",
            timestamp=timestamp
        )


if __name__ == "__main__":
    import uvicorn

    # 从配置读取 API 服务配置
    cfg = load_config()
    host = cfg.gs_api_host
    port = cfg.gs_api_port

    logger.info(f"启动 Sharp ML API 服务: http://{host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
