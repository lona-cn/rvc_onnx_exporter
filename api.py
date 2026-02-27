import os
import uuid
import asyncio
import logging
import traceback
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timedelta

import torch
import onnx
from onnxconverter_common import float16
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RVC PTH to ONNX Exporter",
    description="HTTP API for batch converting RVC .pth model files to ONNX format",
    version="1.0.0"
)

EXPORT_DIR = Path("exported_onnx")
UPLOAD_DIR = Path("uploaded_pth")
STATIC_DIR = Path("static")
EXPORT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

export_tasks = {}


async def cleanup_old_files():
    """清理过期的文件，防止存储空间占用过大"""
    while True:
        try:
            # 清理uploaded_pth目录
            uploaded_pth_dir = UPLOAD_DIR
            if uploaded_pth_dir.exists():
                for file in uploaded_pth_dir.iterdir():
                    if file.is_file():
                        # 检查文件修改时间，超过1小时的文件将被删除
                        if datetime.now().timestamp() - file.stat().st_mtime > 1 * 3600:
                            os.remove(file)
                            logger.info(f"Removed old uploaded file: {file.name}")
            
            # 清理exported_onnx目录
            exported_onnx_dir = EXPORT_DIR
            if exported_onnx_dir.exists():
                for file in exported_onnx_dir.iterdir():
                    if file.is_file():
                        # 检查文件修改时间，超过1小时的文件将被删除
                        if datetime.now().timestamp() - file.stat().st_mtime > 1 * 3600:
                            os.remove(file)
                            logger.info(f"Removed old exported file: {file.name}")
            
            # 清理过期的任务记录
            expired_tasks = []
            for task_id, task in export_tasks.items():
                # 检查任务是否已完成且超过1小时
                if task.status == "completed" and datetime.now().timestamp() - os.path.getmtime(task.output_file) > 1 * 3600:
                    # 删除相关文件
                    if task.input_file and os.path.exists(task.input_file):
                        os.remove(task.input_file)
                    if task.output_file and os.path.exists(task.output_file):
                        os.remove(task.output_file)
                    expired_tasks.append(task_id)
            
            # 从任务列表中移除过期任务
            for task_id in expired_tasks:
                del export_tasks[task_id]
                logger.info(f"Removed expired task: {task_id}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        # 每6小时执行一次清理
        await asyncio.sleep(6 * 3600)


class ExportTask(BaseModel):
    task_id: str
    status: str
    input_file: str
    output_file: Optional[str] = None
    error: Optional[str] = None


class ExportResult(BaseModel):
    task_id: str
    status: str
    input_file: str
    output_file: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None


class BatchExportRequest(BaseModel):
    hidden_channels: int = 256
    opset_version: int = 16


class ModelInfo(BaseModel):
    filename: str
    version: str
    hidden_channels: int
    inter_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    spk_embed_dim: int
    gin_channels: int
    sample_rate: int
    phone_dim: int
    config: list


def get_model_info(pth_path: str, filename: str) -> ModelInfo:
    cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    
    config = cpt["config"]
    version = cpt.get("version", "v2")
    
    if version == "v1":
        phone_dim = 256
    else:
        phone_dim = 768
    
    return ModelInfo(
        filename=filename,
        version=version,
        hidden_channels=config[3],
        inter_channels=config[2],
        filter_channels=config[4],
        n_heads=config[5],
        n_layers=config[6],
        kernel_size=config[7],
        spk_embed_dim=config[15],
        gin_channels=config[16],
        sample_rate=config[17] if isinstance(config[17], int) else 48000,
        phone_dim=phone_dim,
        config=config
    )


def export_pth_to_onnx(
        pth_path: str,
        output_path: str,
        hidden_channels: int = 256,
        opset_version: int = 16,
        quantize_fp16: bool = False
    ) -> bool:
        temp_output_path = None
        try:
            cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            
            logger.info(f"Model config: {cpt['config']}")
            
            version = cpt.get("version", "v2")
            logger.info(f"Model version: {version}")
            logger.info(f"FP16 quantization: {quantize_fp16}")
            
            if version == "v1":
                phone_dim = 256
            else:
                phone_dim = 768
            
            device = "cpu"
            is_half = quantize_fp16

            # 准备测试输入
            test_phone = torch.rand(1, 200, phone_dim)
            test_phone_lengths = torch.tensor([200]).long()
            test_pitch = torch.randint(size=(1, 200), low=5, high=255)
            test_pitchf = torch.rand(1, 200)
            test_ds = torch.LongTensor([0])
            test_rnd = torch.rand(1, 192, 200)

            # 总是使用FP32导出，然后通过ONNX工具转换为FP16
            net_g = SynthesizerTrnMsNSFsidM(*cpt["config"], version, is_half=False)
            net_g.load_state_dict(cpt["weight"], strict=False)
            net_g.eval()
            
            # 导出为FP32模型
            temp_output_path = output_path + ".temp"
            logger.info(f"Exporting to temporary path: {temp_output_path}")
            
            torch.onnx.export(
                net_g,
                (
                    test_phone.to(device),
                    test_phone_lengths.to(device),
                    test_pitch.to(device),
                    test_pitchf.to(device),
                    test_ds.to(device),
                    test_rnd.to(device),
                ),
                temp_output_path,
                dynamic_axes={
                    "phone": [1],
                    "pitch": [1],
                    "pitchf": [1],
                    "rnd": [2],
                },
                do_constant_folding=False,
                opset_version=opset_version,
                verbose=False,
                input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
                output_names=["audio"],
            )
            
            logger.info(f"Successfully exported to temporary path: {temp_output_path}")
            
            # 如果需要FP16量化，使用ONNX工具进行转换
            if quantize_fp16:
                logger.info("Converting to FP16...")
                try:
                    # 加载导出的ONNX模型
                    model = onnx.load(temp_output_path)
                    logger.info(f"Loaded model with {len(model.graph.node)} nodes")
                    # 转换为FP16
                    model_fp16 = float16.convert_float_to_float16(model)
                    logger.info(f"Converted model with {len(model_fp16.graph.node)} nodes")
                    # 保存FP16模型
                    onnx.save(model_fp16, output_path)
                    logger.info("Successfully converted to FP16")
                except Exception as e:
                    logger.error(f"Error during FP16 conversion: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    # 如果转换失败，使用原始的FP32模型
                    logger.warning("Falling back to FP32 model")
                    import shutil
                    shutil.copy(temp_output_path, output_path)
            else:
                # 直接使用FP32模型
                import shutil
                shutil.copy(temp_output_path, output_path)
                logger.info("Using FP32 model")
            
            # 检查生成的文件大小
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Generated file size: {file_size / 1024 / 1024:.2f} MB")
            
            logger.info(f"Successfully exported {pth_path} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export {pth_path}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise e
        finally:
            # 删除临时文件
            if temp_output_path and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                    logger.info(f"Removed temporary file: {temp_output_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file: {str(e)}")


async def export_task_wrapper(
    task_id: str,
    pth_path: str,
    output_path: str,
    hidden_channels: int,
    opset_version: int,
    quantize_fp16: bool = False
):
    try:
        export_tasks[task_id].status = "processing"
        
        await asyncio.to_thread(
            export_pth_to_onnx,
            pth_path,
            output_path,
            hidden_channels,
            opset_version,
            quantize_fp16
        )
        
        export_tasks[task_id].status = "completed"
        export_tasks[task_id].output_file = output_path
        
    except Exception as e:
        export_tasks[task_id].status = "failed"
        export_tasks[task_id].error = str(e)


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/model/info", response_model=ModelInfo)
async def get_pth_info(file: UploadFile = File(...)):
    if not file.filename.endswith('.pth'):
        raise HTTPException(status_code=400, detail="Only .pth files are supported")
    
    temp_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{temp_id}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        info = get_model_info(str(temp_path), file.filename)
        return info
    finally:
        if temp_path.exists():
            os.remove(temp_path)


@app.post("/export/single", response_model=ExportResult)
async def export_single(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    hidden_channels: int = Form(256),
    opset_version: int = Form(16),
    quantize_fp16: str = Form("false")
):
    # 转换字符串为布尔值
    quantize_fp16 = quantize_fp16.lower() == "true"
    if not file.filename.endswith('.pth'):
        raise HTTPException(status_code=400, detail="Only .pth files are supported")
    
    task_id = str(uuid.uuid4())
    pth_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    output_filename = file.filename.replace('.pth', '.onnx')
    output_path = EXPORT_DIR / f"{task_id}_{output_filename}"
    
    with open(pth_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    export_tasks[task_id] = ExportTask(
        task_id=task_id,
        status="pending",
        input_file=str(pth_path)
    )
    
    logger.info(f"Received export request: hidden_channels={hidden_channels}, opset_version={opset_version}, quantize_fp16={quantize_fp16}")
    
    background_tasks.add_task(
        export_task_wrapper,
        task_id,
        str(pth_path),
        str(output_path),
        hidden_channels,
        opset_version,
        quantize_fp16
    )
    
    return ExportResult(
        task_id=task_id,
        status="pending",
        input_file=file.filename,
        download_url=f"/download/{task_id}"
    )


@app.post("/export/batch", response_model=List[ExportResult])
async def export_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    hidden_channels: int = Form(256),
    opset_version: int = Form(16),
    quantize_fp16: str = Form("false")
):
    # 转换字符串为布尔值
    quantize_fp16 = quantize_fp16.lower() == "true"
    results = []
    
    for file in files:
        if not file.filename.endswith('.pth'):
            continue
        
        task_id = str(uuid.uuid4())
        pth_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        output_filename = file.filename.replace('.pth', '.onnx')
        output_path = EXPORT_DIR / f"{task_id}_{output_filename}"
        
        with open(pth_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        export_tasks[task_id] = ExportTask(
            task_id=task_id,
            status="pending",
            input_file=str(pth_path)
        )
        
        background_tasks.add_task(
            export_task_wrapper,
            task_id,
            str(pth_path),
            str(output_path),
            hidden_channels,
            opset_version,
            quantize_fp16
        )
        
        results.append(ExportResult(
            task_id=task_id,
            status="pending",
            input_file=file.filename,
            download_url=f"/download/{task_id}"
        ))
    
    return results


@app.get("/status/{task_id}", response_model=ExportResult)
async def get_status(task_id: str):
    if task_id not in export_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = export_tasks[task_id]
    return ExportResult(
        task_id=task.task_id,
        status=task.status,
        input_file=Path(task.input_file).name,
        output_file=Path(task.output_file).name if task.output_file else None,
        download_url=f"/download/{task_id}" if task.status == "completed" else None,
        error=task.error
    )


@app.get("/download/{task_id}")
async def download_onnx(task_id: str):
    if task_id not in export_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = export_tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    if not task.output_file or not os.path.exists(task.output_file):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # 移除UUID前缀，使用原始文件名
    output_filename = Path(task.output_file).name
    # 提取原始文件名（移除UUID前缀）
    if "_" in output_filename:
        # 找到第一个下划线的位置
        first_underscore = output_filename.find("_")
        # 提取下划线后的部分作为新文件名
        clean_filename = output_filename[first_underscore + 1:]
    else:
        clean_filename = output_filename
    
    return FileResponse(
        task.output_file,
        media_type="application/octet-stream",
        filename=clean_filename
    )


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "input_file": Path(task.input_file).name,
                "output_file": Path(task.output_file).name if task.output_file else None,
                "error": task.error
            }
            for task in export_tasks.values()
        ]
    }


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in export_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = export_tasks[task_id]
    
    if task.input_file and os.path.exists(task.input_file):
        os.remove(task.input_file)
    
    if task.output_file and os.path.exists(task.output_file):
        os.remove(task.output_file)
    
    del export_tasks[task_id]
    
    return {"message": "Task deleted successfully"}


@app.on_event("startup")
async def startup_event():
    """服务启动时执行的任务"""
    # 启动清理任务
    asyncio.create_task(cleanup_old_files())
    logger.info("Cleanup task started")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
