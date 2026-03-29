# RVC ONNX Exporter

[![Docker Hub](https://img.shields.io/docker/pulls/lonacn/rvc_onnx_exporter?style=flat)](https://hub.docker.com/r/lonacn/rvc_onnx_exporter)
[![Docker Hub](https://img.shields.io/docker/v/lonacn/rvc_onnx_exporter?label=docker%20hub&sort=semver)](https://hub.docker.com/r/lonacn/rvc_onnx_exporter)

## 项目简介

RVC ONNX Exporter 是一个专业工具，用于将 RVC (Retrieval-based Voice Conversion) 模型从 .pth 格式转换为 ONNX 格式。

RVC ONNX Exporter is a professional tool for converting RVC (Retrieval-based Voice Conversion) models from .pth format to ONNX format.

## 核心功能

- **Web 界面**：直观的网页界面，支持拖拽上传和模型信息展示
- **自动模型信息探测**：自动读取并展示模型的详细参数信息
- **自动存储空间管理**：定期清理过期文件，避免存储空间占用过大
- **无 UUID 前缀**：下载的 ONNX 文件使用原始文件名，方便用户使用

## Key Features

- **Web Interface**：Intuitive web interface with drag-and-drop upload and model information display
- **Automatic Model Info Detection**：Automatically reads and displays detailed model parameter information
- **Automatic Storage Management**：Regularly cleans up expired files to avoid excessive storage usage
- **No UUID Prefix**：Downloaded ONNX files use original filenames for user convenience

## 快速开始

### 环境要求
- Python 3.9+
- PyTorch 2.1.0+ (CPU-only，约 185MB)
- ONNX 1.14.0+
- FastAPI
- Uvicorn

### 安装步骤
1. 克隆项目：`git clone https://github.com/yourusername/rvc_onnx_exporter.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 启动服务：`python api.py`

> 💡 本地安装时，requirements.txt 会自动安装 CPU-only 版本的 PyTorch

服务将在 http://0.0.0.0:8000 上运行。

## Quick Start

### Requirements
- Python 3.9+
- PyTorch 2.1.0+ (CPU-only, ~185MB)
- ONNX 1.14.0+
- FastAPI
- Uvicorn

### Installation
1. Clone the project: `git clone https://github.com/yourusername/rvc_onnx_exporter.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the service: `python api.py`

> 💡 The requirements.txt will automatically install CPU-only PyTorch

The service will run on http://0.0.0.0:8000.

## 使用方法

1. 打开浏览器，访问 http://localhost:8000
2. 拖拽 .pth 文件到上传区域，或点击选择文件
3. 系统会自动探测模型信息并显示
4. 点击 "开始转换" 按钮
5. 等待转换完成后，点击 "下载 ONNX" 按钮获取转换后的模型

## Usage

1. Open your browser and visit http://localhost:8000
2. Drag and drop .pth files to the upload area, or click to select files
3. The system will automatically detect and display model information
4. Click the "开始转换" (Start Conversion) button
5. After conversion is complete, click the "下载 ONNX" (Download ONNX) button to get the converted model

## API 文档

服务启动后，可以访问 http://localhost:8000/docs 查看详细的 API 文档。

主要 API 端点：
- `GET /`：访问 Web 界面
- `POST /model/info`：获取模型信息
- `POST /export/single`：转换单个模型
- `GET /status/{task_id}`：获取转换任务状态
- `GET /download/{task_id}`：下载转换后的模型

## API Documentation

After starting the service, you can visit http://localhost:8000/docs to view the detailed API documentation.

Main API endpoints:
- `GET /`：Access the web interface
- `POST /model/info`：Get model information
- `POST /export/single`：Convert a single model
- `GET /status/{task_id}`：Get conversion task status
- `GET /download/{task_id}`：Download the converted model

## 常见问题

### Q: 转换过程中出现错误怎么办？
A: 请检查模型文件是否完整，以及是否符合 RVC 模型格式要求。如果问题持续存在，请查看服务日志获取详细错误信息。

## FAQ

### Q: What should I do if I encounter errors during conversion?
A: Please check if the model file is complete and conforms to the RVC model format requirements. If the problem persists, please check the service logs for detailed error information.

## 容器化部署 (Docker)

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PORT` | `8000` | 服务监听端口 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `DATA_DIR` | `/app/data` | 数据存储目录 |
| `CLEANUP_INTERVAL_MINUTES` | `5` | 清理任务执行间隔（分钟） |
| `FILE_EXPIRE_MINUTES` | `30` | 文件过期时间（分钟） |

### Docker Compose 快速部署

```bash
# 克隆项目
git clone https://github.com/yourusername/rvc_onnx_exporter.git
cd rvc_onnx_exporter

# 启动服务（使用默认配置）
docker-compose up -d

# 自定义配置启动
PORT=8080 DATA_DIR=/data docker-compose up -d
```

> 💡 Docker 镜像使用 CPU-only PyTorch，体积仅约 **1GB**

### Docker 手动部署

```bash
# 构建镜像
docker build -t rvc-onnx-exporter .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v ./data:/app/data \
  -e PORT=8000 \
  -e CLEANUP_INTERVAL_MINUTES=5 \
  -e FILE_EXPIRE_MINUTES=30 \
  rvc-onnx-exporter
```

### 健康检查

服务提供 `/health` 端点用于健康检查：

```bash
# 检查健康状态
curl http://localhost:8000/health

# 响应示例
{
  "status": "healthy",
  "config": {
    "port": 8000,
    "data_dir": "/app/data",
    "cleanup_interval_minutes": 5,
    "file_expire_minutes": 30
  },
  "directories": {
    "export_dir": "/app/data/exported_onnx",
    "upload_dir": "/app/data/uploaded_pth",
    "static_dir": "/app/data/static"
  }
}
```

### 数据持久化

通过 `DATA_DIR` 环境变量或 volume 挂载来持久化数据：

```yaml
# docker-compose.yml
volumes:
  - ./my-data:/app/data  # 持久化所有数据
```

## Container Deployment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Service listening port |
| `HOST` | `0.0.0.0` | Service binding address |
| `DATA_DIR` | `/app/data` | Data storage directory |
| `CLEANUP_INTERVAL_MINUTES` | `5` | Cleanup task interval (minutes) |
| `FILE_EXPIRE_MINUTES` | `30` | File expiration time (minutes) |

### Quick Start with Docker Compose

```bash
# Clone project
git clone https://github.com/yourusername/rvc_onnx_exporter.git
cd rvc-onnx_exporter

# Start service with default config
docker-compose up -d

# Start with custom config
PORT=8080 DATA_DIR=/data docker-compose up -d
```

### Manual Docker Deployment

```bash
# Build image
docker build -t rvc-onnx-exporter .

# Run container
docker run -d \
  -p 8000:8000 \
  -v ./data:/app/data \
  -e PORT=8000 \
  -e CLEANUP_INTERVAL_MINUTES=5 \
  -e FILE_EXPIRE_MINUTES=30 \
  rvc-onnx-exporter
```

### Health Check

The service provides a `/health` endpoint for health checks:

```bash
# Check health status
curl http://localhost:8000/health
```

### Data Persistence

Persist data using the `DATA_DIR` environment variable or volume mount:

```yaml
# docker-compose.yml
volumes:
  - ./my-data:/app/data  # Persist all data
```

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
