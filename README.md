# RVC ONNX Exporter

## 项目简介

RVC ONNX Exporter 是一个专业工具，用于将 RVC (Retrieval-based Voice Conversion) 模型从 .pth 格式转换为 ONNX 格式，支持 FP16 量化以减小模型体积和提升推理速度。

RVC ONNX Exporter is a professional tool for converting RVC (Retrieval-based Voice Conversion) models from .pth format to ONNX format, with support for FP16 quantization to reduce model size and improve inference speed.

## 核心功能

- **Web 界面**：直观的网页界面，支持拖拽上传和模型信息展示
- **FP16 量化**：将模型量化为 FP16 格式，减小模型体积约 50%
- **自动模型信息探测**：自动读取并展示模型的详细参数信息
- **自动存储空间管理**：定期清理过期文件，避免存储空间占用过大
- **无 UUID 前缀**：下载的 ONNX 文件使用原始文件名，方便用户使用

## Key Features

- **Web Interface**：Intuitive web interface with drag-and-drop upload and model information display
- **FP16 Quantization**：Quantizes models to FP16 format, reducing model size by approximately 50%
- **Automatic Model Info Detection**：Automatically reads and displays detailed model parameter information
- **Automatic Storage Management**：Regularly cleans up expired files to avoid excessive storage usage
- **No UUID Prefix**：Downloaded ONNX files use original filenames for user convenience

## 快速开始

### 环境要求
- Python 3.9+
- PyTorch 2.0+
- ONNX 1.14.0+
- ONNX Converter Common 1.14.0+
- FastAPI
- Uvicorn

### 安装步骤
1. 克隆项目：`git clone https://github.com/yourusername/rvc_onnx_exporter.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 启动服务：`python api.py`

服务将在 http://0.0.0.0:8000 上运行。

## Quick Start

### Requirements
- Python 3.9+
- PyTorch 2.0+
- ONNX 1.14.0+
- ONNX Converter Common 1.14.0+
- FastAPI
- Uvicorn

### Installation
1. Clone the project: `git clone https://github.com/yourusername/rvc_onnx_exporter.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the service: `python api.py`

The service will run on http://0.0.0.0:8000.

## 使用方法

1. 打开浏览器，访问 http://localhost:8000
2. 拖拽 .pth 文件到上传区域，或点击选择文件
3. 系统会自动探测模型信息并显示
4. 选择是否启用 FP16 量化
5. 点击 "开始转换" 按钮
6. 等待转换完成后，点击 "下载 ONNX" 按钮获取转换后的模型

## Usage

1. Open your browser and visit http://localhost:8000
2. Drag and drop .pth files to the upload area, or click to select files
3. The system will automatically detect and display model information
4. Select whether to enable FP16 quantization
5. Click the "开始转换" (Start Conversion) button
6. After conversion is complete, click the "下载 ONNX" (Download ONNX) button to get the converted model

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

### Q: 为什么导出的模型大小与预期不符？
A: 请确保在转换时选择了 FP16 量化选项。FP16 模型大小应约为 FP32 模型的一半。

### Q: 转换过程中出现错误怎么办？
A: 请检查模型文件是否完整，以及是否符合 RVC 模型格式要求。如果问题持续存在，请查看服务日志获取详细错误信息。

## FAQ

### Q: Why is the exported model size not as expected?
A: Please ensure that you have selected the FP16 quantization option during conversion. FP16 models should be approximately half the size of FP32 models.

### Q: What should I do if I encounter errors during conversion?
A: Please check if the model file is complete and conforms to the RVC model format requirements. If the problem persists, please check the service logs for detailed error information.

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.