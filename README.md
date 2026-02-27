# RVC ONNX Exporter

## 项目简介

RVC ONNX Exporter 是一个专业的工具，用于将 RVC (Retrieval-based Voice Conversion) 模型从 .pth 格式转换为 ONNX 格式。它提供了一个直观的 Web 界面，使用户能够轻松地进行模型转换，同时支持 FP16 量化以减小模型体积和提升推理速度。

该工具旨在解决 RVC 模型在不同平台上的部署问题，通过将模型转换为 ONNX 格式，使其能够在各种支持 ONNX Runtime 的环境中运行，包括移动设备、Web 浏览器和边缘设备。

RVC ONNX Exporter is a professional tool for converting RVC (Retrieval-based Voice Conversion) models from .pth format to ONNX format. It provides an intuitive web interface that allows users to easily perform model conversion, while supporting FP16 quantization to reduce model size and improve inference speed.

This tool aims to solve the deployment issues of RVC models across different platforms. By converting models to ONNX format, they can run in various environments supporting ONNX Runtime, including mobile devices, web browsers, and edge devices.

## 技术原理

RVC ONNX Exporter 基于以下核心技术：

1. **模型加载与分析**：使用 PyTorch 加载 RVC 模型，自动分析模型结构和参数，确定模型版本和隐藏层尺寸。
2. **模型转换**：使用 PyTorch 的 ONNX 导出功能，将模型转换为 ONNX 格式。
3. **FP16 量化**：使用 ONNX Converter Common 库将模型量化为 FP16 格式，减小模型体积和提升推理速度。
4. **Web 界面**：使用 FastAPI 和 HTML/CSS/JavaScript 构建直观的 Web 界面，支持拖拽上传和模型信息展示。
5. **存储空间管理**：实现自动存储空间管理，定期清理过期文件，避免存储空间占用过大。

## Technical Principles

RVC ONNX Exporter is based on the following core technologies:

1. **Model Loading and Analysis**: Uses PyTorch to load RVC models, automatically analyzes model structure and parameters, and determines model version and hidden layer size.
2. **Model Conversion**: Uses PyTorch's ONNX export functionality to convert models to ONNX format.
3. **FP16 Quantization**: Uses the ONNX Converter Common library to quantize models to FP16 format, reducing model size and improving inference speed.
4. **Web Interface**: Builds an intuitive web interface using FastAPI and HTML/CSS/JavaScript, supporting drag-and-drop upload and model information display.
5. **Storage Management**: Implements automatic storage management, regularly cleaning up expired files to avoid excessive storage usage.

## 功能特点

- **Web 界面**：提供直观的网页界面，支持拖拽上传和模型信息展示
- **FP16 量化**：支持将模型量化为 FP16 格式，减小模型体积约 50%
- **自动模型信息探测**：自动读取并展示模型的详细参数信息，包括模型版本、隐藏层尺寸等
- **自动存储空间管理**：定期清理过期文件，避免存储空间占用过大
- **无 UUID 前缀**：下载的 ONNX 文件使用原始文件名，无 UUID 前缀，方便用户使用
- **批量转换**：支持同时转换多个模型文件，提高效率

## Features

- **Web Interface**: Provides an intuitive web interface with drag-and-drop upload and model information display
- **FP16 Quantization**: Supports quantizing models to FP16 format, reducing model size by approximately 50%
- **Automatic Model Info Detection**: Automatically reads and displays detailed model parameter information, including model version and hidden layer size
- **Automatic Storage Management**: Regularly cleans up expired files to avoid excessive storage usage
- **No UUID Prefix**: Downloaded ONNX files use original filenames without UUID prefixes for user convenience
- **Batch Conversion**: Supports converting multiple model files simultaneously to improve efficiency

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- ONNX 1.14.0+
- ONNX Converter Common 1.14.0+
- FastAPI
- Uvicorn

## Requirements

- Python 3.9+
- PyTorch 2.0+
- ONNX 1.14.0+
- ONNX Converter Common 1.14.0+
- FastAPI
- Uvicorn

## 安装步骤

1. 克隆项目到本地：

```bash
git clone https://github.com/yourusername/rvc_onnx_exporter.git
cd rvc_onnx_exporter
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 启动服务：

```bash
python api.py
```

服务将在 http://0.0.0.0:8000 上运行。

## Installation

1. Clone the project to your local machine:

```bash
git clone https://github.com/yourusername/rvc_onnx_exporter.git
cd rvc_onnx_exporter
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the service:

```bash
python api.py
```

The service will run on http://0.0.0.0:8000.

## 使用方法

### 基本使用

1. 打开浏览器，访问 http://localhost:8000
2. 拖拽 .pth 文件到上传区域，或点击选择文件
3. 系统会自动探测模型信息并显示，包括模型版本、隐藏层尺寸等
4. 选择是否启用 FP16 量化
5. 点击 "开始转换" 按钮
6. 等待转换完成后，点击 "下载 ONNX" 按钮获取转换后的模型

### 批量转换

1. 在上传区域同时选择多个 .pth 文件
2. 系统会为每个文件创建单独的转换任务
3. 可以在任务列表中查看每个任务的状态
4. 转换完成后，分别下载每个转换后的模型

## Usage

### Basic Usage

1. Open your browser and visit http://localhost:8000
2. Drag and drop .pth files to the upload area, or click to select files
3. The system will automatically detect and display model information, including model version and hidden layer size
4. Select whether to enable FP16 quantization
5. Click the "开始转换" (Start Conversion) button
6. After conversion is complete, click the "下载 ONNX" (Download ONNX) button to get the converted model

### Batch Conversion

1. Select multiple .pth files simultaneously in the upload area
2. The system will create separate conversion tasks for each file
3. You can view the status of each task in the task list
4. After conversion is complete, download each converted model separately

## API 文档

服务启动后，可以访问 http://localhost:8000/docs 查看详细的 API 文档。

### 主要 API 端点：

- `GET /`：访问 Web 界面
- `POST /model/info`：获取模型信息，返回模型的详细参数
- `POST /export/single`：转换单个模型，支持 FP16 量化
- `POST /export/batch`：批量转换模型，支持 FP16 量化
- `GET /status/{task_id}`：获取转换任务状态
- `GET /download/{task_id}`：下载转换后的模型，使用原始文件名
- `GET /tasks`：列出所有转换任务
- `DELETE /task/{task_id}`：删除转换任务

### 请求和响应格式

#### 获取模型信息

**请求**：
```json
{
  "file": "base64 encoded file content"
}
```

**响应**：
```json
{
  "success": true,
  "info": {
    "version": "v2",
    "hidden_size": 768,
    "n_enc_layers": 12,
    "n_dec_layers": 12,
    "n_heads": 12,
    "window_size": 10,
    "mel_channels": 80
  }
}
```

#### 转换单个模型

**请求**：
```json
{
  "file": "base64 encoded file content",
  "fp16": true
}
```

**响应**：
```json
{
  "success": true,
  "task_id": "uuid"
}
```

## API Documentation

After starting the service, you can visit http://localhost:8000/docs to view the detailed API documentation.

### Main API endpoints:

- `GET /`：Access the web interface
- `POST /model/info`：Get model information, returns detailed model parameters
- `POST /export/single`：Convert a single model, supports FP16 quantization
- `POST /export/batch`：Batch convert models, supports FP16 quantization
- `GET /status/{task_id}`：Get conversion task status
- `GET /download/{task_id}`：Download the converted model, using the original filename
- `GET /tasks`：List all conversion tasks
- `DELETE /task/{task_id}`：Delete a conversion task

### Request and Response Formats

#### Get Model Information

**Request**:
```json
{
  "file": "base64 encoded file content"
}
```

**Response**:
```json
{
  "success": true,
  "info": {
    "version": "v2",
    "hidden_size": 768,
    "n_enc_layers": 12,
    "n_dec_layers": 12,
    "n_heads": 12,
    "window_size": 10,
    "mel_channels": 80
  }
}
```

#### Convert a Single Model

**Request**:
```json
{
  "file": "base64 encoded file content",
  "fp16": true
}
```

**Response**:
```json
{
  "success": true,
  "task_id": "uuid"
}
```

## 故障排除

### 常见问题

1. **Q: 为什么导出的模型大小与预期不符？**
   **A:** 请确保在转换时选择了 FP16 量化选项。FP16 模型大小应约为 FP32 模型的一半。如果仍然有问题，请检查模型文件是否完整。

2. **Q: 转换过程中出现错误怎么办？**
   **A:** 请检查模型文件是否完整，以及是否符合 RVC 模型格式要求。如果问题持续存在，请查看服务日志获取详细错误信息。常见错误包括模型文件损坏、模型版本不兼容等。

3. **Q: 下载的文件名为什么没有 UUID 前缀？**
   **A:** 为了方便用户使用，我们移除了 UUID 前缀，直接使用原始模型文件名。

4. **Q: 转换后的模型在 ONNX Runtime 中运行时出现错误怎么办？**
   **A:** 请检查 ONNX Runtime 的版本是否与模型兼容。我们推荐使用 ONNX Runtime 1.14.0 或更高版本。如果问题持续存在，请查看服务日志获取详细错误信息。

### 日志查看

服务日志会输出到控制台，您可以通过以下方式查看详细日志：

```bash
python api.py > rvc_onnx_exporter.log 2>&1
```

然后使用文本编辑器查看 `rvc_onnx_exporter.log` 文件。

## Troubleshooting

### Common Issues

1. **Q: Why is the exported model size not as expected?**
   **A:** Please ensure that you have selected the FP16 quantization option during conversion. FP16 models should be approximately half the size of FP32 models. If the issue persists, please check if the model file is complete.

2. **Q: What should I do if I encounter errors during conversion?**
   **A:** Please check if the model file is complete and conforms to the RVC model format requirements. If the problem persists, please check the service logs for detailed error information. Common errors include corrupted model files and incompatible model versions.

3. **Q: Why is there no UUID prefix in the downloaded filename?**
   **A:** To facilitate user use, we have removed the UUID prefix and directly use the original model filename.

4. **Q: What should I do if the converted model encounters errors when running in ONNX Runtime?**
   **A:** Please check if the ONNX Runtime version is compatible with the model. We recommend using ONNX Runtime 1.14.0 or higher. If the issue persists, please check the service logs for detailed error information.

### Log Viewing

Service logs are output to the console, and you can view detailed logs in the following way:

```bash
python api.py > rvc_onnx_exporter.log 2>&1
```

Then use a text editor to view the `rvc_onnx_exporter.log` file.

## 存储空间管理

系统会自动管理存储空间，定期清理过期文件：

- **上传的 .pth 文件**：保存 1 小时
- **导出的 .onnx 文件**：保存 1 小时
- **完成的转换任务**：保存 1 小时

清理任务每 6 小时执行一次，确保系统不会因为过多的临时文件而占用过多存储空间。

## Storage Management

The system automatically manages storage space and regularly cleans up expired files:

- **Uploaded .pth files**: Saved for 1 hour
- **Exported .onnx files**: Saved for 1 hour
- **Completed conversion tasks**: Saved for 1 hour

Cleanup tasks are executed every 6 hours to ensure the system does not occupy excessive storage space due to too many temporary files.

## 技术细节

### 模型转换流程

1. **模型加载**：使用 PyTorch 加载 .pth 文件，提取模型参数。
2. **模型分析**：自动分析模型结构，确定模型版本和隐藏层尺寸。
3. **模型初始化**：根据分析结果初始化模型。
4. **模型导出**：使用 PyTorch 的 ONNX 导出功能，将模型转换为 ONNX 格式。
5. **FP16 量化**：如果启用了 FP16 量化，使用 ONNX Converter Common 库将模型量化为 FP16 格式。
6. **文件保存**：将转换后的模型保存到指定目录。

### 支持的模型版本

- **RVC v1**：隐藏层尺寸为 256
- **RVC v2**：隐藏层尺寸为 768

### 性能优化

- **异步处理**：使用 FastAPI 的异步功能，提高并发处理能力。
- **批处理**：支持批量转换多个模型，提高效率。
- **内存管理**：优化内存使用，避免内存泄漏。

## Technical Details

### Model Conversion Process

1. **Model Loading**: Uses PyTorch to load .pth files and extract model parameters.
2. **Model Analysis**: Automatically analyzes model structure to determine model version and hidden layer size.
3. **Model Initialization**: Initializes the model based on the analysis results.
4. **Model Export**: Uses PyTorch's ONNX export functionality to convert the model to ONNX format.
5. **FP16 Quantization**: If FP16 quantization is enabled, uses the ONNX Converter Common library to quantize the model to FP16 format.
6. **File Saving**: Saves the converted model to the specified directory.

### Supported Model Versions

- **RVC v1**: Hidden layer size of 256
- **RVC v2**: Hidden layer size of 768

### Performance Optimization

- **Asynchronous Processing**: Uses FastAPI's asynchronous features to improve concurrent processing capability.
- **Batch Processing**: Supports batch conversion of multiple models to improve efficiency.
- **Memory Management**: Optimizes memory usage to avoid memory leaks.

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目。在提交之前，请确保：

1. 您的代码符合项目的代码风格和质量要求。
2. 您的修改不会破坏现有的功能。
3. 您已经测试了您的修改，确保其正常工作。

## Contributing

Issues and pull requests are welcome to improve this project. Before submitting, please ensure:

1. Your code follows the project's code style and quality requirements.
2. Your changes do not break existing functionality.
3. You have tested your changes to ensure they work correctly.

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
