# RVC ONNX Exporter - Agent 规范文档

## 项目概述

**RVC ONNX Exporter** 是一个将 RVC (Retrieval-based Voice Conversion) 模型从 `.pth` 格式转换为 ONNX 格式的专业工具。

### 核心功能

- Web 界面：直观的网页界面，支持拖拽上传和模型信息展示
- 自动模型信息探测：自动读取并展示模型的详细参数信息
- 自动存储空间管理：定期清理过期文件，避免存储空间占用过大
- 无 UUID 前缀：下载的 ONNX 文件使用原始文件名，方便用户使用

***

## 项目架构

```
rvc_onnx_exporter/
├── api.py                      # FastAPI 主入口，提供 HTTP API
├── static/
│   └── index.html              # Web 前端界面
├── infer_pack/                 # 模型转换核心模块
│   ├── __init__.py
│   ├── models_onnx.py          # ONNX 模型定义 (SynthesizerTrnMsNSFsidM)
│   ├── attentions.py           # Transformer 注意力机制
│   ├── modules.py              # 神经网络模块 (ResBlock, WN, LayerNorm)
│   ├── commons.py               # 通用工具函数
│   └── transforms.py           # 样条变换实现
└── requirements.txt
```

***

## 核心代码模块详解

### 1. api.py - HTTP API 服务

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\api.py`

**主要功能**:

- 提供 FastAPI Web 服务
- 处理文件上传、模型转换、下载等请求
- 管理转换任务状态
- 定期清理过期文件

**关键类/函数**:

| 名称                                   | 类型             | 说明                   |
| ------------------------------------ | -------------- | -------------------- |
| `ExportTask`                         | Pydantic Model | 转换任务数据模型             |
| `ExportResult`                       | Pydantic Model | 转换结果返回模型             |
| `ModelInfo`                          | Pydantic Model | 模型信息数据模型             |
| `get_model_info(pth_path, filename)` | 函数             | 读取 .pth 文件获取模型参数信息   |
| `export_pth_to_onnx(...)`            | 函数             | 核心转换函数，将 pth 转为 onnx |
| `export_task_wrapper(...)`           | 异步函数           | 异步任务包装器              |
| `cleanup_old_files()`                | 异步函数           | 定期清理过期文件（每6小时）       |

**API 端点**:

| 端点                    | 方法     | 说明             |
| --------------------- | ------ | -------------- |
| `/`                   | GET    | 返回 Web 界面      |
| `/model/info`         | POST   | 获取模型信息         |
| `/export/single`      | POST   | 转换单个模型         |
| `/export/batch`       | POST   | 批量转换模型         |
| `/status/{task_id}`   | GET    | 获取任务状态         |
| `/download/{task_id}` | GET    | 下载转换后的 ONNX 文件 |
| `/tasks`              | GET    | 列出所有任务         |
| `/task/{task_id}`     | DELETE | 删除任务           |

***

### 2. infer\_pack/models\_onnx.py - ONNX 模型定义

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\infer_pack\models_onnx.py`

**主要类**:

| 类名                        | 说明                          |
| ------------------------- | --------------------------- |
| `TextEncoder256`          | 文本编码器 (v1版本，phone\_dim=256) |
| `TextEncoder768`          | 文本编码器 (v2版本，phone\_dim=768) |
| `ResidualCouplingBlock`   | 残差耦合块，用于流模型                 |
| `PosteriorEncoder`        | 后验编码器                       |
| `SineGen`                 | 正弦波生成器                      |
| `SourceModuleHnNSF`       | NSF 声源模块                    |
| `GeneratorNSF`            | NSF 声码器生成器                  |
| `SynthesizerTrnMsNSFsidM` | **主模型类**，整合所有组件             |

**关键方法**:

- `SynthesizerTrnMsNSFsidM.forward(phone, phone_lengths, pitch, nsff0, g, rnd, max_len=None)` - 模型前向传播
- `SynthesizerTrnMsNSFsidM.remove_weight_norm()` - 移除权重归一化

***

### 3. infer\_pack/attentions.py - 注意力机制

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\infer_pack\attentions.py`

**主要类**:

| 类名                   | 说明              |
| -------------------- | --------------- |
| `Encoder`            | Transformer 编码器 |
| `MultiHeadAttention` | 多头注意力机制         |
| `FFN`                | 前馈神经网络          |

**特性**:

- 支持相对位置编码 (relative position encoding)
- 支持局部注意力 (local attention with block\_length)
- 支持近端偏置 (proximal bias)

***

### 4. infer\_pack/modules.py - 神经网络模块

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\infer_pack\modules.py`

**主要类**:

| 类名                      | 说明                 |
| ----------------------- | ------------------ |
| `LayerNorm`             | 层归一化               |
| `WN`                    | WaveNet 风格的权重归一化卷积 |
| `ResBlock1`             | 残差块类型1             |
| `ResBlock2`             | 残差块类型2             |
| `Flip`                  | 张量翻转（用于流的标准化）      |
| `ResidualCouplingLayer` | 残差耦合层              |

***

### 5. infer\_pack/commons.py - 通用工具

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\infer_pack\commons.py`

**函数**:

| 函数名                                    | 说明            |
| -------------------------------------- | ------------- |
| `init_weights(m, mean, std)`           | 权重初始化         |
| `get_padding(kernel_size, dilation)`   | 计算卷积填充        |
| `sequence_mask(length, max_length)`    | 生成序列掩码        |
| `fused_add_tanh_sigmoid_multiply(...)` | JIT 优化的激活函数融合 |

***

### 6. infer\_pack/transforms.py - 样条变换

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\infer_pack\transforms.py`

**函数**:

| 函数名                                            | 说明        |
| ---------------------------------------------- | --------- |
| `piecewise_rational_quadratic_transform(...)`  | 分段有理二次变换  |
| `searchsorted(...)`                            | 二分搜索      |
| `rational_quadratic_spline(...)`               | 有理二次样条    |
| `unconstrained_rational_quadratic_spline(...)` | 无约束有理二次样条 |

***

### 7. static/index.html - Web 前端

**文件路径**: `F:\work\pythonProject\rvc_onnx_exporter\static\index.html`

**功能特性**:

- 拖拽文件上传
- 自动模型信息探测和展示
- 转换任务状态轮询
- 下载和删除任务

***

## 转换流程

### 1. 模型信息获取流程

```
用户上传 .pth 文件 
    ↓
POST /model/info
    ↓
torch.load() 读取 pth 文件
    ↓
解析 config 和 version
    ↓
返回 ModelInfo (版本、通道数、层数等)
```

### 2. ONNX 转换流程

```
用户请求转换
    ↓
POST /export/single
    ↓
创建 ExportTask，状态为 "pending"
    ↓
后台任务调用 export_pth_to_onnx()
    ↓
1. 加载 pth 权重
2. 构建 SynthesizerTrnMsNSFsidM 模型
3. 准备测试输入 (phone, pitch, nsff0 等)
4. torch.onnx.export() 导出
5. (可选) FP16 量化转换
    ↓
保存 ONNX 文件到 exported_onnx/
    ↓
更新任务状态为 "completed"
    ↓
用户通过 /download/{task_id} 下载
```

***

## 模型参数说明

RVC 模型的 `config` 数组包含以下参数（索引从0开始）:

| 索引  | 参数名              | 说明             |
| --- | ---------------- | -------------- |
| 0   | -                | (历史遗留)         |
| 1   | -                | (历史遗留)         |
| 2   | inter\_channels  | 中间通道数          |
| 3   | hidden\_channels | 隐藏通道数          |
| 4   | filter\_channels | 过滤器通道数         |
| 5   | n\_heads         | 注意力头数          |
| 6   | n\_layers        | Transformer 层数 |
| 7   | kernel\_size     | 卷积核大小          |
| ... | ...              | ...            |
| 15  | spk\_embed\_dim  | 说话人嵌入维度        |
| 16  | gin\_channels    | 说话人条件通道数       |
| 17  | sample\_rate     | 采样率            |

***

## 运行方式

### 1. 直接运行

```bash
pip install -r requirements.txt
python api.py
```

服务将在 <http://0.0.0.0:8000> 运行

### 2. Docker 运行

```bash
docker-compose up -d
```

***

## 依赖项

- Python 3.9+
- PyTorch 2.0+
- ONNX 1.14.0+
- onnxconverter-common (FP16转换)
- FastAPI
- Uvicorn
- pydantic

***

## 注意事项

1. **版本兼容性**: 支持 RVC v1 和 v2 版本的模型
2. **ONNX 导出**: 导出时使用 FP32，然后可选择转换为 FP16
3. **动态轴**: phone, pitch, pitchf, rnd 等张量支持动态长度
4. **存储清理**: 上传和导出的文件会在1小时后自动清理
5. **任务清理**: 已完成的任务在文件被删除后1小时也会从内存中清理

