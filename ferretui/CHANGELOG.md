# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2024-11-07]

### Code Structure
- 数据处理模块 (`ferretui/train/data_processing.py`)
  - 重构了预处理函数的实现
    - `preprocess`: 优化了通用数据预处理流程
    - `preprocess_llama3`: 调整了针对 LLaMA3 的特定处理逻辑
  - 改进了 `LazySupervisedDataset` 类的实现
    - 优化了数据加载机制
    - 重构了坐标处理方法
    - 改进了数据预处理流程

### Documentation
- 添加了更详细的中文注释和说明
- 改进了代码的可读性和可维护性
- 统一了文档风格和格式
