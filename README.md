# BGE-M3 文本分类器微调与推理

## 简介

本项目旨在使用 Hugging Face `transformers` 库微调 BGE-M3 (或其他兼容的预训练语言模型) 以执行文本分类任务。训练过程利用 `Trainer` API 实现，同时提供了推理脚本来使用微调后的模型进行预测。

## 文件结构

```
finetune_bert/
├── m3_trainner.py         # 主要的训练脚本
├── m3_trainner.sh         # 运行训练的 Shell 脚本示例
├── inference.py           # 推理脚本
├── inference.sh           # 运行推理的 Shell 脚本示例
├── requirements.txt       # 项目依赖
└── README.md              # 本文档
```

## 环境设置

1.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # 或者
    # .venv\Scripts\activate  # Windows
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *注意：* 根据您的硬件情况，可能需要安装特定版本的 `torch` (例如，带 CUDA 支持的版本)。请参考 PyTorch 官方文档。

## 数据准备

训练脚本 `m3_trainner.py` 需要一个 JSON 文件作为输入 (`--data_path` 参数)。该 JSON 文件应包含一个列表，列表中的每个元素是一个字典，代表一个训练样本。

每个样本字典需要包含一个 `messages` 键，其值是一个至少包含两个元素的列表：
*   `messages[1]['content']`: 包含要分类的文本。
*   `messages[-1]['content']`: 包含该文本对应的标签名称 (例如 "Human", "GPT" 等)。

脚本内部会根据 `label_dict` 将标签名称映射为整数 ID。

**示例 JSON 结构:**
```json
[
  {
    "messages": [
      {"role": "user", "content": "一些指令或上下文"},
      {"role": "assistant", "content": "这是需要分类的文本内容..."},
      {"role": "tool", "content": "GPT"} // 标签名称
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "另一个指令"},
      {"role": "assistant", "content": "另一段需要分类的文本..."},
      {"role": "tool", "content": "Human"} // 标签名称
    ]
  }
  // ... 更多样本
]
```

## 模型训练

1.  **配置训练参数:**
    打开 `m3_trainner.sh` 文件。根据您的需求修改以下关键参数：
    *   `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU ID。
    *   `--model_name_or_path`: 预训练模型的路径或 Hugging Face Hub 上的名称。
    *   `--data_path`: 训练数据 JSON 文件的路径。
    *   `--output_dir`: 保存微调模型、检查点和日志的目录。
    *   `--num_train_epochs`: 训练轮数。
    *   `--per_device_train_batch_size`: 训练批次大小。
    *   `--learning_rate`: 学习率。
    *   `--max_length`: 输入序列的最大长度。
    *   `--logging_steps`, `--eval_steps`, `--save_steps`: 日志、评估和保存的频率。
    *   `--fp16`: 是否启用混合精度训练 (需要兼容的 GPU)。
    *   `--gradient_accumulation_steps`: 梯度累积步数。

2.  **启动训练:**
    ```bash
    bash m3_trainner.sh
    ```
    训练将在后台运行，日志输出到 `m3_trainner.sh` 中重定向的文件 (例如 `conan_trainner.log`)。训练好的模型和检查点将保存在 `--output_dir` 指定的目录下。

## 模型推理

1.  **配置推理参数:**
    打开 `inference.sh` 文件。修改以下参数：
    *   `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU ID。
    *   `--model_path`: 微调后模型 checkpoint 的路径 (通常是 `--output_dir` 下的某个 `checkpoint-xxxx` 目录)。
    *   `--max_length`: 与训练时使用的最大长度保持一致或根据需要调整。
    *   `--device`: 使用 `cuda` 或 `cpu`。
    *   `--temperature`: 温度系数，用于调整输出概率分布的平滑度 (默认为 1.0)。大于 1 使分布更平滑，小于 1 使分布更尖锐。
    *   `--text`: 要进行分类的输入文本。

2.  **执行推理:**
    ```bash
    bash inference.sh
    ```
    脚本将输出：
    *   输入的文本。
    *   使用的温度系数。
    *   预测的最可能的标签及其对应的 ID。
    *   所有标签及其对应的概率 (经过温度系数调整)。

## 自定义

*   **更换基础模型:** 修改 `m3_trainner.sh` 和 `m3_trainner.py` 中的 `--model_name_or_path` 参数。确保新模型与 `AutoModelForSequenceClassification` 兼容。
*   **调整超参数:** 直接修改 `m3_trainner.sh` 中的命令行参数或 `m3_trainner.py` 中的 `argparse` 默认值。
*   **使用不同数据集:** 准备符合格式要求的 JSON 数据，并更新 `m3_trainner.sh` 中的 `--data_path` 参数。如果标签集不同，需要修改 `m3_trainner.py` 中的 `label_dict`。
*   **修改数据处理:** 如果您的数据格式不同，需要调整 `m3_trainner.py` 中的 `load_and_preprocess_data` 函数。 