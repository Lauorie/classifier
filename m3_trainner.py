import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
import numpy as np
from loguru import logger
import argparse

# --- 配置与参数 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a sequence classification model using Hugging Face Trainer.")
    parser.add_argument("--model_name_or_path", type=str, default="/root/app/models/bge-m3", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_path", type=str, default="/root/app/rag_data/paper_rewrite/from_zhaominjie/2015ago/papers_cn_1500_rewrite_all_models_sft_16444.json", help="Path to the JSON data file.")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_m3_model_trainer", help="Directory to save the fine-tuned model and training logs.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs.") # 减少默认 epochs 数量，原先20可能过长
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation.") # 评估时 batch size 可以稍大
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.") # 减少 max_length，1024 可能对显存压力大，根据需要调整
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset to use for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log training info every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate model every X updates steps during training.") # 根据数据量和epoch调整评估频率
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.") # 根据需要调整保存频率
    parser.add_argument("--fp16", action='store_true', help="Enable mixed precision training (requires compatible GPU).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply.")


    args = parser.parse_args()
    return args

# --- 数据处理 ---
def load_and_preprocess_data(data_path, tokenizer, max_length, label_dict, test_size, seed):
    logger.info(f"加载数据从: {data_path}")
    import json
    with open(data_path, 'r') as f:
        data_list = json.load(f)
    # 转换为 datasets 需要的格式
    processed_data = {'text': [], 'label': []}
    for item in data_list:
        try:
            processed_data['text'].append(item['messages'][1]['content'])
            label_name = item['messages'][-1]['content']
            if label_name in label_dict:
                processed_data['label'].append(label_dict[label_name])
            else:
                logger.warning(f"在 label_dict 中未找到标签: {label_name}，跳过该样本。")
        except (IndexError, KeyError) as err:
            logger.warning(f"处理数据项时出错: {item}, 错误: {err}, 跳过该样本。")

    from datasets import Dataset
    raw_datasets = Dataset.from_dict(processed_data).train_test_split(test_size=test_size, seed=seed)
    logger.info("成功将 JSON 列表转换为 DatasetDict。")

    # 标签名称到 ID 的映射
    id2label = {v: k for k, v in label_dict.items()}
    label2id = label_dict

    def tokenize_function(examples):
        # 确保 'text' 列存在
        if 'text' not in examples:
            raise ValueError("数据集中未找到 'text' 列。请检查 JSON 结构或加载逻辑。")
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False) # Trainer 会处理 padding

    logger.info("开始分词...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    logger.info("分词完成。")

    # 移除不再需要的原始文本列，并重命名标签列以匹配模型期望
    if 'label' in tokenized_datasets['train'].column_names and 'labels' not in tokenized_datasets['train'].column_names:
         tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    if 'text' in tokenized_datasets['train'].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])


    logger.info(f"处理后的数据集结构: {tokenized_datasets}")
    logger.info(f"训练集大小: {len(tokenized_datasets['train'])}, 测试集大小: {len(tokenized_datasets['test'])}")

    return tokenized_datasets, label2id, id2label

# --- 评估指标 ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions 可能是 logits，需要 argmax
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# --- 主程序 ---
if __name__ == "__main__":
    args = parse_args()

    # 设置日志
    logger.add("training_{time}.log")
    logger.info("开始训练脚本...")
    logger.info(f"参数: {args}")

    # 检查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    if args.fp16 and not torch.cuda.is_available():
        logger.warning("指定了 fp16 但 CUDA 不可用，将禁用 fp16。")
        args.fp16 = False

    # 标签字典 (保持与原代码一致)
    label_dict = {
        'Human': 0, 'GPT': 1, 'Claude': 2, 'Gemini': 3, 'Grok': 4,
        'DeepSeek': 5, 'GLM': 6, 'Qwen': 7, 'Kimi': 8, 'Doubao': 9, 'Ernie': 10
    }
    num_labels = len(label_dict)

    # 加载 Tokenizer 和 Model
    logger.info(f"加载模型和分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        # 如果需要，可以提供 id2label 和 label2id 以获取更友好的输出
        id2label={v: k for k, v in label_dict.items()},
        label2id=label_dict
    )
    # model.to(device) # Trainer 会自动处理设备放置

    # 加载和预处理数据
    tokenized_datasets, label2id, id2label = load_and_preprocess_data(
        args.data_path, tokenizer, args.max_length, label_dict, args.test_size, args.seed
    )

    # 数据整理器 (自动处理批次内的填充)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 配置训练参数
    logger.info("配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio, # 使用预热比例
        weight_decay=args.weight_decay, # 添加权重衰减
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps", # 在训练过程中进行评估
        eval_steps=args.eval_steps,
        save_strategy="steps", # 按步数保存模型
        save_steps=args.save_steps,
        save_total_limit=3, # 最多保存最近的2个 checkpoints
        load_best_model_at_end=True, # 训练结束后加载最佳模型
        save_only_model=True,
        metric_for_best_model="accuracy", # 使用准确率选择最佳模型
        greater_is_better=True, # 准确率越大越好
        fp16=args.fp16, # 混合精度训练
        gradient_accumulation_steps=args.gradient_accumulation_steps, # 梯度累积
        seed=args.seed,
        report_to="tensorboard", # 可以配置 wandb, tensorboard 等
        # remove_unused_columns=False # 如果数据集包含模型不需要的列，设为 False
    )

    # 初始化 Trainer
    logger.info("初始化 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], # 使用测试集作为评估集
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    logger.info("开始训练...")
    train_result = trainer.train()
    logger.info("训练完成。")

    # 保存训练指标和状态
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 在测试集上进行最终评估
    logger.info("开始最终评估...")
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    logger.info(f"最终评估结果: {eval_results}")
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)


    # 保存最终模型 (Trainer 在 load_best_model_at_end=True 时会自动保存最佳模型)
    # 如果需要显式保存最终状态的模型
    # trainer.save_model(args.output_dir)
    logger.info(f"最佳模型已保存在: {args.output_dir}")

    # 进行预测 (示例)
    logger.info("进行单个预测示例...")
    text_to_predict = "我的职业生涯也经历了很多变化，从Tech lead到技术经理，再到技术总监；从推荐系统到广告系统；从中国的互联网公司和美国的互联网公司之间切换。这期间，我对技术和行业的理解也有所不同。五年前，我在「深度学习推荐系统」的最后一页说，“这不是结束，而是另一个开始。在不远的将来，笔者会持续更新书中的内容，让本书的知识体系持续枝繁叶茂”。"
    inputs = tokenizer(text_to_predict, return_tensors="pt", truncation=True, max_length=args.max_length).to(device) # 确保输入在正确的设备上
    model.eval() # 切换到评估模式
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_label = id2label[predicted_class_id]
    logger.info(f"预测文本: '{text_to_predict}'")
    logger.info(f"预测标签 ID: {predicted_class_id}, 预测标签: {predicted_label}")
