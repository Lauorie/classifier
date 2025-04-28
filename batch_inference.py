# batch_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import argparse
import os
import json
from tqdm import tqdm # 用于显示进度条

def parse_args():
    parser = argparse.ArgumentParser(description="Perform batch inference on a JSON file and add predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model and tokenizer.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file with predictions.")
    parser.add_argument("--text_field", type=str, default="verify_text", help="The field in the JSON object containing the text to classify.")
    parser.add_argument("--output_field", type=str, default="from_source", help="The field to add to the JSON object with the predicted label.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length used during tokenization.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu'). If None, auto-detect.")
    args = parser.parse_args()
    return args

def batch_predict(model_path, input_json_path, output_json_path, text_field, output_field, max_length, batch_size, device=None):
    """
    Loads a model, processes a JSON file, adds predictions, and saves to a new JSON file.
    """
    # --- 自动检测设备 ---
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"使用设备: {device}")

    # --- 加载模型和分词器 ---
    if not os.path.isdir(model_path):
        logger.error(f"模型路径不存在或不是一个目录: {model_path}")
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    logger.info(f"从以下路径加载模型和分词器: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval() # 设置为评估模式
    except Exception as e:
        logger.error(f"加载模型或分词器失败: {e}")
        raise

    # --- 获取标签映射 ---
    if model.config.id2label:
        id2label = model.config.id2label
        id2label = {int(k): v for k, v in id2label.items()} # 确保 key 是整数
        logger.info(f"从模型配置加载标签映射: {id2label}")
    else:
        logger.warning("在模型配置中未找到 id2label 映射，将使用默认映射。请确保这与训练时一致！")
        # 使用与训练脚本一致的默认映射
        label_dict = {
            'Human': 0, 'GPT': 1, 'Claude': 2, 'Gemini': 3, 'Grok': 4,
            'DeepSeek': 5, 'GLM': 6, 'Qwen': 7, 'Kimi': 8, 'Doubao': 9, 'Ernie': 10
        }
        id2label = {v: k for k, v in label_dict.items()}


    # --- 读取输入 JSON 文件 ---
    logger.info(f"读取输入 JSON 文件: {input_json_path}")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("输入 JSON 文件顶层应为一个列表 (list)。")
            return
    except json.JSONDecodeError as e:
        logger.error(f"解析 JSON 文件失败: {e}")
        return
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        return

    # --- 批量处理和推理 ---
    results = []
    logger.info(f"开始对 {len(data)} 条记录进行推理，批处理大小: {batch_size}...")

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_items = data[i:i+batch_size]
        batch_texts = []
        valid_indices = [] # 记录原始批次中有效项的索引

        # 提取文本并检查有效性
        for idx, item in enumerate(batch_items):
            if isinstance(item, dict) and text_field in item and isinstance(item[text_field], str):
                batch_texts.append(item[text_field])
                valid_indices.append(idx)
            else:
                logger.warning(f"跳过索引 {i+idx} 处的无效或缺失 '{text_field}' 的条目: {item}")
                # 仍然将原始项目添加到结果中，但不添加预测字段
                results.append(item)


        if not batch_texts: # 如果这个批次没有有效的文本
             continue

        # 分词
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)

        # 将输入移动到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        # 将预测结果添加到对应的原始条目中
        pred_idx = 0
        processed_batch_results = []
        original_batch_idx = 0
        while original_batch_idx < len(batch_items):
             item = batch_items[original_batch_idx]
             if original_batch_idx in valid_indices: # 如果这个 item 是有效的并且被预测了
                 predicted_id = predictions[pred_idx].item()
                 predicted_label = id2label.get(predicted_id, "未知标签")
                 item[output_field] = predicted_label
                 processed_batch_results.append(item)
                 pred_idx += 1
             # else: item was already added to results list if invalid
             original_batch_idx +=1

        results.extend(processed_batch_results)


    # --- 写入输出 JSON 文件 ---
    logger.info(f"推理完成。正在将结果写入: {output_json_path}")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4) # indent 用于美化输出
        logger.info("输出文件写入成功。")
    except Exception as e:
        logger.error(f"写入输出 JSON 文件失败: {e}")

if __name__ == "__main__":
    args = parse_args()

    # 配置日志
    log_file_name = f"batch_inference_{os.path.basename(args.input_json)}_{os.path.basename(args.model_path)}.log"
    logger.add(log_file_name)

    batch_predict(
        model_path=args.model_path,
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        text_field=args.text_field,
        output_field=args.output_field,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )