# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
# )
# from loguru import logger
# import argparse
# import os

# def parse_args():
#     parser = argparse.ArgumentParser(description="Perform inference with a fine-tuned sequence classification model.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model and tokenizer (output_dir from training).")
#     parser.add_argument("--text", type=str, required=True, help="The input text to classify.")
#     parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length used during tokenization.")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu'). If None, auto-detect.")
#     args = parser.parse_args()
#     return args

# def predict(model_path, text, max_length, device=None):
#     """
#     Loads a fine-tuned model and tokenizer, and predicts the class for the given text.

#     Args:
#         model_path (str): Path to the saved model directory.
#         text (str): Input text for classification.
#         max_length (int): Maximum sequence length for the tokenizer.
#         device (str, optional): 'cuda' or 'cpu'. Defaults to auto-detection.

#     Returns:
#         str: The predicted label name.
#         int: The predicted label ID.
#     """
#     # --- 自动检测设备 ---
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)
#     logger.info(f"使用设备: {device}")

#     # --- 加载模型和分词器 ---
#     if not os.path.isdir(model_path):
#         logger.error(f"模型路径不存在或不是一个目录: {model_path}")
#         raise FileNotFoundError(f"Model directory not found: {model_path}")

#     logger.info(f"从以下路径加载模型和分词器: {model_path}")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         model.to(device)
#         model.eval() # 设置为评估模式
#     except Exception as e:
#         logger.error(f"加载模型或分词器失败: {e}")
#         raise

#     # --- 获取标签映射 ---
#     # Trainer 保存的模型配置中通常包含 id2label 和 label2id
#     if model.config.id2label:
#         id2label = model.config.id2label
#          # id2label 的 key 可能是字符串形式的整数，需要转换
#         id2label = {int(k): v for k, v in id2label.items()}
#         logger.info(f"从模型配置加载标签映射: {id2label}")
#     else:
#         # 如果模型配置中没有，需要手动提供（确保与训练时一致）
#         logger.warning("在模型配置中未找到 id2label 映射，将使用默认映射。请确保这与训练时一致！")
#         label_dict = {
#             'Human': 0, 'GPT': 1, 'Claude': 2, 'Gemini': 3, 'Grok': 4,
#             'DeepSeek': 5, 'GLM': 6, 'Qwen': 7, 'Kimi': 8, 'Doubao': 9, 'Ernie': 10
#         }
#         id2label = {v: k for k, v in label_dict.items()}


#     # --- 分词和推理 ---
#     logger.info("进行分词...")
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)

#     # 将输入移动到正确的设备
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     logger.info("开始推理...")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     # --- 获取预测结果 ---
#     predicted_class_id = logits.argmax().item()
#     predicted_label = id2label.get(predicted_class_id, "未知标签") # 使用 get 以防 ID 不在映射中

#     logger.info("推理完成。")
#     return predicted_label, predicted_class_id

# if __name__ == "__main__":
#     args = parse_args()

#     logger.info("开始推理...")
#     predicted_label, predicted_id = predict(args.model_path, args.text, args.max_length, args.device)

#     print("-" * 30)
#     print(f"输入文本: '{args.text}'")
#     print(f"预测标签: {predicted_label} (ID: {predicted_id})")
#     print("-" * 30)


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from loguru import logger
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference with a fine-tuned sequence classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model and tokenizer (output_dir from training).")
    parser.add_argument("--text", type=str, required=True, help="The input text to classify.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length used during tokenization.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu'). If None, auto-detect.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for scaling logits before softmax (controls randomness). Must be positive.")
    args = parser.parse_args()
    return args

def predict(model_path, text, max_length, temperature=1.0, device=None):
    """
    Loads a fine-tuned model and tokenizer, predicts the class for the given text,
    and returns the probabilities for all classes adjusted by temperature.

    Args:
        model_path (str): Path to the saved model directory.
        text (str): Input text for classification.
        max_length (int): Maximum sequence length for the tokenizer.
        temperature (float): Temperature for scaling logits. Must be positive. Defaults to 1.0.
        device (str, optional): 'cuda' or 'cpu'. Defaults to auto-detection.

    Returns:
        str: The predicted label name (based on highest probability after temperature scaling).
        int: The predicted label ID.
        dict: A dictionary mapping label names to their (temperature-scaled) probabilities.
    """
    # --- 检查 Temperature ---
    if temperature <= 0:
        logger.warning(f"Temperature必须为正数，但收到了 {temperature}。将使用默认值 1.0。")
        temperature = 1.0

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
    # Trainer 保存的模型配置中通常包含 id2label 和 label2id
    if model.config.id2label:
        id2label = model.config.id2label
         # id2label 的 key 可能是字符串形式的整数，需要转换
        id2label = {int(k): v for k, v in id2label.items()}
        logger.info(f"从模型配置加载标签映射: {id2label}")
    else:
        # 如果模型配置中没有，需要手动提供（确保与训练时一致）
        logger.warning("在模型配置中未找到 id2label 映射，将使用默认映射。请确保这与训练时一致！")
        # Fallback, though ideally this should match training
        num_labels = model.config.num_labels
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}


    # --- 分词和推理 ---
    logger.info("进行分词...")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)

    # 将输入移动到正确的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.info("开始推理...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # --- 应用 Temperature 并计算概率 ---
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)[0] # 获取第一个（也是唯一一个）输入的概率
    probabilities_cpu = probabilities.cpu().tolist() # 移至 CPU 并转换为列表

    # --- 获取预测结果 (基于调整后的概率) ---
    predicted_class_id = probabilities.argmax().item()
    predicted_label = id2label.get(predicted_class_id, "未知标签") # 使用 get 以防 ID 不在映射中

    # --- 创建标签到概率的映射 ---
    label_probabilities = {
        id2label.get(i, f"未知标签_{i}"): prob
        for i, prob in enumerate(probabilities_cpu)
    }


    logger.info("推理完成。")
    return predicted_label, predicted_class_id, label_probabilities

if __name__ == "__main__":
    args = parse_args()

    logger.info("开始推理...")
    predicted_label, predicted_id, label_probabilities = predict(
        args.model_path,
        args.text,
        args.max_length,
        temperature=args.temperature, # 传递 temperature
        device=args.device
    )

    print("-" * 30)
    print(f"输入文本: '{args.text}'")
    print(f"Temperature: {args.temperature}")
    print("-" * 30)
    print(f"预测标签: {predicted_label} (ID: {predicted_id})")
    print("-" * 30)
    print(f"各标签概率 (temperature = {args.temperature}):")
    # 对概率进行排序（可选，但更清晰）
    sorted_probs = sorted(label_probabilities.items(), key=lambda item: item[1], reverse=True)
    for label, prob in sorted_probs:
        print(f"  - {label:<10}: {prob:.4f}") # 格式化输出
    print("-" * 30)

