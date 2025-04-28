import os
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bge_model_name = "/root/app/models/bge-m3"

class TextClassifier:
    def __init__(self, model_name=bge_model_name, num_labels=11, lr=2e-5):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
    def prepare_data(self, texts, labels, max_length=1024, batch_size=8):
        # 使用batch处理以提高效率
        encoded_data = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(labels)
        
        return TensorDataset(input_ids, attention_masks, labels)
    
    def train(self, train_dataset, val_dataset=None, epochs=4, batch_size=8):
        train_dataloader = DataLoader(
            train_dataset, 
            sampler=RandomSampler(train_dataset), 
            batch_size=batch_size
        )
        
        # 学习率调度器
        total_steps = len(train_dataloader) * epochs
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print('-' * 10)
            
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                self.model.zero_grad()
                outputs = self.model(
                    b_input_ids, 
                    attention_mask=b_input_mask, 
                    labels=b_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 更新参数
                self.optimizer.step()
                
                # 输出进度
                if (step + 1) % 20 == 0:
                    print(f"  Batch {step + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"  Average training loss: {avg_train_loss:.4f}")
            
            # 在每个epoch后评估模型
            if val_dataset:
                val_accuracy = self.evaluate_accuracy(val_dataset, batch_size=batch_size)
                print(f"  Validation Accuracy: {val_accuracy:.4f}")
    
    def evaluate_accuracy(self, dataset, batch_size=8):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                outputs = self.model(
                    b_input_ids,
                    attention_mask=b_input_mask
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                total += b_labels.size(0)
                correct += (predictions == b_labels).sum().item()
        
        return correct / total
    
    def predict(self, text, max_length=1024):
        # 对输入进行编码
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
        
        return predicted_class_id
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

# 加载数据
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_list = []
    for i in data:
        text = i['messages'][1]['content']
        label = i['messages'][-1]['content']
        data_list.append({'text': text, 'models': label})
    
    df = pd.DataFrame(data_list)
    return df

# 主程序
if __name__ == "__main__":
    # 参数设置
    BATCH_SIZE = 8
    MAX_LENGTH = 1024
    EPOCHS = 20
    
    # 加载数据
    json_path = "/root/app/rag_data/paper_rewrite/from_zhaominjie/2015ago/papers_cn_1500_rewrite_all_models_sft_16444.json"
    df = load_data(json_path)
    
    # 标签字典
    label_dict = {
        'Human': 0, 'GPT': 1, 'Claude': 2, 'Gemini': 3, 'Grok': 4,
        'DeepSeek': 5, 'GLM': 6, 'Qwen': 7, 'Kimi': 8, 'Doubao': 9, 'Ernie': 10
    }
    
    # 将标签转换为数字
    df['label_id'] = df['models'].map(label_dict)
    
    # 分割训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label_id'])
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    # 初始化分类器
    classifier = TextClassifier(bge_model_name, num_labels=len(label_dict))
    
    # 准备数据
    train_dataset = classifier.prepare_data(train_df.text.values, train_df.label_id.values, max_length=MAX_LENGTH)
    test_dataset = classifier.prepare_data(test_df.text.values, test_df.label_id.values, max_length=MAX_LENGTH)
    
    # 训练模型
    classifier.train(train_dataset, test_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 评估模型
    test_accuracy = classifier.evaluate_accuracy(test_dataset, batch_size=BATCH_SIZE)
    print(f"最终测试准确率: {test_accuracy:.4f}")
    
    # 保存模型
    classifier.save_model("./fine_tuned_m3_model")
