import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载测试数据
test_df = pd.read_csv('test_data.csv')
print(f"测试集大小: {len(test_df)}")

def predict_with_model(model_path, model_name):
    """使用模型进行预测"""
    print(f"加载{model_name}...")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    predictions = []
    
    for i, text in enumerate(test_df['review_clean']):
        # 编码文本，确保截断到最大长度
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
            
        if i % 500 == 0:
            print(f"{model_name} 处理进度: {i}/{len(test_df)}")
    
    return predictions

# 测试基线模型
print("\n测试基线模型...")
base_preds = predict_with_model('./chinese_bert', '基线模型')

# 测试宏平均最佳模型
print("\n测试宏平均最佳模型...")
macro_preds = predict_with_model('./results/best_f1_macro', '宏平均最佳模型')

# 测试加权平均最佳模型
print("\n测试加权平均最佳模型...")
weighted_preds = predict_with_model('./results/best_f1_weighted', '加权平均最佳模型')

# 计算准确率
base_acc = accuracy_score(test_df['rating'], base_preds)
macro_acc = accuracy_score(test_df['rating'], macro_preds)
weighted_acc = accuracy_score(test_df['rating'], weighted_preds)

# 计算序号
if os.path.exists('model_results.txt'):
    with open('model_results.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        count = sum(1 for line in lines if line.startswith('第') and '次评估' in line)
        eval_number = count + 1
else:
    eval_number = 1

# 保存结果
with open('model_results.txt', 'a', encoding='utf-8') as f:
    f.write(f"第{eval_number}次评估\n")
    f.write(f"测试集大小: {len(test_df)}\n")
    
    f.write("=== 基线模型 ===\n")
    f.write(f"准确率: {base_acc:.4f}\n\n")
    
    f.write("=== 宏平均最佳模型 ===\n")
    f.write(f"准确率: {macro_acc:.4f}\n")
    f.write(f"相比基线 - 准确率提升: {macro_acc - base_acc:.4f}\n\n")
    
    f.write("=== 加权平均最佳模型 ===\n")
    f.write(f"准确率: {weighted_acc:.4f}\n")
    f.write(f"相比基线 - 准确率提升: {weighted_acc - base_acc:.4f}\n\n")
    
    f.write("基线模型分类报告:\n")
    f.write(classification_report(test_df['rating'], base_preds, target_names=['差评', '中评', '好评'], digits=4))
    f.write("\n宏平均最佳模型分类报告:\n")
    f.write(classification_report(test_df['rating'], macro_preds, target_names=['差评', '中评', '好评'], digits=4))
    f.write("\n加权平均最佳模型分类报告:\n")
    f.write(classification_report(test_df['rating'], weighted_preds, target_names=['差评', '中评', '好评'], digits=4))
    f.write("\n" + "=" * 50 + "\n\n")

print("评估完成！结果已追加到 model_results.txt")
print(f"基线模型准确率: {base_acc:.4f}")
print(f"宏平均最佳模型准确率: {macro_acc:.4f}")
print(f"加权平均最佳模型准确率: {weighted_acc:.4f}")
print(f"宏平均模型相比基线提升: {macro_acc - base_acc:.4f}")
print(f"加权平均模型相比基线提升: {weighted_acc - base_acc:.4f}")