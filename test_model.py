<<<<<<< HEAD
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
=======
# test_model.py
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import torch
>>>>>>> 155009d (🎉 init:项目初版)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载测试数据
test_df = pd.read_csv('test_data.csv')
print(f"测试集大小: {len(test_df)}")

<<<<<<< HEAD
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
=======
# 1. 测试基线模型（使用置信度划分三分类）
print("加载基线模型...")
base_tokenizer = AutoTokenizer.from_pretrained('./chinese_bert')
base_model = AutoModelForSequenceClassification.from_pretrained('./chinese_bert')
base_model.to(device)

def score_to_label(score):
    """将置信度分数转换为三分类标签"""
    if score < 0.33:
        return 0  # 差评
    elif score < 0.66:
        return 1  # 中评
    else:
        return 2  # 好评

def predict_with_model(model, tokenizer, texts):
    """使用模型进行预测，处理长文本截断"""
    predictions = []
    for text in texts:
        # 对文本进行编码，确保截断到最大长度
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,  # 与训练时保持一致
>>>>>>> 155009d (🎉 init:项目初版)
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
<<<<<<< HEAD
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
=======
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            predictions.append(pred)
    
    return predictions

print("基线模型预测中...")
base_preds = predict_with_model(base_model, base_tokenizer, test_df['review_clean'].tolist())

# 2. 测试微调模型（直接使用预测标签）
print("加载微调模型...")
fine_tokenizer = AutoTokenizer.from_pretrained('./trained_model')
fine_model = AutoModelForSequenceClassification.from_pretrained('./trained_model')
fine_model.to(device)

print("微调模型预测中...")
fine_preds = predict_with_model(fine_model, fine_tokenizer, test_df['review_clean'].tolist())

# 3. 计算准确率
base_acc = accuracy_score(test_df['rating'], base_preds)
fine_acc = accuracy_score(test_df['rating'], fine_preds)

# 4. 保存评估结果
with open('model_results.txt', 'w', encoding='utf-8') as f:
    f.write("商品评价情感分析模型评估结果\n")
    f.write("=" * 50 + "\n")
    f.write(f"测试集大小: {len(test_df)}\n")
    f.write(f"基线模型准确率: {base_acc:.4f}\n")
    f.write(f"微调模型准确率: {fine_acc:.4f}\n")
    f.write(f"准确率提升: {fine_acc - base_acc:.4f}\n\n")

    f.write("基线模型分类报告:\n")
    f.write(classification_report(test_df['rating'], base_preds, target_names=['差评', '中评', '好评'], digits=4))
    f.write("\n微调模型分类报告:\n")
    f.write(classification_report(test_df['rating'], fine_preds, target_names=['差评', '中评', '好评'], digits=4))

print("评估完成！结果已保存到 model_results.txt")
print(f"基线模型准确率: {base_acc:.4f}")
print(f"微调模型准确率: {fine_acc:.4f}")
print(f"准确率提升: {fine_acc - base_acc:.4f}")
>>>>>>> 155009d (🎉 init:项目初版)
