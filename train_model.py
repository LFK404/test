import torch
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification, 
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score
import os
import shutil

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('./chinese_bert')
model = BertForSequenceClassification.from_pretrained('./chinese_bert', num_labels=3)
model.to(device)

# 加载训练数据和验证数据
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
print(f"训练集: {len(train_df)}条, 验证集: {len(val_df)}条")

# 显示数据分布
print("训练集分布:", train_df['rating'].value_counts().sort_index().to_dict())
print("验证集分布:", val_df['rating'].value_counts().sort_index().to_dict())

# 加载并加强类别权重
class_weights = np.load('class_weights.npy')
print("原始类别权重:", class_weights)

# 加强类别权重
class_weights = class_weights * 3
print("加强后类别权重:", class_weights)

# 文本编码
def encode_texts(texts, labels):
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=128, 
        return_tensors='pt'
    )
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'], 
        'labels': torch.tensor(labels, dtype=torch.long)
    }

train_encodings = encode_texts(train_df['review_clean'].tolist(), train_df['rating'].tolist())
val_encodings = encode_texts(val_df['review_clean'].tolist(), val_df['rating'].tolist())

# 创建数据集
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['labels'])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

train_dataset = ReviewDataset(train_encodings)
val_dataset = ReviewDataset(val_encodings)

# 自定义Trainer以处理类别不平衡
class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1_macro = 0
        self.best_f1_weighted = 0
        self.best_macro_checkpoint = None
        self.best_weighted_checkpoint = None
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 应用加强后的类别权重
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 计算评估指标 - 同时使用宏平均和加权平均F1分数
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')  
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# 创建保存最佳模型的回调
class SaveBestModelsCallback(TrainerCallback):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.best_f1_macro = 0
        self.best_f1_weighted = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 保存基于宏平均F1的最佳模型
        if metrics and 'eval_f1_macro' in metrics:
            current_f1_macro = metrics['eval_f1_macro']
            if current_f1_macro > self.best_f1_macro:
                self.best_f1_macro = current_f1_macro
                print(f"新的宏平均F1最佳模型: {current_f1_macro:.4f}")
                # 保存宏平均最佳模型
                output_dir = os.path.join(args.output_dir, "best_f1_macro")
                kwargs['model'].save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        
        # 保存基于加权平均F1的最佳模型
        if metrics and 'eval_f1_weighted' in metrics:
            current_f1_weighted = metrics['eval_f1_weighted']
            if current_f1_weighted > self.best_f1_weighted:
                self.best_f1_weighted = current_f1_weighted
                print(f"新的加权平均F1最佳模型: {current_f1_weighted:.4f}")
                # 保存加权平均最佳模型
                output_dir = os.path.join(args.output_dir, "best_f1_weighted")
                kwargs['model'].save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

# 训练参数 - 不使用自动保存最佳模型
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=96,  
    per_device_eval_batch_size=128,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1, 
    weight_decay=0.01, 
    fp16=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=False,  # 关键：关闭自动加载最佳模型
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    max_grad_norm=1.0,
    optim='adamw_torch_fused',
    dataloader_num_workers=6,  
)

# 创建回调实例
save_best_callback = SaveBestModelsCallback(tokenizer)

# 开始训练
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4), save_best_callback]
)

print("开始训练模型...")
trainer.train()

print("训练完成！")
print(f"训练过程中保存的最佳模型:")
print(f"- 宏平均F1最佳模型: ./results/best_f1_macro (验证集F1: {save_best_callback.best_f1_macro:.4f})")
print(f"- 加权平均F1最佳模型: ./results/best_f1_weighted (验证集F1: {save_best_callback.best_f1_weighted:.4f})")

# 检查模型是否真的保存了
print("\n检查保存的模型:")
for model_dir in ['./results/best_f1_macro', './results/best_f1_weighted']:
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"✅ {model_dir}: {len(files)}个文件")
        for file in files:
            print(f"   - {file}")
    else:
        print(f"❌ {model_dir}: 目录不存在")