# download_model.py
import os
from transformers import BertTokenizer, BertForSequenceClassification

print("使用国内镜像下载BERT-base-chinese模型...")

# 设置环境变量使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)

# 保存到本地
model.save_pretrained('./chinese_bert')
tokenizer.save_pretrained('./chinese_bert')

print("模型下载完成！")