import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def clean_text(text):
    """清洗文本数据"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\u4e00-\u9fff，。！？；：""\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def make_three_class(df):
    """使用star字段构建三分类标签"""
    def star_to_label(star):
        star = int(star)
        if star in [1,2]:
            return 0  # 差评
        elif star == 3:
            return 1  # 中评
        else:
            return 2  # 好评
    
    df['rating'] = df['star'].apply(star_to_label)
    return df

def load_data():
    """加载和处理ASAP数据集"""
    train_url = "https://raw.githubusercontent.com/Meituan-Dianping/asap/refs/heads/master/data/train.csv"
    test_url = "https://raw.githubusercontent.com/Meituan-Dianping/asap/refs/heads/master/data/test.csv"

    # 加载训练集和测试集
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    print(f"成功加载数据：训练集{len(train_df)}条，测试集{len(test_df)}条")

    # 基础清洗
    train_df = train_df.dropna().drop_duplicates()
    test_df = test_df.dropna().drop_duplicates()

    # 使用star字段生成三分类标签
    train_df = make_three_class(train_df)
    test_df = make_three_class(test_df)

    # 文本清洗
    train_df['review_clean'] = train_df['review'].apply(clean_text)
    test_df['review_clean'] = test_df['review'].apply(clean_text)

    # 过滤过短文本
    train_df = train_df[train_df['review_clean'].str.len() > 5]
    test_df = test_df[test_df['review_clean'].str.len() > 5]

    # 从训练集中划分验证集
    train_df, val_df = train_test_split(
        train_df[['review_clean', 'rating']],
        test_size=0.2,
        random_state=42,
        stratify=train_df['rating']
    )

    # 保存处理后的数据
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df[['review_clean', 'rating']].to_csv('test_data.csv', index=False)

    print("数据处理完成！")
    
    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1, 2]),
        y=train_df['rating'].values
    )
    
    # 保存类别权重
    np.save('class_weights.npy', class_weights)

if __name__ == "__main__":
    load_data()