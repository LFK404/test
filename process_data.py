<<<<<<< HEAD
=======
# process_data.py
>>>>>>> 155009d (🎉 init:项目初版)
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
<<<<<<< HEAD
    def star_to_label(star):
        star = int(star)
        if star in [1,2]:
            return 0  # 差评
        elif star == 3:
            return 1  # 中评
        else:
            return 2  # 好评
    
    df['rating'] = df['star'].apply(star_to_label)
=======
    star_to_label = {5: 2, 4: 2, 3: 1, 2: 0, 1: 0}
    df['rating'] = df['star'].map(star_to_label)
    df = df.dropna(subset=['rating'])
>>>>>>> 155009d (🎉 init:项目初版)
    return df

def load_data():
    """加载和处理ASAP数据集"""
    train_url = "https://raw.githubusercontent.com/Meituan-Dianping/asap/refs/heads/master/data/train.csv"
    test_url = "https://raw.githubusercontent.com/Meituan-Dianping/asap/refs/heads/master/data/test.csv"

<<<<<<< HEAD
    # 加载训练集和测试集
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    print(f"成功加载数据：训练集{len(train_df)}条，测试集{len(test_df)}条")
=======
    try:
        # 加载训练集和测试集
        train_df = pd.read_csv(train_url)
        test_df = pd.read_csv(test_url)
        print(f"成功加载数据：训练集{len(train_df)}条，测试集{len(test_df)}条")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
>>>>>>> 155009d (🎉 init:项目初版)

    # 基础清洗
    train_df = train_df.dropna().drop_duplicates()
    test_df = test_df.dropna().drop_duplicates()

    # 使用star字段生成三分类标签
    train_df = make_three_class(train_df)
    test_df = make_three_class(test_df)

<<<<<<< HEAD
    # 文本清洗
=======
    # 文本清洗 - 使用review列
>>>>>>> 155009d (🎉 init:项目初版)
    train_df['review_clean'] = train_df['review'].apply(clean_text)
    test_df['review_clean'] = test_df['review'].apply(clean_text)

    # 过滤过短文本
    train_df = train_df[train_df['review_clean'].str.len() > 5]
    test_df = test_df[test_df['review_clean'].str.len() > 5]

<<<<<<< HEAD
    # 从训练集中划分验证集
=======
    # 从训练集中划分验证集（80%训练，20%验证）
>>>>>>> 155009d (🎉 init:项目初版)
    train_df, val_df = train_test_split(
        train_df[['review_clean', 'rating']],
        test_size=0.2,
        random_state=42,
<<<<<<< HEAD
        stratify=train_df['rating']
=======
        stratify=train_df['rating']  # 分层抽样保持类别比例
>>>>>>> 155009d (🎉 init:项目初版)
    )

    # 保存处理后的数据
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df[['review_clean', 'rating']].to_csv('test_data.csv', index=False)

    print("数据处理完成！")
    
<<<<<<< HEAD
    # 计算类别权重
=======
    # 检查类别是否平衡
    label_counts = train_df['rating'].value_counts().sort_index()
    min_count = label_counts.min()
    max_count = label_counts.max()
    imbalance_ratio = max_count / min_count
    
    print("训练集标签分布：", label_counts.to_dict())
    print("验证集标签分布：", val_df['rating'].value_counts().sort_index().to_dict())
    print("测试集标签分布：", test_df['rating'].value_counts().sort_index().to_dict())
    print(f"类别不平衡比例（最大/最小）：{imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0:
        print("⚠️ 类别不平衡较严重，已计算类别权重用于训练。")
    else:
        print("✅ 类别基本平衡。")

    # 计算类别权重（用于处理类别不平衡）
>>>>>>> 155009d (🎉 init:项目初版)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1, 2]),
        y=train_df['rating'].values
    )
<<<<<<< HEAD
=======
    print("类别权重:", class_weights)
>>>>>>> 155009d (🎉 init:项目初版)
    
    # 保存类别权重
    np.save('class_weights.npy', class_weights)

if __name__ == "__main__":
    load_data()