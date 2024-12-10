import pandas as pd
import random
import json

# 假设您的数据集存储在一个 CSV 文件中，并且已经加载到 DataFrame 中
df = pd.read_csv('Children_QA.csv')

# 假设我们有一个完整的类别列表（可以从数据集中提取或预定义）
categories = df['category'].unique()
# 打印类别信息
print("Unique Categories in the dataset:")
for category in categories:
    print(category)

# 创建一个任务相关的 prompt
categories_str = ", ".join(categories)


# 创建新的数据框架用于存储每个样本的相关信息
new_data = []

# 遍历每一行数据，生成适合生成式文本分类的数据格式
# 构建每个样本的 prompt，并将其添加到新数据结构中
for _, row in df.iterrows():
    text = row['text']
    correct_category = row['category']

    # 从数据集中获取 false_categories（即不是该文本的类别）
    false_categories = [cat for cat in categories if cat != correct_category]

    # 构建任务相关的 prompt
    prompt = f"""
    Your task is to classify the given text into one of the following categories: {categories_str}. Please read the description below and choose the correct category based on the content.
    The description and title of the item are as follows:
    {text}
    This item likely belongs to:
    """

    # 将新的信息添加到数据集
    new_data.append({
        'prompt': prompt,
        'raw_text': text,
        'correct_category': correct_category,
        'false_categories': false_categories,
        'node_id': row['node_id'],
        'neighbour': row['neighbour']
    })

# 将新的数据框架转化为 Pandas DataFrame
new_df = pd.DataFrame(new_data)

# 输出新的数据框架
print(new_df.head())

# 保存为 CSV 格式
new_df.to_csv('Children_QA.csv', index=False)

print("Data conversion complete!")
