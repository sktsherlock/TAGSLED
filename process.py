import pandas as pd
import random
import json
import argparse
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 创建解析器
parser = argparse.ArgumentParser(description="Process dataset for LLM.")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset CSV file.")
parser.add_argument('--text_column', type=str, default='text', help="Name of the column containing the text.")
parser.add_argument('--category_column', type=str, default='category',
                    help="Name of the column containing the categories.")
parser.add_argument('--node_id_column', type=str, required=False, default='node_id',
                    help="Name of the column containing the node ID.")
parser.add_argument('--neighbour_column', type=str, required=False, default='neighbour',
                    help="Name of the column containing neighbours.")
parser.add_argument('--false_category_ratio', type=float, default=0.5,
                    help="Ratio of false categories to sample (0.0 - 1.0).")  # 添加新的超参数

# 解析参数
args = parser.parse_args()

# 确定输出文件路径
input_dir, input_file = os.path.split(args.dataset)
file_name, file_ext = os.path.splitext(input_file)
output_file = os.path.join(input_dir, f"{file_name}_LLM{file_ext}")
stats_file = os.path.join(input_dir, f"{file_name}_stats.txt")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载数据集
df = pd.read_csv(args.dataset)

# 获取类别列表
categories = df[args.category_column].unique()

# 打印类别信息
print("Unique Categories in the dataset:")
for category in categories:
    print(category)

# 创建一个任务相关的 prompt
categories_str = ", ".join(categories)

# 创建新的数据框架用于存储每个样本的相关信息
new_data = []
tokenized_lengths = []

# 遍历每一行数据，生成适合生成式文本分类的数据格式
# 构建每个样本的 prompt，并将其添加到新数据结构中
for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing dataset"):  # 使用tqdm包装iterrows
    text = row[args.text_column]
    correct_category = row[args.category_column]

    # 从数据集中获取 false_categories（即不是该文本的类别）
    # 从数据集中获取 false_categories（即不是该文本的类别）
    other_categories = [cat for cat in categories if cat != correct_category]
    # false_categories = [cat for cat in categories if cat != correct_category]


    # 随机采样 false_categories，采样比例由参数控制
    sample_size = max(1, int(len(other_categories) * args.false_category_ratio))  # 至少采样1个
    false_categories = random.sample(other_categories, sample_size)

    # 构建任务相关的 prompt
    prompt = f"""
    Your task is to classify the given text into one of the following categories: {categories_str}. Please read the description below and choose the correct category based on the content.
    The description and title of the item are as follows:
    {text}
    Please think step by step. This item likely belongs to:
    """

    # 分词并统计长度
    tokenized_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    tokenized_lengths.append(tokenized_length)

    # 将新的信息添加到数据集
    new_entry = {
        'prompt': prompt,
        'raw_text': text,
        'correct_category': correct_category,
        'false_categories': false_categories,
    }

    # 可选列
    if args.node_id_column in df.columns:
        new_entry['node_id'] = row[args.node_id_column]
    if args.neighbour_column in df.columns:
        new_entry['neighbour'] = row[args.neighbour_column]

    new_data.append(new_entry)


# 计算统计信息
length_stats = {
    "min": np.min(tokenized_lengths),
    "max": np.max(tokenized_lengths),
    "mean": np.mean(tokenized_lengths),
    "std": np.std(tokenized_lengths),
    "25%": np.percentile(tokenized_lengths, 25),
    "50% (median)": np.percentile(tokenized_lengths, 50),
    "75%": np.percentile(tokenized_lengths, 75),
}

# 写入统计信息到文本文件
with open(stats_file, 'w') as f:
    f.write("Prompt Tokenized Length Statistics:\n")
    for stat, value in length_stats.items():
        f.write(f"{stat}: {value:.2f}\n")

# 输出统计信息到控制台
print("Statistics of tokenized lengths:")
for stat, value in length_stats.items():
    print(f"{stat}: {value:.2f}")

# 将新的数据框架转化为 Pandas DataFrame
new_df = pd.DataFrame(new_data)

# 输出新的数据框架
print(new_df.head())

# 保存为 CSV 格式
new_df.to_csv(output_file, index=False)

print(f"Data conversion complete! Output saved to {output_file}")
print(f"Statistics saved to {stats_file}")
