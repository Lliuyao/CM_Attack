import datasets
import pandas as pd

# 提供.arrow文件的路径
arrow_file_path = "G:/datasets/THUCNews/thuc_news-test.arrow"

# 加载数据集
dataset = datasets.load_dataset("arrow", data_files=arrow_file_path)

# 获取数据集中的所有样本
samples = dataset["train"]

# 将样本转换为 DataFrame
df = pd.DataFrame(samples)

# 保存 DataFrame 到 CSV 文件
csv_file_path = "D:/TextAttack/examples/dataset/thuc_news_validation.csv"
df.to_csv(csv_file_path, index=False)

print("Data saved to", csv_file_path)
