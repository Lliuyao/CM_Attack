import pandas as pd
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf
import torch
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
# from textattack.shared.utils import LazyLoader
#
# hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")

# 定义计算欧氏距离的函数
def get_neg_euclidean_dist(vec1, vec2):
    return euclidean(vec1, vec2)

# 读取CSV文件
df = pd.read_csv('successful_attacks1.csv', encoding='gbk')

# 加载本地MUSE模型
model_path = 'D:/AttackCode/USE/universal-sentence-encoder-multilingual'  # 替换为你的模型目录路径
embed = hub.load(model_path)

# 获取Original Text和Perturbed Text的嵌入
original_texts = df['Original Text'].tolist()
perturbed_texts = df['Perturbed Text'].tolist()

original_embeddings = embed(original_texts)
perturbed_embeddings = embed(perturbed_texts)

# 将 TensorFlow 的 Tensor 转换为 PyTorch 的 Tensor
original_embeddings_torch = torch.tensor(original_embeddings.numpy())
perturbed_embeddings_torch = torch.tensor(perturbed_embeddings.numpy())

# 计算欧氏距离
euclidean_distances = [get_neg_euclidean_dist(original_embeddings_torch[i], perturbed_embeddings_torch[i]) for i in range(len(original_texts))]

# 将欧氏距离添加到DataFrame中
df['Euclidean Distance'] = euclidean_distances

# 计算平均欧氏距离
average_distance = df['Euclidean Distance'].mean()

# 设置pandas显示选项以显示所有行和列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# 打印结果
# print("样本的平均欧氏距离:", average_distance)
# print(df[['Original Text', 'Perturbed Text', 'Euclidean Distance']])
print(df[['Euclidean Distance']])