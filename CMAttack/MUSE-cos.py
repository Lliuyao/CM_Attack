import pandas as pd
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
# from textattack.shared.utils import LazyLoader
#
# hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


# 读取CSV文件
df = pd.read_csv('successful_attacks.csv', encoding='gbk')

# 加载本地MUSE模型
model_path = 'D:/AttackCode/USE/universal-sentence-encoder-multilingual'  # 替换为你的模型目录路径
embed = hub.load(model_path)

# 获取Original Text和Perturbed Text的嵌入
original_texts = df['Original Text'].tolist()
perturbed_texts = df['Perturbed Text'].tolist()

original_embeddings = embed(original_texts)
perturbed_embeddings = embed(perturbed_texts)

# 计算余弦相似度
cosine_similarities = cosine_similarity(original_embeddings, perturbed_embeddings)

# 将相似度添加到DataFrame中
df['Semantic Similarity'] = [cosine_similarities[i][i] for i in range(len(df))]

# 计算平均语义相似性
average_similarity = df['Semantic Similarity'].mean()

# 设置pandas显示选项以显示所有行和列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


# 打印结果
print("样本的平均语义相似性:", average_similarity)
# print(df[['Original Text', 'Perturbed Text', 'Semantic Similarity']])
print(df[['Semantic Similarity']])
