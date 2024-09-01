import random
import csv

texts = []
labels = []

# 读取原始数据集文件
with open('D:/TextAttack/examples/dataset/thuc_news.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = list(reader)  # 将数据转换为列表形式

# 随机选择500个样本
random.shuffle(data)
selected_data = data[:500]

# 提取review和label列的值
for row in selected_data:
    text = row['text']
    label = row['label']
    texts.append(text)
    labels.append(label)

# 将选中的样本保存为TSV文件
with open('D:/TextAttack/examples/dataset/500samples_thuc_news.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')  # 指定分隔符为制表符
    writer.writerow(['index', 'text', 'label'])  # 写入标题行
    writer.writerows(zip(texts, labels))  # 写入数据行

print("数据已保存为TSV文件。")