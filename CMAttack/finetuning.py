from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, BertTokenizer, \
    BertForSequenceClassification, AutoTokenizer
import torch



# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. 下载 THUCNews_bert 模型权重
# thucnews_bert_model = BertForSequenceClassification.from_pretrained('uer/THUCNews_bert', from_tf=True)
# thucnews_bert_tokenizer = BertTokenizer.from_pretrained('uer/THUCNews_bert')

# 2. 创建 RoBERTa 模型
roberta_model = RobertaForSequenceClassification.from_pretrained('uer/roberta-base-wwm-chinese-cluecorpussmall', num_labels=10)

# # 3. 加载 THUCNews_bert 模型权重
# roberta_model.load_state_dict(thucnews_bert_model.state_dict())

# 加载 RoBERTa 分词器
roberta_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-wwm-chinese-cluecorpussmall')

# 4. 微调训练
# 加载训练集、测试集和验证集
dataset = load_dataset("arrow", data_files={"train": "datasets/THUCNews/thuc_news-train.arrow", "test": "datasets/THUCNews/thuc_news-test.arrow", "val": "datasets/THUCNews/thuc_news-validation.arrow"})

# 将模型移动到 GPU
roberta_model.to(device)

# # 将文本转换为模型输入特征
# def tokenize_batch(batch):
#     tokens = roberta_tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors='pt', max_length=128)
#     tokens['labels'] = torch.tensor(batch['label'], dtype=torch.long)
#     return tokens
#
# # 定义 DataLoader
# train_loader = DataLoader(dataset["train"].map(tokenize_batch), batch_size=16, shuffle=True)
# test_loader = DataLoader(dataset["test"].map(tokenize_batch), batch_size=16, shuffle=False)
# val_loader = DataLoader(dataset["val"].map(tokenize_batch), batch_size=16, shuffle=False)
# 定义分词处理函数
def tokenize_function(examples):
    return roberta_tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# 预处理数据集
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 创建 DataLoader
train_loader = DataLoader(tokenized_datasets["train"], batch_size=64, shuffle=True)
val_loader = DataLoader(tokenized_datasets["val"], batch_size=64, shuffle=False)

# 优化器和损失函数
optimizer = AdamW(roberta_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    roberta_model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = roberta_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    roberta_model.eval()
    val_loss = 0
    total_correct = 0
    total_samples = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = roberta_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            val_loss += criterion(logits, labels).item()  # 使用损失函数计算损失

            _, predicted = torch.max(logits, 1)  # 获取预测结果

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print(f"Epoch {epoch}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Validation Accuracy: {accuracy}")

# 5. 模型评估
# 在 THUCNews 测试集上评估模型性能,并与 THUCNews_bert 模型进行比较

# 6. 模型部署
roberta_model.save_pretrained('roberta-thucnews')
roberta_tokenizer.save_pretrained('roberta-thucnews')

# thucnews_roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-thucnews')