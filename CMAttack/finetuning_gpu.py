import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.model_selection import train_test_split

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的BERT模型和tokenizer，并将其移动到GPU上
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=15)
model.to(device)

# 加载数据集
train_data = pd.read_csv('testdata/tnews/train.tsv', sep='\t', header=None, skiprows=1)
dev_data = pd.read_csv('testdata/tnews/dev.tsv', sep='\t', header=None, skiprows=1)

# 提取文本和标签
texts = train_data[0].tolist()
labels = train_data[1].tolist()

# 将文本转换为BERT模型所需的格式，并将其移动到GPU上
input_ids = []
attention_masks = []

for text in texts:
    encoded_dict = tokenizer.encode_plus(
        text,  # 文本
        add_special_tokens=True,  # 添加特殊标记
        max_length=128,  # 设定最大长度
        pad_to_max_length=True,  # 填充至最大长度
        return_attention_mask=True,  # 生成 attention mask
        return_tensors='pt',  # 返回 PyTorch 张量
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0).to(device)
attention_masks = torch.cat(attention_masks, dim=0).to(device)
labels = torch.tensor(labels).to(device)

# 划分训练集和验证集
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42,
                                                                                    test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.1)

# 创建DataLoaders时指定设备
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(validation_data, batch_size=64)

# 定义优化器和学习率
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

# 提前停止的参数
patience = 2
best_val_loss = float('inf')
current_patience = 0

# 训练模型
epochs = 10


# 辅助函数：计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    # 在验证集上评估模型
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    avg_val_loss = eval_loss / len(validation_dataloader)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Avg training loss: {avg_train_loss}")
    print(f"Avg validation loss: {avg_val_loss}")
    print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")

    # 提前停止
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        current_patience = 0
    else:
        current_patience += 1
        if current_patience >= patience:
            print("Early stopping triggered.")
            break

# 保存模型
output_dir = './tnews_bert_model_gpu/'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



