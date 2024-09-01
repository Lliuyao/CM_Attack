import pandas as pd
import transformers
from transformers import AutoTokenizer, BertModel, AutoModelForMaskedLM

from textattack import Attack, DatasetArgs, attacker
from textattack.dataset import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.transformations import WordInsertionMaskedLM
from textattack.constraints.pre_transformation import (
   RepeatModification,
   StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch

from transformers import BertTokenizer
from console import Console
def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum():
            word += c
        elif c in "'-" and len(word) > 0:
            # Allow apostrophes and hyphens as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words

def read_imdb_dataset(path, data):
    df = pd.read_csv(path)
    dataset = []

    if data == '500samples_imdb.csv':
        s_list = df["sentence"]
        s_label = df["polarity"]
    elif data == '500samples_mnli.csv':
        s_list = df["sentence"]
        s_label = df["polarity"]
    else:
        s_list = df["text"]
        s_label = df["label"]

    for index, label in enumerate(s_label):
        sent = s_list[index]
        sent = sent.replace("<br />", " ")

        # limit the number of tokens to 510 for bert (since only bert is used for language modeling, see lm_sampling.py)
        sent = words_from_text(sent)
        # print(sent)
        # print(len(sent))
        sent = ' '.join(sent)

        tok = BertTokenizer.from_pretrained('D:/TextAttack/bert-base-uncased')
        tokens = tok.tokenize(sent)
        tokens = tokens[:510]
        text = ' '.join(tokens).replace(' ##', '').replace(' - ', '-').replace(" ' ", "'")

        target = int(label)
        dataset.append((text, target))

    return dataset

# 加载模型和数据集
model_path = 'D:/TextAttack/textattack/bert-base-uncased-imdb'
model_name = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = HuggingFaceModelWrapper(model_name, tokenizer)

dataset = read_imdb_dataset("D:/TextAttack/testdata/500samples_imdb.csv", '500samples_imdb.csv')
dataset = dataset[0:4]

# 创建转换器
transformation = WordInsertionMaskedLM(max_candidates=50, min_confidence=0.0)

# 创建约束条件
constraints = [RepeatModification(), StopwordModification()] # 限制每次转换只能插入一个单词
use_constraint = UniversalSentenceEncoder(
           threshold=0.936338023,
           metric="cosine",
           compare_against_original=True,
           window_size=15,
           skip_text_shorter_than_window=True,
       )
constraints.append(use_constraint)

# 创建目标函数
goal_function = UntargetedClassification(model)

# 创建搜索方法
search_method = GreedySearch()

# 创建攻击模型
attack = Attack(goal_function, constraints, transformation, search_method)

print(dataset)

# result = Console(dataset, 4, attack)
# result.print_console()
# 执行攻击
# input_text = "I really enjoyed the new movie that came out last month."
# label = 1 #Positive
# attack_result = attack.attack(input_text, label)

# results = attack.attack_dataset(dataset)
# 对数据集中的每个样本进行攻击，并获取攻击结果生成器
results_generator = attack.attack_dataset(dataset)

# 遍历生成器，获取攻击结果并打印
# for result in results_generator:
#     print("原始文本:", result.original_text)
#     print("攻击后文本:", result.perturbed_text)
#     print("攻击成功:", result.success)
#     print("-------------------------------")
# 遍历攻击结果
# for result in results:
#    print(result.__str__(color_method="ansi"))