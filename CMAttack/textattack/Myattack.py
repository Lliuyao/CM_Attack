import textattack
import transformers

# Load model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("D:/TextAttack/textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("D:/TextAttack/textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding, WordInsertionMaskedLM
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch

goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
constraints = [
    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.9)
]
# transformation = WordSwapEmbedding(max_candidates=50)
transformation = WordInsertionMaskedLM()
# search_method = GreedyWordSwapWIR(wir_method="delete")
search_method = GreedySearch()

# Construct the actual attack
attack = textattack.Attack(goal_function, constraints, transformation, search_method)

input_text = "I really enjoyed the new movie that came out last month."
label = 1 #Positive
attack_result = attack.attack(input_text, label)
print(attack_result)