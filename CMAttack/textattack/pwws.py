import textattack
import transformers

import csv
from collections import namedtuple




# class Attack:
#     """An attack generates adversarial examples on text.
#
#     An attack is comprised of a goal function, constraints, transformation, and a search method. Use :meth:`attack` method to attack one sample at a time.
#
#     Args:
#         goal_function (:class:`~textattack.goal_functions.GoalFunction`):
#             A function for determining how well a perturbation is doing at achieving the attack's goal.
#         constraints (list of :class:`~textattack.constraints.Constraint` or :class:`~textattack.constraints.PreTransformationConstraint`):
#             A list of constraints to add to the attack, defining which perturbations are valid.
#         transformation (:class:`~textattack.transformations.Transformation`):
#             The transformation applied at each step of the attack.
#         search_method (:class:`~textattack.search_methods.SearchMethod`):
#             The method for exploring the search space of possible perturbations
#         transformation_cache_size (:obj:`int`, `optional`, defaults to :obj:`2**15`):
#             The number of items to keep in the transformations cache
#         constraint_cache_size (:obj:`int`, `optional`, defaults to :obj:`2**15`):
#             The number of items to keep in the constraints cache
#
#     Example::
#     """
#

# Load model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR
from textattack import Attack

goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
constraints = [
   RepeatModification(),
   StopwordModification(),
   WordEmbeddingDistance(min_cos_sim=0.9)
   ]
transformation = WordSwapEmbedding(max_candidates=50)
search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack
attack = Attack(goal_function, constraints, transformation, search_method)

input_text = "I really enjoyed the new movie that came out last month."
label = 1 #Positive
attack_result = attack.attack(input_text, label)
print(attack_result)


# Define a named tuple for data sample
# DataSample = namedtuple("DataSample", ["text", "label"])
# def load_custom_dataset():
#     dataset = []
#
#     # Open the CSV file
#     with open("custom_dataset.csv", "r") as file:
#         # Create a CSV reader
#         reader = csv.reader(file)
#
#         # Skip the header row if it exists
#         next(reader, None)
#
#         # Read each row in the CSV file
#         for row in reader:
#             # Extract the text and label from the row
#             text = row[1]
#             label = row[3]
#
#             # Create a data sample object and add it to the dataset list
#             sample = DataSample(text=text, label=label)
#             dataset.append(sample)
#
#     return dataset
# # Load your custom dataset
# dataset = load_custom_dataset()
#
# # Load model, tokenizer, and model wrapper
# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
# tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
# model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
#
# # Construct the attack components
# goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
# constraints = [textattack.constraints.RepeatModification(), textattack.constraints.StopwordModification()]
# transformation = textattack.transformations.WordSwapEmbedding(max_candidates=50)
# search_method = textattack.search_methods.GreedyWordSwapWIR(wir_method="delete")
#
# # Construct the attack
# attack = textattack.Attack(goal_function, constraints, transformation, search_method)
#
# # Perform the attack on each sample in the dataset
# for sample in dataset:
#     input_text = sample.text
#     label = sample.label
#     attack_result = attack.attack(input_text, label)
   # Process the attack result as needed