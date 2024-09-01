import os
import pandas as pd
import pinyin

from . import WordSwap


class ChinesePinyinSwap(WordSwap):
    """Transforms an input by replacing its words with their corresponding pinyin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add any additional initialization code or property assignments here


    def _get_replacement_words(self, word):
        """Returns a list containing the pinyin representation of the word."""
        pinyin_word = pinyin.get(word, format="strip", delimiter=" ")
        return [pinyin_word]