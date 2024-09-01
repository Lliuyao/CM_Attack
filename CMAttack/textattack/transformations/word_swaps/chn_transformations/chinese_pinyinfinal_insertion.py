import os
import pandas as pd
import pinyin

from . import WordSwap


class ChinesePinyinInsertion(WordSwap):
    """Transforms an input by replacing its words with their corresponding pinyin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加任何额外的初始化代码或属性赋值在这里


    def _get_replacement_words(self, word):
        """返回一个包含原词和其最后一个字对应拼音韵母的列表."""
        pinyin_word = pinyin.get(word, format="strip", delimiter=" ")

        # 获取最后一个字的拼音韵母
        last_character = word[-1]
        last_character_pinyin = pinyin.get(last_character, format="strip", delimiter=" ")

        # 获取最后一个字的拼音韵母部分
        last_vowel = ''
        for pinyin_syllable in last_character_pinyin.split(' '):
            for vowel in [
                          'ang', 'eng', 'ing', 'ong', 'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 'üe', 'er',
                          'an', 'en', 'in', 'un', 'a', 'o', 'e', 'i', 'u', 'ü']:
                if vowel in pinyin_syllable:
                    last_vowel = vowel
                    break
            if last_vowel:
                break

        # 将原词和最后一个字的拼音韵母部分拼接在一起
        word_with_last_vowel = f"{word}{last_vowel}"

        return [word_with_last_vowel]

    def extract_vowel(self, pinyin_word):
        """Extracts the vowel part from pinyin."""
        vowels = ['a', 'o', 'e', 'i', 'u', 'ü']
        for vowel in vowels:
            if vowel in pinyin_word:
                return vowel

        return ''

