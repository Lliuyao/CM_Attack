import os

import pandas as pd
import pinyin

from . import WordSwap


class ChineseHomophoneFinalSwap(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by a
    homophone dictionary."""

    def __init__(self):
        # Get the absolute path of the homophone dictionary txt
        path = os.path.dirname(os.path.abspath(__file__))
        path_list = path.split(os.sep)
        path_list = path_list[:-2]
        path_list.append("shared/chinese_homophone_char.txt")
        homophone_dict_path = os.path.join("/", *path_list)
        homophone_dict = pd.read_csv("D:\\TextAttack\\textattack\\shared/chinese_homophone_char.txt", header=None, sep=",")
        homophone_dict = homophone_dict[0].str.split("\t", expand=True)
        self.homophone_dict = homophone_dict

    def _get_replacement_words(self, word):

        """返回一个包含原词和其最后一个字对应拼音韵母的列表."""

        # 获取最后一个字的拼音韵母
        last_character = word[-1]
        last_character_pinyin = pinyin.get(last_character, format="strip", delimiter=" ")

        # 获取最后一个字的拼音韵母部分
        last_vowel = ''
        for pinyin_syllable in last_character_pinyin.split(' '):
            for vowel in ['ang', 'eng', 'ing', 'ong', 'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 'üe', 'er', 'an', 'en',
                          'in', 'un', 'a', 'o', 'e', 'i', 'u', 'ü']:
                if vowel in pinyin_syllable:
                    last_vowel = vowel
                    break
            if last_vowel:
                break

        """Returns a list containing all possible words with 1 character
        replaced by a homophone."""
        candidate_words = []
        for i in range(len(word)):
            character = word[i]
            character = pinyin.get(character, format="strip", delimiter=" ")
            if character in self.homophone_dict.values:
                for row in range(self.homophone_dict.shape[0]):  # df is the DataFrame
                    for col in range(0, 1):
                        if self.homophone_dict._get_value(row, col) == character:
                            for j in range(1, 4):
                                repl_character = self.homophone_dict[col + j][row]
                                if repl_character is None:
                                    break
                                candidate_word = (
                                    word[:i] + repl_character + word[i + 1 :]
                                )
                                candidate_words.append(candidate_word+last_vowel)
            else:
                pass

        return candidate_words
