# # 原始文本和对抗文本
# original_text = "火龙果并不新鲜，外皮打蔫，而且已经软了，让我非常不满意，申请了优鲜赔！希望京东能够严格把关生鲜食品的质量"
# perturbed_text = "火龙果并不心鲜，外皮打蔫，而且已经软了，让我非常不慢意，申请了优鲜配！希望京东能够严格把关生鲜食品的质量"
#
# # 计算词汇差异
# diff_indices = set()
# for i in range(min(len(original_text), len(perturbed_text))):
#     if original_text[i] != perturbed_text[i]:
#         diff_indices.add(i)
#
# # 输出结果
# num_words_changed = len(diff_indices)
# print("修改的词数:", num_words_changed)
import jieba

# import jieba
# # 原始文本和对抗文本
# # original_text = "火龙果并不新鲜，外皮打蔫，而且已经软了，让我非常不满意，申请了优鲜赔！希望京东能够严格把关生鲜食品的质量"
# # perturbed_text = "火龙果并不心鲜，外皮打蔫，而且已经软了，让我非常不慢意，申请了优鲜配！希望京东能够严格把关生鲜食品的质量"
# original_text = "说句实话，之前买东西一直都是在京东，比较信任，可是这次太失望了，东西刚买来就降价*，申请了价格保护，直接回复不支持。。第一次对京东这么失望。"
# perturbed_text = "说句实话，之前买东西一直都是在京东，比较信任，可是这次太忙了，东西刚买来就降价*，申请了价格保护，直接回复不支持。。第一次对京东这么满意。"
#
# # 分词函数
# def tokenize(text):
#     return list(jieba.cut(text))
#
# # 分词
# original_tokens = tokenize(original_text)
# perturbed_tokens = tokenize(perturbed_text)
# print(original_tokens)
# print(perturbed_tokens)
#
# # 判断长度是否一致
# if len(original_tokens) == len(perturbed_tokens):
#     # 使用字符级别的比较方法
#     diff_indices = set()
#     for i in range(min(len(original_tokens), len(perturbed_tokens))):
#         if original_tokens[i] != perturbed_tokens[i]:
#             diff_indices.add(i)
# else:
#     # 使用分词级别的比较方法
#     diff_indices = set()
#     for i in range(min(len(original_tokens), len(perturbed_tokens))):
#         if original_tokens[i] != perturbed_tokens[i]:
#             diff_indices.add(i)
#
# # 输出结果
# num_words_changed = len(diff_indices)
# print("修改的词数:", num_words_changed)

# 原始文本和对抗文本
# original_text = "说句实话，之前买东西一直都是在京东，比较信任，可是这次太失望了，东西刚买来就降价*，申请了价格保护，直接回复不支持。。第一次对京东这么失望。"
# perturbed_text = "说句实话，之前买东西一直都是在京东，比较信任，可是这次太忙了，东西刚买来就降价*，申请了价格保护，直接回复不支持。。第一次对京东这么满意。"
# original_text = "火龙果并不新鲜，外皮打蔫，而且已经软了，让我非常不满意，申请了优鲜赔！希望京东能够严格把关生鲜食品的质量"
# perturbed_text = "火龙果并不心鲜，外皮打蔫，而且已经软了，让我非常不慢意，申请了优鲜配！希望京东能够严格把关生鲜食品的质量"
# original_text = "键盘很容易烂.大家要注意这一点.维修点少得可怜."
# perturbed_text = "键盘很容易蓝.大家要注意这一点.维秀点少得可练."
original_text = "苹果很难吃，超一半都是有碰坏，一个削完剩半个，口感偏酸又不甜，跟我之前在山姆店买的华圣苹果无法比，返还了部分钱，还是不值，跟商场卖的特价处理品一样。"
perturbed_text = "苹果很难吃，超一半都是有碰坏，一个削完剩半个，口感偏酸又不甜，跟我之前在山姆店买的华圣苹果无法比，返还了部分钱，还是很不错，跟商场卖的特价处理品一样。"


# 分词函数
def tokenize(text):
    return list(jieba.cut(text))


# 分词
original_tokens = tokenize(original_text)
perturbed_tokens = tokenize(perturbed_text)


# 动态规划计算最长公共子序列
def longest_common_subsequence(tokens1, tokens2):
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# 计算修改的词数
num_words_changed = max(len(original_tokens), len(perturbed_tokens)) - longest_common_subsequence(original_tokens,
                                                                                                  perturbed_tokens)

# 输出结果
print("修改的词数:", num_words_changed)

    # def perform_search(self, initial_result):
    #     attacked_text = initial_result.attacked_text
    #
    #     # Sort words by order of importance
    #     index_order, search_over = self._get_index_order(attacked_text)
    #     i = 0
    #     cur_result = initial_result
    #     results = None
    #     while i < len(index_order) and not search_over:
    #         transformed_text_candidates = self.get_transformations(
    #             cur_result.attacked_text,
    #             original_text=initial_result.attacked_text,
    #             indices_to_modify=[index_order[i]],
    #         )
    #         i += 1
    #         if len(transformed_text_candidates) == 0:
    #             continue
    #         results, search_over = self.get_goal_results(transformed_text_candidates)
    #         results = sorted(results, key=lambda x: -x.score)
    #         # Skip swaps which don't improve the score
    #         if results[0].score > cur_result.score:
    #             cur_result = results[0]
    #         else:
    #             continue
    #         # If we succeeded, return the index with best similarity.
    #         if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
    #             best_result = cur_result
    #             # @TODO: Use vectorwise operations
    #             max_similarity = -float("inf")
    #             for result in results:
    #                 if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
    #                     break
    #                 candidate = result.attacked_text
    #                 try:
    #                     similarity_score = candidate.attack_attrs["similarity_score"]
    #                 except KeyError:
    #                     # If the attack was run without any similarity metrics,
    #                     # candidates won't have a similarity score. In this
    #                     # case, break and return the candidate that changed
    #                     # the original score the most.
    #                     break
    #                 if similarity_score > max_similarity:
    #                     max_similarity = similarity_score
    #                     best_result = result
    #             return best_result
    #
    #     return cur_result