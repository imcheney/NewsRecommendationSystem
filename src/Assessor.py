"""
正确性衡量
precision: hit/L, L=RecommendList
recall: hit/T, T=TotalActualRead

测试通过
"""

import Engine
import pandas as pd
import numpy as np


def create_recommendResult(method, L):
    wholeSet_newsid_and_readtime_df = pd.read_csv('../data/wholeSet_newsid_and_readtime_table.csv').loc[:,
                                      ['news_id', 'read_time']]
    testSet_users = list(np.load("../data/testSet_useridSet.npy").item())
    d = {}
    count = 0
    for userid in testSet_users:
        recommended_newsid_list = []
        if method == 'cb':
            recommended_newsid_list = Engine.content_based_service(wholeSet_newsid_and_readtime_df, userid, L)
        elif method == 'cf':
            recommended_newsid_list = Engine.collaborative_filtering_service(wholeSet_newsid_and_readtime_df, userid, L)
        elif method == 'hot':
            if count == 0:
                print('start running hot method...')
            recommended_newsid_list = Engine.hot_news_service(wholeSet_newsid_and_readtime_df, userid, L)
        else:
            print('ERROR INPUT!')
            exit(1)
        d[userid] = recommended_newsid_list
        count += 1
        if count % 100 == 0:
            print(str.format("%d saving list of %d for user%d..." % (count, len(recommended_newsid_list), userid)))
    filename = str.format("../data/result_%s_L=%d.npy" % (method, L))
    np.save(filename, d)
    print("job done!")


def get_precision_rate(rec_d, actual_d):
    if isinstance(rec_d, dict):
        hit = 0
        total = 0
        for userid in rec_d.keys():
            rec_news_list = rec_d[userid]
            total += len(rec_news_list)
            actual_news_list = actual_d[userid]
            for newsid in rec_news_list:
                if newsid in actual_news_list:  # 只要有一篇新闻被点击即可
                    hit += 1
        precision_rate = hit / total
        return precision_rate


def get_recall_rate(rec_d, actual_d):
    if isinstance(actual_d, dict):
        hit = 0
        total = 0
        for userid in actual_d.keys():
            actual_news_list = actual_d[userid]
            rec_news_list = rec_d[userid]
            total += len(actual_news_list)
            for newsid in actual_news_list:
                if newsid in rec_news_list:
                    hit += 1
        recall_rate = hit / total
        return recall_rate


# def get_revised_precision_rate(rec_d, actual_d):
#     if isinstance(rec_d, dict):
#         hit_user = 0
#         total_user = 0
#         for userid in rec_d.keys():
#             total_user += 1
#             rec_news_list = rec_d[userid]
#             all = 0
#             take = 0
#             all += len(rec_news_list)
#             actual_news_list = actual_d[userid]
#             for newsid in rec_news_list:
#                 if newsid in actual_news_list:
#                     take += 1
#             if take / all >= 0.2:  # 如果能够有r比例的新闻被阅读, 就认为是成功
#                 hit_user += 1
#         revised_precision_rate = hit_user / total_user
#         return revised_precision_rate
#
#
# def get_revised_recall_rate(rec_d, actual_d):
#     if isinstance(actual_d, dict):
#         hits = 0
#         total = len(actual_d.keys())
#         for userid in actual_d.keys():
#             flag = 0
#             actual_news_list = actual_d[userid]
#             rec_news_list = rec_d[userid]
#             for newsid in actual_news_list:
#                 if newsid in rec_news_list:
#                     flag = 1  # 如果新闻被阅读, 标记为成功
#             if flag >= 1:
#                 hits += 1
#         recall_rate = hits / total
#         return recall_rate


if __name__ == "__main__":
    # ---- 生成数据 ----
    L = 3
    method = "cb"
    # create_recommendResult(method, L)  # 如果已经生成好了, 则可以注释掉本行

    # ---- 评价自然定义的正确率 ----
    filename1 = str.format("../data/result_%s_L=%d.npy") % (method, L)
    rec_d = np.load(filename1).item()
    filename2 = "../data/testSet_userid_to_actualReadNewsid_table.npy"
    actual_d = np.load(filename2).item()

    print(str.format("precision: %f") % get_precision_rate(rec_d, actual_d))
    print(str.format("recall: %f") % get_recall_rate(rec_d, actual_d))

    # ---- 评价修正定义的正确率 ----
    # print(str.format("revised_precision: %f") % get_revised_precision_rate(rec_d, actual_d))
    # print(str.format("revised_recall: %f") % get_revised_recall_rate(rec_d, actual_d))
