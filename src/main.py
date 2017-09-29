# -*- coding: utf-8 -*-
# @Time    : 23/09/2017 09:55
# @Author  : Cheney
# @Software: PyCharm

import argparse
from collections import Counter

import pandas as pd
import numpy as np
import logging
import sys

logging.basicConfig(filename="recommend.log", format="(%(levelname)s:%(asctime)%:%(message)s))", \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
L = 5  # Length, i.e. number of news recommended to a user

# tagset代表用户的兴趣标签, 也就是其读过的文章的topK keyword(依据TF-IDF算出的topK)
testSet_newsid_to_tags_dict = np.load('../data/testSet_newsid_to_tags_table.npy').item()
trainSet_userid_to_tagset_dict = np.load('../data/trainSet_userid_to_tagset_table.npy').item()
trainSet_userid_to_groupTagset_dict = np.load('../data/trainSet_userid_to_groupTagset_table.npy').item()

testSet_userid_to_lastReadTime_dict = np.load('../data/testSet_userid_to_lastReadTime_table.npy').item() # 导入测试数据集中由用户id和用户最后一次浏览时间组成的表
wholeSet_newsid_and_readtime_df = pd.read_csv('../data/wholeSet_newsid_and_readtime_table.csv').loc[:, ['news_id', 'read_time']]
wholeSet_newsid_to_newstitle_dict = np.load('../data/wholeSet_newsid_to_newstitle_table.npy').item()


def calc_simVal(SetA, SetB):
    '''
    计算两个关系集之间的相似度. TODO:这里的实现应该是有问题的, 缺少了分母的并集计算. 不过我们要以实践做测试.
    这个计算的就是最简单的Jaccard similarity: ﻿sim(C1, C2) = |C1∩C2| / |C1∪C2|
    :param SetA: 集合A
    :param SetB: 集合B
    :return: 相似度
    '''
    sim = 0
    for key in SetA:
        if key in SetB:
            sim = sim + 1
    return sim  # TODO: 没有分母部分


def get_topK_key(dic, k):
    """
    返回value值最大的k个key
    :param dic: a dict
    :param k: k值
    :return: list of topK key of this dict
    """
    return [t[0] for t in sorted(dic.items(), key=lambda d:d[1], reverse=True)][:k]


def get_hot_news_list(newsid_and_readtime_df, k=3, time_end=1394788902, days=1):  # time_end默认是3-14日上午
    """
    获取time_end之前一整天中最热的k条新闻
    :param newsid_and_readtime_df:
    :param k:
    :param time_end:
    :param days:
    :return:
    """
    time_range = days * 24 * 3600  # days*一天的秒数
    news = newsid_and_readtime_df[newsid_and_readtime_df['read_time'] < time_end]  # time_end之前的阅读记录
    news = news[news['read_time'] > time_end - time_range]  # time_end前一整天的阅读记录
    newsid_df = news.loc[:, 'news_id']
    counter = Counter(newsid_df.values)
    hotNews_list = [ituple[0] for ituple in counter.most_common(k)]  # k=3, 所以是一天中阅读记录里出现次数最多的top3新闻
    return hotNews_list


def calc_simVal_between_givenUser_and_eachTestSetNews(userid, userid_to_tags_dict, testSet_newsid_to_tags_dict):
    """
    两种算法的共同中间步骤
    :param userid: 用户id
    :param userid_to_tags_dict: 单用户(或者被判定相似用户group)的喜好标签
    :param testSet_newsid_to_tags_dict: 每条新闻的topK标签(TF-IDF关键词)
    :return: newsid_to_simVal_dict: testSet中每个新闻id最后与该指定用户喜好标签算出的相似度
    """
    newsid_to_simVal_dict = {}
    if isinstance(testSet_newsid_to_tags_dict, dict):
        for newsid in testSet_newsid_to_tags_dict.keys():
            newsid_to_simVal_dict[newsid] = calc_simVal(userid_to_tags_dict[userid],
                                                        testSet_newsid_to_tags_dict[newsid])
    return newsid_to_simVal_dict  # 在content-based情形下, 是返回testSet中所有newsid:相似度的映射dict;


def collaborative_filtering_service(newsid_and_readtime_df, userid, L):
    """
    协同过滤推荐服务
    :param newsid_and_readtime_df:
    :param userid:
    :param L: 推荐列表长度
    :return: a list of recommended newsid
    """
    if userid in trainSet_userid_to_groupTagset_dict.keys():
        newsid_to_simVal_dict = calc_simVal_between_givenUser_and_eachTestSetNews(userid,
                                                                                  trainSet_userid_to_groupTagset_dict,
                                                                                  testSet_newsid_to_tags_dict)
        return get_topK_key(newsid_to_simVal_dict, L)
    else:
        return get_hot_news_list(newsid_and_readtime_df, k=L, time_end=testSet_userid_to_lastReadTime_dict[userid],
                                 days=1)


def content_based_service(newsid_and_readtime_df, userid, L):
    """
    基于内容推荐服务
    :param newsid_and_readtime_df:
    :param userid:
    :param L: 推荐列表长度
    :return: a list of recommended newsid
    """
    if userid in trainSet_userid_to_tagset_dict.keys():  # 如果用户id能找得到其的过去兴趣标签(非冷启动的)
        newsid_to_simVal_dict = calc_simVal_between_givenUser_and_eachTestSetNews(userid,
                                                                                  trainSet_userid_to_tagset_dict,
                                                                                  testSet_newsid_to_tags_dict)
        return get_topK_key(newsid_to_simVal_dict, L)  # 返回相似度最高的5个
    else:  # 用户最后阅读时间之前一天的新闻top5
        return get_hot_news_list(newsid_and_readtime_df, k=L, time_end=testSet_userid_to_lastReadTime_dict[userid],
                                 days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="to parse news recommendation input")
    parser.add_argument("-m", "--method", help="Recommendation Method: cb(ContentBased), cf(Collaborative Filtering)")
    parser.add_argument("-i", "--userid", type=int, help="The user's ID to be recommended, such as 3506171 and 436906")
    args = parser.parse_args()

    user_list = list(np.load("../data/wholeSet_userid_set.npy").item())

    if (args.method not in ['cb', 'cf']) or (args.userid not in user_list):
        print(parser.print_help())
        sys.exit()

    if args.method == "cf":
        recommended_newsid_list = collaborative_filtering_service(wholeSet_newsid_and_readtime_df, args.userid, L)
    elif args.method == "cb":
        recommended_newsid_list = content_based_service(wholeSet_newsid_and_readtime_df, args.userid, L)

    print("向用户`{id}`推荐如下{L}篇新闻".format(id=args.userid, L=5))
    template_str = '{num}.新闻id:`{id}`\t标题:{title}'
    for index, newsid in enumerate(recommended_newsid_list):
        output_str = template_str.format(num=index, id=newsid, title=wholeSet_newsid_to_newstitle_dict[newsid])
        print(output_str)
        logging.debug(output_str)
