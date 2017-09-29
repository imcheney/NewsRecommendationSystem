# -*- coding: utf-8 -*-
# @Time    : 24/09/2017 10:22
# @Author  : Cheney
# @Software: PyCharm
import jieba.analyse
import pandas as pd
import numpy as np
import time
import os

from collections import Counter


def get_basic_data(filename, sep, names, septime="2014-03-21 00:00:00"):
    '''
    从filename获取数据去重之后返回训练集和测试集
    :param filename: 数据集文件名
    :param sep: 数据之间的分隔符
    :param headers: headers
    :param names: 数据列名
    :param septime: 分隔训练集和测试集的时间界
    :return: 整个数据集 训练集 测试集合 pandas.DataFrame格式
    '''
    raw_data = pd.read_table(filename, sep=sep, header=None, names=names).dropna(
        how='any')  # drop知乎从116225降低到102204条浏览记录
    read_times = raw_data['read_time']
    sep_time = "2014-03-20 23:59:00"
    time_array = time.strptime(sep_time, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(time_array))

    index_before_sep_time = read_times.index[read_times < timestamp]  # 根据浏览时间对用户浏览数据进行划分, 分为训练集和测试集
    index_after_sep_time = read_times.index[read_times >= timestamp]

    train_data = raw_data.drop(index_after_sep_time)
    test_data = raw_data.drop(index_before_sep_time)

    return raw_data, train_data, test_data


def create_testSet_userid_to_lastReadTime_table(filename, test_data):
    '''
    返回测试集中用户id和最后一次阅读时间组成的表
    :param filename:
    :param test_data:
    :return:
    '''
    userid_readTime = test_data.loc[:, ['user_id', 'read_time']]
    userid_to_lastReadTime_dict = {}
    for userid, readtime in userid_readTime.values:
        if not userid_to_lastReadTime_dict.__contains__(userid) or readtime > userid_to_lastReadTime_dict[userid]:
            userid_to_lastReadTime_dict[userid] = readtime
    np.save(filename, userid_to_lastReadTime_dict)


def create_wholeSet_newsid_and_readTime_table(filename, raw_data):
    """
    生成包含newsid和readTime两个字段的df table
    :param filename:
    :param raw_data:
    :return:
    """
    id_time = raw_data.loc[:, ['news_id', 'read_time']]
    id_time.to_csv(filename)


def create_wholeSet_newsid_to_newstitle_table(filename, raw_data):
    """
    创建全集中新闻id与title之间映射的dict, 并持久化到文件中
    :param raw_data:
    :return:
    """
    newsid_title_list = raw_data.loc[:, ['news_id', 'news_title']].drop_duplicates().values.tolist()  # df-->list
    id_to_title_dict = {}
    for tuple in newsid_title_list:
        id_to_title_dict[tuple[0]] = tuple[1]  # dict[newsid] = news_title, 建立id --> title映射
    np.save(filename, id_to_title_dict)  # output #01, 注意这个id-->title映射是对所有的raw_data的


def create_trainSet_userid_to_tagset_table(filename, train_data):
    '''
    返回每个用户的读过的新闻中最重要的10个关键词(运用TF-IDF分析)
    TODO: 这个算法存在着严重的大数吃小数的问题, 长文章的内容会明显的压倒了短文章的内容, 应当提高标题文本的权重
    :param filename:
    :param train_data:
    :return:
    '''
    userid_and_newsContent_df = train_data.loc[:, ['user_id', 'news_content']]
    newsContent_groupby_userid_df = userid_and_newsContent_df.groupby('user_id')
    userid_to_tagset_dict = {}
    for userid, df in newsContent_groupby_userid_df:
        strs = [content for id, content in df.values]
        strs = '.'.join(strs)  # 把读过的新闻的内容用"."来进行拼接
        features = set(jieba.analyse.extract_tags(strs, topK=10))  # feature is a tag set
        userid_to_tagset_dict[userid] = features
    np.save(filename, userid_to_tagset_dict)


# 临时变量区域 TODO: 后期要做响应修改

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


def findFreqUserAndOldUserAndSaveThem(filepath_freq, filepath_old, train_data, test_data):
    """
    把训练集中的用户浏览记录用dict进行wordcount, 取出排名前100的用户
    :param filepath:
    :param train_data:
    """
    d = {}
    for index, row in train_data.iterrows():
        d[row['user_id']] = d.get(row['user_id'], 0) + 1
    countList = sorted(d.items(), key=lambda d: d[1], reverse=True)
    userid_count_top100_List = countList[0:100]
    freqUserList = [userid for userid, count in userid_count_top100_List]
    np.save(filepath_freq, set(freqUserList))
    oldUserList = []
    for index, row in test_data.iterrows():
        if row['user_id'] in d:
            oldUserList.append(row['user_id'])
    np.save(filepath_old, set(oldUserList))


def create_trainSet_userid_to_groupTagset_table(filepath, user_tags, oldUserList, freqUserList,
                                                threshold=4):  # TODO: 此处计算群体tags的算法有待商榷...存在问题
    """
    求出训练集中每个userid对应的组tagset
    :param filepath:
    :param user_tags:
    :param oldUserList:
    :param freqUserList:
    :param threshold:
    :return:
    """
    groupTagset = user_tags.copy()  # 应当采用深拷贝
    for oldUser in oldUserList:  # 对那些老用户
        similarFreqUserCount = 0
        for freqUser in freqUserList:  # 对每个高频用户中的热门用户
            print(type(user_tags[oldUser]))
            print(type(user_tags[freqUser]))
            sim = calc_simVal(user_tags[oldUser], user_tags[freqUser])  # 计算老用户与热门用户的相似度
            if sim >= threshold:  # threshold值是临界值
                similarFreqUserCount += 1  # count值, 当前老用户的同好热门用户数目
                groupTagset[oldUser] = set(list(groupTagset[oldUser]) + list(user_tags[freqUser]))  # 把相似的热门用户并进来
    np.save(filepath, groupTagset)


def create_testSet_newsid_to_tags_table(filename, data):
    '''
    计算新闻的TFIDF值,返回每个新闻值最大的前10个关键词
    :param data:  The DataFrame where id and content exist
    :param id: The id of news
    :param content: The content of news
    :return: The frequence dict of each news
    '''
    id = 'news_id'
    content = 'news_content'
    newsid_newscontent_df = data.loc[:, [id, content]].drop_duplicates().values  # 去重, 去掉第一列序号列
    newsid_to_tags_dict = {}
    for id, content in newsid_newscontent_df:
        newsid_to_tags_dict[id] = set(jieba.analyse.extract_tags(content, topK=10))  # 建立id-->top10 keyword set的映射
    np.save(filename, newsid_to_tags_dict)


def create_wholeSet_useridSet(filename, raw_data):
    """
    生成一份不重复的用户idset
    :param filename:
    :param raw_data:
    :return:
    """
    if isinstance(raw_data, pd.DataFrame):
        userid_df = raw_data['user_id']
        s1 = userid_df.values
        s = set(list(s1))
        np.save(filename, s)


def generate_all_file():
    # main函数中应该写如何生成这些文件的基本逻辑, 也就是先调用谁再调用谁, 并写上持久化文件的filepath
    sep = '\t'
    names = ['user_id', 'news_id', 'read_time', 'news_title', 'news_content', 'news_publi_time']
    raw_data, train_data, test_data = get_basic_data("../data/raw_data.txt", sep, names)

    # 生成简单系列文件
    create_testSet_userid_to_lastReadTime_table("../data/testSet_userid_to_lastReadTime_table.npy", test_data)
    create_wholeSet_newsid_and_readTime_table("../data/wholeSet_newsid_and_readtime_table.csv", raw_data)
    create_wholeSet_newsid_to_newstitle_table("../data/wholeSet_newsid_to_newstitle_table.npy", raw_data)
    create_wholeSet_useridSet('../data/wholeSet_userid_set.npy', raw_data)

    # 生成tags系列文件
    create_testSet_newsid_to_tags_table("../data/testSet_newsid_to_tags_table.npy", test_data)
    create_trainSet_userid_to_tagset_table("../data/trainSet_userid_to_tagset_table.npy", train_data)

    findFreqUserAndOldUserAndSaveThem("../data/trainSet_freqUser_set.npy", "../data/testSet_oldUser_set.npy",
                                      train_data, test_data)
    user_tags = np.load("../data/trainSet_userid_to_tagset_table.npy").item()
    oldUserList = list(np.load("../data/testSet_oldUser_set.npy").item())
    freqUserList = list(np.load("../data/trainSet_freqUser_set.npy").item())
    create_trainSet_userid_to_groupTagset_table("../data/trainSet_userid_to_groupTagset_table.npy", user_tags,
                                                oldUserList,
                                                freqUserList, 4)


if __name__ == "__main__":
    #generate_all_file()
    print('job done!')
