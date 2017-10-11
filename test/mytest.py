# -*- coding: utf-8 -*-
# @Time    : 23/09/2017 10:44
# @Author  : Cheney
# @Software: PyCharm

import numpy as np
import jieba.analyse

def test1():
    # dict排序的方法测试
    d1 = {"BJ": 22, "SH": 24, "GZ":14}
    d2 = sorted(d1.items())  #
    # print(d1)
    # print(d2)
    # dict用value逆序排列, 返回key的列表
    l3 = sorted(d1.items(), key=lambda d:d[1], reverse=True)
    print(l3)
    print([t[0] for t in l3])

def test2():
    d1 = {"NYC": 8, "LA": 6, "PHL": 2}
    np.save('../data/test2.npy', d1)

def test3():
    d = np.load('../data/test2.npy').item()
    print(d)
    #测试结论: np.save(path), np.load(path).item()是一种十分好用的python数据结构序列化存储和读出的方法

def test4():
    import jieba
    text = "10时45分，北京市公安局轨道建设分局筹备组支援民警张新和天安门地区分局巡警二大队副大队长汪湘江成功处置一起突发事件。当日，2人获市局嘉奖，公安部副部长、北京市委常委、公安局局长傅政华随即为他们颁发嘉奖证书。"
    res = jieba.analyse.extract_tags(text, topK=5)
    print(res)
    print(type(res))
    #测试结论: jieba可以直接算topK, 应该是以中文大型语料库作为背景

def test5():
    d1 = np.load("../data/testSet_newsid_to_tags_table.npy").item()
    print(d1[100651212])
    #测试结论: 生成成功

def test6():
    d1 = np.load("../data/trainSet_userid_to_tagset_table.npy").item()
    print(d1[52550])
    #测试结论: 生成成功

def test7():
    l_freq = np.load("../data/trainSet_freqUser_set.npy").item()
    l_old = np.load("../data/testSet_oldUser_set.npy").item()
    print(l_freq)
    print(len(l_freq))
    print(type(l_freq))
    print("oldUserList:", l_old)
    print(len(l_old))
    print(type(l_old))
    print(list(l_freq))
    #测试结论: 生成成功, 需要借用set或者dict才能使用npy来保存, 这个是个trick

def test8():
    user_tags = np.load("../data/trainSet_userid_to_tagset_table.npy").item()
    print(user_tags)
    print(type(user_tags))

def test9():
    groupTags = np.load("../data/trainSet_userid_to_groupTagset_table.npy").item()
    print(groupTags)
    print(type(groupTags))

def test10():
    userlist = list(np.load("../data/wholeSet_userid_set.npy").item())
    print(userlist)
    print(len(userlist))
    print(type(userlist))

def test11():
    from src.Engine import get_topK_key
    dic = {"BJ": 22, "SH":24, "SZ":12, "TY": 30}
    l = get_topK_key(dic, 1)
    print(l)
    #测试成功, 原来是sorted(d.items(), key=lambda d:d[1], reverse=True)我漏掉了items()这个调用

def test12():
    trainSet_dict = np.load("../data/trainSet_userid_to_tagset_table.npy").item()
    print(trainSet_dict)
    print(len(trainSet_dict))
    #测试成功, 验证的用户数字和DataFactory中的方法验证一致

def test13():
    testSet_actualReadNews = np.load("../data/testSet_userid_to_actualReadNewsid_table.npy").item()
    print(testSet_actualReadNews)
    print(len(testSet_actualReadNews))

def test14():
    users = list(np.load("../data/testSet_useridSet.npy").item())
    print(users)
    print(len(users))



if __name__ == "__main__":
    test7()
