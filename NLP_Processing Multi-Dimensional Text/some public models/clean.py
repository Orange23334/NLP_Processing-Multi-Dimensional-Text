import os
from turtle import pd
# from bert_run import run

import jieba
import re
import pandas as pd

def clean(word):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    line = re.sub(r, '', word)
    str = re.sub('[^\u4e00-\u9fa5]+', '', line)
    print(str)
    if len(str) > 3:
        return str
    else :
        return None

def readjson(i):
    # filePath = r'G:\毕业设计项目\数据集\躁郁症数据集\new'
    # file_list = os.listdir(filePath)
    # newlist = []
    # import random
    # for file in file_list:
    #     # 读取原文件名
    #     i = file
    #     # 去除后缀
    #     j = os.path.splitext(file)[0]
    #     # print(i)
    newlist=[]
    newpath = r'G:\毕业设计项目\数据集\躁郁症数据集\new\{}'.format(i)
    print(newpath)
    import json
    import json
    with open(newpath, 'r',encoding='utf-8') as file:
        str = file.read()
        data = json.loads(str)
    for x in data["weibo"]:
        y=clean(x["content"])
        if y:
            newlist.append(y)
    return newlist

                    
def wenjian():
    filePath = r'G:\毕业设计项目\数据集\躁郁症数据集\weibo'
    file_list = os.listdir(filePath)
    newlist = []
    for file in file_list:
        # 读取原文件名
        i = file
        # 去除后缀
        j = os.path.splitext(file)[0]
        #print(i)
        newpath= r'G:\毕业设计项目\数据集\躁郁症数据集\weibo\{}'.format(i)
        file_list2 = os.listdir(newpath)
        for filex in  file_list2:
            ii=filex
            jj = os.path.splitext(filex)[0]
            if 'json' in ii:
                import shutil
                shutil.move(r'G:\毕业设计项目\数据集\躁郁症数据集\weibo\{}\{}'.format(i,ii), r'G:\毕业设计项目\数据集\躁郁症数据集\new')
                newlist.append(ii)
            # newlist.append(ii[2])
    print(newlist)



#readjson()
# print(len(line))
# data = pd.read_csv('G:/毕业设计项目/数据集/躁郁症数据集/weibo/___小狗/7726954738.csv',usecols=[1])
# # print(data['微博正文'])
# savelist=[]
# for x in data['微博正文']:
#     word = x
#     r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
#
#     line = re.sub(r, '', word)
#     str = re.sub('[^\u4e00-\u9fa5]+', '', line)
#     print(str)
#     if len(str)>3:
#         savelist.append(str)
#
# df1 = pd.DataFrame(data=savelist,
#                       columns=['text'])
# df1.to_csv('result.csv',index=False)

#seg_list = jieba.cut(line, cut_all=False)


# def scan(sentence):
#     r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
#     # try:
#     #     f = open(self.path, "r", encoding='UTF-8')
#     # except Exception as err:
#     #     print(err)
#     # finally:
#     #     print("文件读取结束")
#     word_list = []
#     while True:
#         line = f.readline()
#         if line:
#             line = line.strip()
#             line = re.sub(r, '', line)
#             seg_list = jieba.cut(line, cut_all=False)
#             word_list.append(list(seg_list))
#         else:
#             break
#     f.close()
#     print(word_list)

