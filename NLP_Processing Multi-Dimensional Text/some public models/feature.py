import datetime
import os
import time
from datetime import datetime
import numpy as np
import torch
import h5py as h5


def getbotuser():
    filePath = r'G:\毕业设计项目\数据集\躁郁症数据集\new'
    file_list = os.listdir(filePath)
    newlist = []
    import random
    for file in file_list:
    # 读取原文件名
        i = file
    # 去除后缀
        j = os.path.splitext(file)[0]
    # print(i)

        newpath = r'G:/毕业设计项目/数据集/躁郁症数据集/new/{}'.format(i)
        #print(newpath)
        import json
        import json
        with open(newpath, 'r', encoding='utf-8') as f:
            str = f.read()
            data = json.loads(str)
            #print(data)
        newlist.append(data)
    return newlist

def getnamelist():
    x=getbotuser()
    namelist=[]
    for xx in x:
        namelist.append(xx['user']['id'])
    return namelist

def sextype(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id']==id:
            if xx["user"]=='男':
                return 1
            else:return 0

def tt(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M")
    return int(time.mktime(timeArray))

def getjiangetimelist(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            timelist=[tt(_["publish_time"]) for _ in xx["weibo"]]
            timelist.reverse()

    jiangelist=[]
    for i in range(len(timelist)-1):
        jiange=timelist[i+1]-timelist[i]
        jiangelist.append(jiange)
    print(jiangelist)
    return jiangelist

def gettimechange(id):
    jiangelist=getjiangetimelist(id)
    std = np.std(jiangelist)
    mean = np.mean(jiangelist)
    print(std/mean)
    return std/mean

def pinjunlike(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            likelist = [_["up_num"] for _ in xx["weibo"]]
            #timelist.reverse()
    mean = np.mean(likelist)
    return mean

def pingjunretweet(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            retweetlist = [_["retweet_num"] for _ in xx["weibo"]]
            #timelist.reverse()
    mean = np.mean(retweetlist)
    return mean

def pingjuncomment(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            commentlist = [_["comment_num"] for _ in xx["weibo"]]
            #timelist.reverse()
    mean = np.mean(commentlist)
    return mean
# print(pingjuncomment('5768148849'))
#深夜发文占比

def tweettimeper(id):
    x = getbotuser()
    late = '06:00'
    start = '23:00'
    for xx in x:
        if xx['user']['id'] == id:
            timelist = [_["publish_time"] for _ in xx["weibo"]]
    daynum=0
    for x in timelist:
        b = datetime.strptime(x, '%Y-%m-%d %H:%M').strftime('%H:%M')
        #print(b)
        if b>late and b<start:
            #print('xxx',b)
            daynum+=1
    print(daynum)
    return (len(timelist)-daynum)/len(timelist)
# getnamelist()

def shuangxiang(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            postlist = [_["content"] for _ in xx["weibo"]]
    count=0
    for xxx in postlist:
        if '双相情感障碍' in xxx:
            count+=1
    return count/len(postlist)

def numberweibo(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            return xx['user']['weibo_num']

def numberfollowing(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            return xx['user']['following']

def numberfollowers(id):
    x = getbotuser()
    for xx in x:
        if xx['user']['id'] == id:
            return xx['user']['followers']

def getfeature(embedding_path='./data/'):
    x=getnamelist()
    ff = h5.File(embedding_path+f'botfeature.h5', 'w')
    count=0
    for xx in x:
        a=sextype(xx)
        b=gettimechange(xx)
        c=pinjunlike(xx)
        d=pingjunretweet(xx)
        e=pingjuncomment(xx)
        f=tweettimeper(xx)
        g=shuangxiang(xx)
        h=numberweibo(xx)
        i=numberfollowing(xx)
        j=numberfollowers(xx)
        featurelist=[a,b,c,d,e,f,g,h,i,j]
        print(featurelist)
        tensorfea=torch.FloatTensor(featurelist)
        print(tensorfea)
        dataset_name = xx
        print(xx)
        ff.create_dataset(dataset_name, data=tensorfea)
        count+=1
        print(count)
        if count==10:
            break

getfeature()




