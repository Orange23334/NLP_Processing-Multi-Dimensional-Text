import h5py
import h5py as h5
from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self,verify=True):
        if verify:
            #躁郁症用户路径
            featurepath='./data/botfeature.h5'
            tweetpath='./data/bottweet.h5'
            self.lable=1
        else:
            #人类用户的路径
            featurepath = './data/botfeature.h5'
            tweetpath = './data/bottweet.h5'
            self.lable = 0
        tweetdata = h5.File(tweetpath, 'r')
        idlist=[key for key in tweetdata.keys()]
        tweetlist = []
        for key in tweetdata.keys():
            tweetlist.append(tweetdata[key][()])
        #print(len(tweetlist))
        featuredata = h5.File(featurepath, 'r')
        featurelist = []
        for key in idlist:
            data=featuredata[key][()].reshape(1,10)
            featurelist.append(data)
        #print(len(featurelist))
        self.len=len(tweetlist)
        self.tweet=tweetlist
        self.feature=featurelist
        # featuredata=h5.File(featurepath, 'w')
        # self.len=len(tweetdata)
        # self.x = torch.linspace(11, 20, 10)
        # self.y = torch.linspace(1, 10, 10)
        # self.len = len(self.x)

    def __getitem__(self, index):
        tweet=self.tweet[index]
        feature=self.feature[index]

        return tweet,feature,self.lable

    def __len__(self):
        return self.len






#
# for key in tweetdata.keys():
#     print(tweetdata[key])

