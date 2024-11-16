from dataset import MyDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
size=1
class DeBD(nn.Module):
    def __init__(self,batch_size=10,input_size=10,hidden_size=128,num_layer=1):
        super(DeBD,self).__init__()
        #cnn
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.cnn_con1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5,1024), stride=1, padding=(2,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #128*100*1
            #128*1*1
            nn.MaxPool2d(kernel_size=(100, 1))
            #
        )
        #lstm
        self.lstmlayer=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layer,batch_first=True,bidirectional=False)
        #self.lstm_fc1 = nn.Linear(hidden_size, 128)
        self.w = nn.Parameter(torch.ones(2))
        self.softmaxlayer=nn.Softmax(dim=1)
        self.f_fc = nn.Linear(128, 2)
    def forward(self,x,y):
        cnn_out=self.cnn_con1(x)
        cnn_out=cnn_out.reshape(self.batch_size, 128*1*1)
        #print(cnn_out)
        lstm_out,(hn,cn)=self.lstmlayer(y)
        #lstm_out=self.lstm_fc1(lstm_out)
        lstm_out=lstm_out.reshape(self.batch_size, self.hidden_size)
        #print(lstm_out)
        #lstm_out=lstm_out.reshape(self.batch_size, 128)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        out=cnn_out*w1+lstm_out*w2
        out=self.f_fc(out)
        #print(out)
        return out
#tain最终的，test也要这样操作
traindatabot=MyDataset()
print(traindatabot[1])
# traindatahuman=MyDataset(0)
# traindata=traindatahuman+traindatabot
# print(len(traindata))
#
# # testdatabot=MyDataset()[2:3]
# # testdatahuman=MyDataset(0)[2:3]
# # testdata=testdatabot+testdatahuman
# # print(len(testdata))
train_dataloader=DataLoader(traindatabot[1],batch_size=size)
print(train_dataloader[0])
# print(train_dataloader)
# #test_dataloader=DataLoader(testdata,batch_size=size)
# model=DeBD(batch_size=size)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
# # init loss function
# criterion = nn.CrossEntropyLoss()
#
# for tweet,feature,label in train_dataloader:
#     model.train()
#     print(tweet,feature,label)
#     logits = model(tweet,feature)
#     # logits = model(feature)
#     # calc loss
#     loss = criterion(logits, label)
    # zero grad
    # optimizer.zero_grad()
    # # back forward
    # loss.backward()
    # # keep grad
    # optimizer.step()
    # # predicting res
    # pred = logits.argmax(dim=1)
    # print('xxxx')