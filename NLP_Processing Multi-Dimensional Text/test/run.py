import sys
sys.path.append('/users/yangfei/desktop/test/')
from word2vec import *
from load_data import*
import time
from torch.utils.data import TensorDataset,DataLoader
from train import *

#计算程序运行时间
begin_time=time.time()

#加载词表数据，词向量数据
word2id=get_word2id()
word2vec = build_word2vec('./data/wiki_word2vec_50.bin', word2id)

#训练集
print('train set: ')
train_contents, train_labels = load_corpus('./data/train.csv', word2id, max_sen_len=50)
train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float),
                              torch.from_numpy(train_labels).type(torch.long))
                              
train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,
                              shuffle = True, num_workers = 2)

#验证集
print('\nvalidation set: ')
val_contents, val_labels = load_corpus('./data/validation.csv', word2id, max_sen_len=50)
val_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.float),
                              torch.from_numpy(val_labels).type(torch.long))
val_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,
                              shuffle = True, num_workers = 2)

#测试集
print('\ntest set: ')
test_contents, test_labels = load_corpus('./data/test.csv', word2id, max_sen_len=50)
test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float), 
                              torch.from_numpy(test_labels).type(torch.long))
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, 
                              shuffle = True, num_workers = 2)


train_losses = []
train_acces = []
val_losses = []
val_acces = []

for epoch in range(1,EPOCHS+1):
    tr_loss,tr_acc = train(train_dataloader,epoch)
    val_loss,val_acc = validation(val_dataloader,epoch)
    train_losses.append(tr_loss)
    train_acces.append(tr_acc)
    val_losses.append(val_loss)
    val_acces.append(val_acc)
    

test(test_dataloader)
end_time=time.time()
run_time=end_time-begin_time
print('程序执行时间：',run_time)

