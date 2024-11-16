import sys
sys.path.append('/users/yangfei/desktop/test/')
from config import *
from model.TextCNN import *
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *


config = CONFIG()          # 配置模型参数
learning_rate = 0.001      # 学习率
BATCH_SIZE = 50            # 训练批量
EPOCHS = 10                # 训练轮数
model_path = None          # 预训练模型路径
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#模型确定
model = TextCNN(config)


if model_path:
    model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# 设置损失函数
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5)

def train(dataloader,epoch):
    # 定义训练过程
    train_loss,train_acc = 0.0,0.0
    count, correct = 0,0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)
        
        if (batch_idx+1) % 100 == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                                                                           epoch, batch_idx * len(x), len(dataloader.dataset),
                                                                           100. * batch_idx / len(dataloader), loss.item()))

    train_loss *= BATCH_SIZE
    train_loss /= len(dataloader.dataset)
    train_acc = correct/count
    print('\ntrain epoch: {}\taverage loss: {:.6f}\taccuracy:{:.4f}%\n'.format(epoch,train_loss,100.*train_acc))
    scheduler.step()
    
    return train_loss,train_acc


def validation(dataloader,epoch):
    model.eval()
    # 验证过程
    val_loss,val_acc = 0.0,0.0
    count, correct = 0,0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)
    
    val_loss *= BATCH_SIZE
    val_loss /= len(dataloader.dataset)
    val_acc = correct/count
    # 打印准确率
    print('validation:train epoch: {}\taverage loss: {:.6f}\t accuracy:{:.2f}%\n'.format(epoch,val_loss,100*val_acc))
    
    return val_loss,val_acc

model_pth = './model/TextRNN_Att/model_' + str(time.time()) + '.pth'
torch.save(model.state_dict(), model_pth)

def test(dataloader):
    model.eval()
    model.to(DEVICE)
    
    # 测试过程
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)
    
    # 打印准确率
    print('test accuracy:{:.2f}%.'.format(100*correct/count))
