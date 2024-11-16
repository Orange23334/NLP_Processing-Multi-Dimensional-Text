# 64 128 128
# 64 128 128
# 128 64 64
# 256 32 32
# 512 16 16
# 512 8  8

# 512
# 10
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import xgboost

class RESblock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(RESblock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RGAnet(nn.Module):
    def __init__(self, block=RESblock,num_blocks=[2, 2, 2, 2],input_size=12, hidden_size=20, num_layers=3, bidirectional=True, dropout_p=0.1,
                 gru_output=128):
        super(RGAnet, self).__init__()
        self.in_planes = 3
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(16384*128, 128)
        self.layer_normal = nn.LayerNorm(normalized_shape=[1, input_size])
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_directions = 2 if bidirectional else 1  # 2 BiGRU and 1 GRU

        # 输入维度、输出维度、层数、bidirectional用来说明是单向还是双向
        self.GRU_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True, bidirectional=True)

        # set gru fully connected layer
        self.gru_fc1 = nn.Linear(self.n_directions * hidden_size, gru_output)

        # set dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # set softmax layer
        self.softmax = nn.Softmax(dim=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __init__hidden(self, random=True):
        # to init hidden layer by random or zero
        h0 = torch.rand(self.num_layers * self.n_directions, self.batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.n_directions, self.batch_size, self.hidden_size).to(device)
        return h0, c0
    def forward(self,p_feature):
        out=self.layer1(p_feature)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.linear(out)
        h0, c0 = self.__init__hidden(random=True)
        # h0 = h0.to(device)
        # c0 = c0.to(device)
        out, hn = self.GRU_layer(out, (h0, c0))
        out = out[:, -1, :]
        out=self.dropout(out)
        out=self.gru_fc1(out)

class RESblock1(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(RESblock1, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RGAnet1(nn.Module):
    def __init__(self,block=RESblock1,planes=1,input_size=12, hidden_size=20, num_layers=3, bidirectional=True, dropout_p=0.1,gru_output=128
                 ):
        super(RGAnet1, self).__init__()
        self.in_planes = 1
        self.batch_size=10
        #resnet
        self.bnlayer=nn.BatchNorm1d(planes)
        self.block1=block(in_planes=1,planes=1)
        self.block2 = block(in_planes=1, planes=1)
        self.block3 = block(in_planes=1, planes=1)
        #bigru
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_directions = 2 if bidirectional else 1  # 2 BiGRU and 1 GRU
        self.GRU_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=bidirectional, dropout=dropout_p, batch_first=True)
        # set gru fully connected layer
        self.gru_fc1 = nn.Linear(hidden_size * self.n_directions, gru_output)
        # set dropout layer
        self.dropout = nn.Dropout(p=dropout_p)
        # set softmax layer
        self.softmax = nn.Softmax(dim=1)
        #Attention
        self.w_omega=nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        self.decoder = nn.Linear(2 * hidden_size, 4)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def __init__hidden(self, random=True):
        # to init hidden layer by random or zero
        if random:
            hidden = torch.rand(self.n_layers * self.n_directions, self.batch_size, self.hidden_size)
        else:
            hidden = torch.zeros(self.n_layers * self.n_directions, self.batch_size, self.hidden_size)
        return hidden

    def forward(self,x):
        out=self.bnlayer(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        print('yyyy',out.size())
        hidden= self.__init__hidden(random=True)
        # h0 = h0.to(device)
        # c0 = c0.to(device)
        out, hn = self.GRU_layer(out, hidden)
        #out = out[:, -1, :]
        print('xxx',out.size())
        #Attention 开始
        u = torch.tanh(torch.matmul(out, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = out * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        #out = self.decoder(feat)
        # out形状是(batch_size, 4)
        out = self.dropout(feat)
        print(out.size())
        out = self.gru_fc1(out)

        #out=self.softmax(out)
        return out

class DeBD(nn.Module):
    def __init__(self,batch_size=10,input_size=12,hidden_size=128,num_layer=1):
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


class MLP(nn.Module):
    def __init__(self, batch_size=10, input_size=12, hidden_size=128):
        super(MLP, self).__init__()
        self.batchsize=batch_size
        self.MLP_layer = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=2)
        )

    def forward(self,x):
        print('saxax',x.shape)
        out=x.reshape(self.batchsize,12)
        #print('ccccc',out.shape)
        out=self.MLP_layer(out)
        return out


def test():
    # net = RGAnet()
    # y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())
    # x=torch.randn(10,1,100,1024)
    z=torch.randn(10,1,12)
    # layer_normal = nn.LayerNorm(normalized_shape=[1, 12])
    # print(layer_normal(x).size())
    # x=out=F.normalize(x,dim=1)
    # print(x)
    #100,1,15
    # conv1=nn.Conv1d(in_channels=1,out_channels=10,kernel_size=5,padding=2)
    # y=conv1(x)
    # resblock=RESblock1(in_planes=1,planes=1)
    # y=resblock(x)
    # print(y.size())
    # net=DeBD()
    # y=net(x,z)
    # net=MLP()
    # y=net(z)
    # print(y.size())
    #print(y)
    # x,y=shap.datasets.boston()
    # print(len(x))
    # print(type(x))
    # model=xgboost.XGBClassifier().fit(x,y)
    # ex=shap.Explainer(model)
    # shap_values=ex(x)
    # #print(shap_values.)
    # # shap.DeepExplainer
    # shap.waterfall_plot(shap_values)
#test()

def tteet():
    import xgboost
    import shap.plots
    z = torch.randn(10, 1, 12)
    y=torch.zeros(10).long()
    x=torch.randn(10, 1, 12)
    xy=torch.zeros(10).long()
    model=MLP()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    # init loss function
    criterion = nn.CrossEntropyLoss()
    for i in range(2):
        model.train()
        if i==0:
            logits = model(z.float())
            # logits = model(feature)

            # calc loss
            loss = criterion(logits,y)

            # zero grad
            optimizer.zero_grad()

            # back forward
            loss.backward()

            # keep grad
            optimizer.step()

            # predicting res
            pred = logits.argmax(dim=1)
            #print(pred)
        else:
            logits = model(x.float())
            # logits = model(feature)

            # calc loss
            loss = criterion(logits, xy)

            # zero grad
            optimizer.zero_grad()

            # back forward
            loss.backward()

            # keep grad
            optimizer.step()

            # predicting res
            pred = logits.argmax(dim=1)
            print(pred)
    import sklearn
    import matplotlib
    #train an XGBoost model
    # X, y = shap.datasets.boston()
    # model = xgboost.XGBRegressor().fit(X, y)
    #x=
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.
    # )
    new_x=torch.randn(10,1,12)
    xl=[new_x]
    # out=model(*xl)
    # print(out.shape)
    explainerk=shap.DeepExplainer(model,xl)
    # #explainer = shap.Explainer(model)
    #
    shap_values = explainerk.shap_values(xl)
    # #shap.plots.waterfall(shap_values[1])
    # # visualize the first prediction's explanation
    # print(shap_values[0])
    # testx = [xx for xx in new_x]
    # z=[1,2]
    # #*testx
    # print(*z)
    #shap.waterfall_plot(shap_values[0])
    # shap.plots.force(shap_values[0],feature_names=['1','2','3'])
    # shap.plots.scatter(shap_values[:, "RM"], color=shap_values)
    # shap.plots.beeswarm(shap_values)
    # print(shap_values.feature_names)
tteet()

#test()


