import pandas as pd
import numpy as np
import csv
import re
import jieba
from collections import Counter
from torch.utils.data import TensorDataset,DataLoader

def get_word2id():
    id2word = pd.read_table('./temp/wordLabel.txt')
    words=id2word["word"]
    #生成词表word2id字典wordid
    word2id = {'_PAD_': 0}
    value=1
    for i in words:
        word2id[i]=value
        value=value+1
    return word2id
    

def cat_to_id(classes=None):

    """
    :param classes: 分类标签；默认为0:true, 1:fake
    :return: {分类标签：id}
    """

    if not classes:
        classes = ['0', '1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}

    return classes, cat2id


def load_corpus(path, word2id, max_sen_len=50):

    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """

    _, cat2id = cat_to_id()
    print(cat2id)
    contents, labels = [], []
    with open(path, 'r') as f:
        next(f)	#跳过头
        reader = csv.reader(f)
        print(type(reader))
        for row in reader:
            label=str(row[0])
            if label != '1':
                label = '0'
            else:
                label='1'
            #print(label)
            pattern = re.compile(r'[^\u4e00-\u9fa5]')
            chinese = re.sub(pattern, '', row[4])
            content = jieba.cut(chinese)
            content=[word2id.get(w,0) for w in content]
            content=content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)

    print('Total sample num：%d' % (len(labels)))
    print('class num：')

    for w in counter:
        print(w, counter[w])
    contents = np.asarray(contents)
    labels = np.array([cat2id[l] for l in labels])
    #print (labels)
    return contents, labels


if __name__ == '__main__':

    word2id=get_word2id()
    print('train set: ')
    train_contents, train_labels = load_corpus('./data/train.csv', word2id, max_sen_len=50)
    train_dataset =TensorDataset(torch.from_numpy(train_contents).type(torch.float),.from_numpy(train_labels).type(torch.long))
    print(train_dataset)
