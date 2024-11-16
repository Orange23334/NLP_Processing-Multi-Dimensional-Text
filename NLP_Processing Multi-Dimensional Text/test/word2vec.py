import gensim
import pandas as pd
import numpy as np
import csv
import re
import jieba


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

def build_word2vec(fname, word2id, save_to_path='./temp/w2c.txt'):
    """
    fname: 预训练的word2vec. ./word2vec_50.bin
    word2id: 语料文本中包含的词汇集. ./wordLabel.txt
    save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = len(open('./temp/wordLabel.txt','rU').readlines())
    #读取词向量模型
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
            #print(wordid[word])
        except KeyError:
            pass
    
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                #print(vec)
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

if __name__ == '__main__':
    word2id=get_word2id()
    #生成词向量 id: word2vec
    word2vec = build_word2vec('./data/wiki_word2vec_50.bin', word2id)
    
    print(word2vec)
    
    #assert word2vec.shape == (58954, 50)

