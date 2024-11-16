from word2vec import *

class CONFIG():
    update_w2v = True           # 是否在训练中更新w2v
    vocab_size = 49011          # 词汇量，与word2id中的词汇量一致
    n_class = 2                 # 分类数：分别为true和fake
    embedding_dim = 50          # 词向量维度
    drop_keep_prob = 0.5        # dropout层，参数keep的比例
    kernel_num = 64             # 卷积层filter的数量
    kernel_size = [3,4,5]       # 卷积核的尺寸
    pad_size = 50               #每句话处理成的长度(短填长切)
    word2id=get_word2id()
    word2vec = build_word2vec('./data/wiki_word2vec_50.bin', word2id)
    pretrained_embed = word2vec # 预训练的词嵌入模型
