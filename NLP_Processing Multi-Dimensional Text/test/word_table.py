'''
将训练数据使用jieba分词工具进行分词。并且剔除非中文的词，仅保留中文。
得到词表：
        词表的每一行的内容为：词的序号 词 
'''
import pandas as pd
import jieba
import re

class TokenizerTable():
    """
    构建分词表
    """
    def __init__(self, data):
        """
        :param data:传进去的评论列表
        """
        #set() 函数创建一个无序不重复元素集
        self.word_set = set()
        self.data = data

    def tokenizer(self):
        # 数据清理
        for seq in self.data:
            #print(seq)
            pattern = re.compile(r'[^\u4e00-\u9fa5]')
            chinese = re.sub(pattern, '', str(seq))
            #print(chinese)
            content = jieba.cut(chinese)
            self.word_set.update(content)
    
    def write_tokenizer_table(self, path='./temp/word.txt'):
        #idx = [i for i in range(len(self.word_set))]
        df = pd.DataFrame({
            'word': list(self.word_set)
        })
        df.to_csv(path,sep='\t')

if __name__ == '__main__':
    path = './data/all.csv'
    df = pd.read_csv(path)
    #df.iloc:选取第四列 to_list:将pandas数据帧的索引转换为一个列表
    data = df.iloc[:, 4].to_list()
    #print(data)
    x = TokenizerTable(data)
    x.tokenizer()
    x.write_tokenizer_table()
