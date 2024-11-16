import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Embedder:
    def __init__(self, pooling_pattern='MAX_POOLING', pooling_layer=-1, model_dir='./resources/'):
        """
        :param pooling_pattern: choose pooling pattern, default MAX POOLING
        :param pooling_layer: chose which layer of hidden states to be the output text embedding matrix,
        default -1 which means use the last second layer output
        :param model_dir: default None, choose your own model
        """

        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__pooling_pattern = pooling_pattern
        self.__pooling_layer = pooling_layer
        self.__model_dir = model_dir

        # init model and tokenizer
        self.__tokenizer = BertTokenizer.from_pretrained(self.__model_dir, tokenize_chinese_chars=True)
        self.__model = BertModel.from_pretrained(self.__model_dir).to(self.__device)


    def __call__(self, text, max_length=200, truncation=True, padding=False, add_special_tokens=True):
        """
        :param text: list, input text content
        :param max_length: the max length to truncate
        :param padding: see doc: https://huggingface.co/transformers/main_classes/tokenizer.html
        :param truncation: ...
        :param add_special_tokens:
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to add the special tokens associated with the corresponding model.
        :return: word embedding list
        """
        word_embedding_list = []
        for _ in text:
            input_ids = torch.tensor(self.__tokenizer.encode(_, max_length=max_length, truncation=truncation,
                                                             padding=padding, add_special_tokens=add_special_tokens)).unsqueeze(0).to(self.__device)  # Batch size 1
            with torch.no_grad():
                outputs = self.__model(input_ids)
            origin_tensor = outputs[0]  # 选择默认的tensor输出
            origin_tensor_shape = origin_tensor.shape  # shape的第一维是channel，此处channel为1，但是也需要指明
            kernel_size = (origin_tensor_shape[1], 1)  # 指定Pooling的核，此处以一列为核，进行Pooling
            pooling = nn.MaxPool2d(kernel_size=kernel_size)  # 这里要做一个pooling，默认使用最大池化，后续可以拓展池化方法
            output_tensor = pooling(origin_tensor)  # output 1024*1
            word_embedding_list.append(output_tensor)

        tensor_list = self.__reshape_tensor(word_embedding_list)
        return tensor_list

    def __reshape_tensor(self, word_embedding_list):
        tensor = word_embedding_list[0][0]
        for index in range(1, len(word_embedding_list)):
            tensor = torch.cat((tensor, word_embedding_list[index][0]), 0)
        return tensor.cpu().detach().numpy()


def main():
    import time
    text_list = ['牛年大吉🎉 西安·曲江六号']
    start = time.time()
    embedder = Embedder()
    print(embedder(text_list))
    # end = time.time()
    # print(end - start)
    # print((end - start) / 60)


if __name__ == '__main__':
    main()