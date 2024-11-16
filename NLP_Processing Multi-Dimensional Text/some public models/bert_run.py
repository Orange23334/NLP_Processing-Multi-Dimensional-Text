import os
import time
import json
import h5py as h5
import numpy as np
from bert_embedder import Embedder
from clean import readjson


def padding(tensor, row=100):
    """
    :param tensor: np array,
    :param row: how many row the matrix owns
    :return: padding tensor
    """
    shape = tensor.shape
    tensor_row, tensor_line = shape
    if row == tensor_row:
        return tensor
    if row > tensor_row:
        # 获取缺的行数
        row = row - tensor_row
        temp = [0]*tensor_line
        temp = [temp for i in range(row)]
        # 生产补全的array
        np_arr = np.array(temp)
        # 将两个array拼接在一起
        tensor = np.vstack((tensor, np_arr))
        return tensor
    # 将增补的np_arr直接拼接到tensor后，满足至少为 row*tensor.shape.line (eg. 100*1024)


def get_text(tweet_list: list):
    return [tweet for tweet in tweet_list if tweet != '' and tweet is not None]

def runnew(dst_path='./data/',tweet_num=100):
    start = time.time()

    print("start.\n")

    embdder = Embedder()

    print('reading source file...\n')

    filePath = r'G:\毕业设计项目\数据集\躁郁症数据集\new'
    file_list = os.listdir(filePath)
    newlist = []
    import random
    counter = 0
    embedding_path = dst_path + f'bottweet.h5'
    f= h5.File(embedding_path, 'w')
    for file in file_list:
        # 读取原文件名
        i = file
        # 去除后缀
        j = os.path.splitext(file)[0]
        print('start embedding...\n')



        # for row in tweet_arr:
        # 获取所有推文 list
        tweet_list = readjson(i)
        if len(tweet_list)<=50:
            continue
        # 清洗掉空的tweet
        tweet_list = get_text(tweet_list)

        # 推文数量，如果推文数量小于目标推文数量（用户推文矩阵的行数），则需要padding
        tweet_len = len(tweet_list)
        if tweet_len < tweet_num:
            tensor_arr = embdder(tweet_list)
            tensor_arr = padding(tensor_arr)
        else:
            tensor_arr = embdder(tweet_list[: tweet_num])

        # 塑形
        # row, line = tensor_arr.shape
        tensor_arr = tensor_arr.reshape(1, tweet_num, 1024)

        # 保存第一次生成的tensor
        # if flag is True:
        embedding_arr = tensor_arr
    #     flag = False
    # else:
    #     embedding_arr = np.concatenate((embedding_arr, tensor_arr))
        print(embedding_arr)
        # break
        counter += 1
        print(f'start write files{i}...\n')


        dataset_name=j
        print(j)
        f.create_dataset(dataset_name, data=embedding_arr)
        if counter==3:
            break
        print(f'{counter} user(s) finished, time used {time.time() - start} s, {(time.time() - start) / 60} min...')


    print(f'embedding finished, time used {time.time() - start} s, {(time.time() - start) / 60} min...')

    # 创建对应推文矩阵向量信息的辅助文件
    # info_arr = np.column_stack((uid_arr, type_arr))
    # #
    # # print(info_arr[0],info_arr[1],info_arr[2])
    # # 创建输出路径
    # embedding_path = dst_path + f'{i}tweet.h5'
    # info_arr_path = dst_path + f'{i}info.h5'
    # print(embedding_arr.shape)
    # print(f'start write files{i}...\n')
    #
    # # 存入文件输出
    # with h5.File(info_arr_path, 'w') as f:
    #     f.create_dataset(dataset_name, data=info_arr)
    #
    # with h5.File(embedding_path, 'w') as f:
    #     f.create_dataset(dataset_name, data=embedding_arr)
    #
    # print(f'Program finished, {time.time() - start} s, {(time.time() - start) / 60} min.\n')

def readh5():
    import h5py
    with h5py.File('./data/botfeature.h5', "r") as f:
        for key in f.keys():
            print(f[key])


def run(src_path='./data/src/bot/bot.h5', dst_path='./data/dst/bot', dataset_name='bot_embeddings', tweet_num=100):

    start = time.time()

    print("start.\n")

    embdder = Embedder()

    print('reading source file...\n')
    for i in range(10):
        with h5.File(src_path, 'r') as input_file:
            # 读取key
            key = list(input_file.keys())[0]
            # 读取所有信息
            src_arr = input_file[key][()][i*200:i+200]
            # 读取uid信息
            uid_arr = src_arr[:, 0]
            # 读取用户类型arr
            type_arr = src_arr[:, 1]
            # 读取tweet信息
            tweet_arr = src_arr[:, 2]
            # set flag
            flag = True

            print('start embedding...\n')

            counter = 0

            for row in tweet_arr:
                # 获取所有推文 list
                tweet_list = json.loads(row)

                # 清洗掉空的tweet
                tweet_list = get_text(tweet_list)

                # 推文数量，如果推文数量小于目标推文数量（用户推文矩阵的行数），则需要padding
                tweet_len = len(tweet_list)
                if tweet_len < tweet_num:
                    tensor_arr = embdder(tweet_list)
                    tensor_arr = padding(tensor_arr)
                else:
                    tensor_arr = embdder(tweet_list[: tweet_num])

                # 塑形
                # row, line = tensor_arr.shape
                tensor_arr = tensor_arr.reshape(1, tweet_num, 1024)

                # 保存第一次生成的tensor
                if flag is True:
                    embedding_arr = tensor_arr
                    flag = False
                else:
                    embedding_arr = np.concatenate((embedding_arr, tensor_arr))

                counter += 1
                print(f'{counter} user(s) finished, time used {time.time() - start} s, {(time.time() - start) / 60} min...')

            print(f'embedding finished, time used {time.time() - start} s, {(time.time() - start) / 60} min...')

            # 创建对应推文矩阵向量信息的辅助文件
            info_arr = np.column_stack((uid_arr, type_arr))
            #
            # print(info_arr[0],info_arr[1],info_arr[2])
            # 创建输出路径
            embedding_path = dst_path + f'{i}tweet.h5'
            info_arr_path = dst_path + f'{i}info.h5'
            print(embedding_arr.shape)
            print(f'start write files{i}...\n')

            # 存入文件输出
            with h5.File(info_arr_path, 'w') as f:
                f.create_dataset(dataset_name, data=info_arr)

            with h5.File(embedding_path, 'w') as f:
                f.create_dataset(dataset_name, data=embedding_arr)

            print(f'Program finished, {time.time() - start} s, {(time.time() - start) / 60} min.\n')

def test_file():
    src = './data/src/bot/bot.h5'
    dst = './data/dst/bot/'
    dataset_name = 'bot_embeddings'
    run(src_path=src, dst_path=dst, dataset_name=dataset_name)
def test_new():
    src = './dataset/test/bot_test.h5'
    dst = './data/test/'
    dataset_name = 'bot_embeddings'
    run(src_path=src, dst_path=dst, dataset_name=dataset_name)

def task():
    for i in range(5):
        src = f'./data/src/bot/bot_{i}.h5'
        dst = f'./data/dst/bot/{i}/'
        dataset_name = f'bot_{i}'
        run(src_path=src, dst_path=dst, dataset_name=dataset_name)


def task_1():
    src = f'./data/src/bot/bot_{0}.h5'
    dst = f'./data/dst/bot/{0}/'
    dataset_name = f'bot_{0}'
    run(src_path=src, dst_path=dst, dataset_name=dataset_name)


def task_2():
    src = f'./data/src/human/human_{0}.h5'
    dst = f'./data/dst/human/{0}/'
    dataset_name = f'human_{0}'
    run(src_path=src, dst_path=dst, dataset_name=dataset_name)

def task_new():
    # for i in range(5):
    src = f'./dataset/bot/bot_{i}.h5'
    dst = f'./data/bot/{i}/'
    dataset_name = f'bot_{i}'
    run(src_path=src, dst_path=dst, dataset_name=dataset_name)
        # src = f'./dataset/human/human_{i}.h5'
        # dst = f'./data/human/{i}/'
        # dataset_name = f'human_{i}'
        # run(src_path=src, dst_path=dst, dataset_name=dataset_name)

if __name__ == '__main__':
    #task_2()
    # for i in range(5,10):
    #     print(i)
    #test_new()
    runnew()
    #readh5()