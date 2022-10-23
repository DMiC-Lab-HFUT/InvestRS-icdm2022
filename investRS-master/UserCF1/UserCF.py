# coding = utf-8

# 基于用户的协同过滤推荐算法实现
import random

import math
from operator import itemgetter
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class UserBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 100
        self.n_rec_movie = 10 #top50

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = %d' % self.n_rec_movie)


    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    # 计算用户之间的相似度
    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        user_set = set()
        for user, movies in self.trainSet.items():
            user_set.add(user)
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')
        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        unadded = set(range(len(user_set))) - set(self.user_sim_matrix.keys())
        # print(unadded)
        for user in unadded:
            self.user_sim_matrix.setdefault(user, {})
        print('Build user co-rated movies matrix success!')

        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')

    def get_train_info_train(self,path, filename):

        with open(path + filename) as file:
            next(file)
            lines = file.readlines()
            person_set = set()
            place_set = set()
            person_dict = {}
            for line in lines:
                triple = line.strip().split('\t')
                if int(triple[2]) == 8:
                    person_set.add(int(triple[1]))
                    place_set.add(int(triple[0]))
                    if int(triple[1]) not in person_dict.keys():
                        person_dict[int(triple[1])] = []
                        person_dict[int(triple[1])].append(int(triple[0]))
                    else:
                        person_dict[int(triple[1])].append(int(triple[0]))
        return  person_set,place_set, person_dict


    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainSet[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def get_train_info_train2(self,path, filename):

        with open(path + filename) as file:
            next(file)
            lines = file.readlines()

            person_dict = {}
            for line in lines:
                triple = line.strip().split('\t')
                if int(triple[2]) == 8:

                    if int(triple[0]) not in person_dict.keys():
                        person_dict[int(triple[0])] = []
                        person_dict[int(triple[0])].append(int(triple[1]))
                    else:
                        person_dict[int(triple[0])].append(int(triple[1]))
        return person_dict

    def get_train_person_range(self,train_place_dict, test_place_set):
        top_person_dict = dict(sorted(train_place_dict.items(), key=lambda x: len(x[-1]), reverse=True)[:10])
        place = set()
        for key, item_list in top_person_dict.items():
            for item in item_list:
                if item in test_place_set:
                    place.add(item)
        return place

    def calculate_score(self,user):
        rec_movies = self.recommend(user)
        return rec_movies
    def get_train_info(self,path,filename,company2id):
        train_company2place = {}
        with open(path + filename, encoding='utf-8') as file:
            next(file)
            lines = file.readlines()
            for line in lines:
                entity = line.strip().split(',')
                if entity[0] not in train_company2place:
                    train_company2place[int(entity[0])] = []
                    train_company2place[int(entity[0])].append(int(company2id[entity[1]]))
                else:
                    train_company2place[int(entity[0])].append(int(company2id[entity[1]]))
        return train_company2place

    def get_entity2id(self,path, filename):
        id2entity = {}
        with open(path + filename, encoding='utf-8') as file:
            next(file)
            lines = file.readlines()
            for line in lines:
                entity = line.strip().split('\t')
                id2entity[entity[0]] = int(entity[1])
        return id2entity
    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        hit_enhanced = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        user_degree = {}
        for user, items in self.testSet.items():
            degree = len(items)
            user_degree[user] = len(items)

        hit10_info = {}
        hit10_top10 = {}

        rank_all = 0
        count =0
        sss = 0
        path = os.getcwd()+'/data/'
        # print(path)
        company2id = self.get_entity2id(path, 'person2id.csv')
        train_company2place = self.get_train_info(path,'rating_train_static.csv',company2id)


        # train_person_set, train_place_set, train_person_dict= self.get_train_info_train('/home/yuxc/ZJW_beifen/ZJW','/benchmarks/top100/graph_simple.txt')
        # top_person_dict = dict(sorted(train_person_dict.items(), key=lambda x: len(x[-1]), reverse=True)[:700])

        # test_person_set, test_place_set, test_person_dict = self.get_train_info_train('/home/yuxc/ZJW_beifen/ZJW','/benchmarks/top100/test2id.txt')
        # train_place_dict = self.get_train_info_train2('/home/yuxc/ZJW_beifen/ZJW','/benchmarks/top100/graph_simple.txt')
        # train_range = self.get_train_person_range(train_place_dict, test_place_set)

        for i, user, in enumerate(self.testSet):
            # print(111)
        # for i, user, in enumerate(top_person_dict.keys()):

            test_movies = self.testSet.get(user, {})
            if user in self.trainSet.keys():

                rec_movies = self.recommend(user)
            else:
                continue

            if len(test_movies) == 0 or user_degree[user] < 10:
                continue
            hit10_info[int(user)] = []
            hit10_top10[int(user)] = []
            rank = 0
            sss+=1
            kkk=0
            p=0
            train_moves = train_company2place[user]
            for movie, w in rec_movies:
                if movie in train_moves:
                    continue
                if movie in test_movies:
                    hit10_info[int(user)].append(movie)
                    p+=1
                    hit += 1
                    rank+=kkk
                    rank_all+=kkk
                    print('=====kkk========',kkk)
                # if movie in test_movies or j:
                #     hit_enhanced += 1
                # all_rec_movies.add(movie)
            # if p!=0:
            #     print('=====rank========',rank/p)
            # count += rank/p
            rec_count += N
            test_count += len(test_movies)

        # with open('/home/yuxc/open-ke-pytorch/target_subgraph/xietong_hit10.txt','w') as f:
        #     str_hit = ''
        #     for key,item in hit10_info.items():
        #         str_hit+= str(key)+'\t'
        #         if len(item)!=0:
        #             for node in item:
        #                 str_hit+=str(node)+','
        #         else:
        #             str_hit+=str(0)+','
        #         str_hit+='\n'
        #     f.write(str_hit)
        # with open('/home/yuxc/open-ke-pytorch/target_subgraph/xietong_hit10_top10.txt','w') as f:
        #     str_rank = ''
        #     for key,item in hit10_top10.items():
        #         str_rank += str(key) + '\t'
        #         for node in item:
        #             str_rank += str(node) + ','
        #         str_rank += '\n'
        #     f.write(str_rank)
        # with open('/home/yuxc/open-ke-pytorch/target_subgraph/xietong_hit10_top10.txt','w') as f:
        #     str_rank = ''
        #     for key,item in hit10_top10.items():
        #         str_rank += str(key) + '\t'
        #         if len(item) != 0:
        #             for node in item:
        #                 str_rank += str(node) + ','
        #         else:
        #             str_rank+=str(0)+','
        #         str_rank += '\n'
        #     f.write(str_rank)

        print(rec_count)
        # print('==========rank==========',rank_all/hit)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        precision_enhanced = hit_enhanced / (1.0 * rec_count)
        recall_enhanced = hit_enhanced / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
        print('precisioin_ehhanced=%.4f\trecall_ehhanced=%.4f\tcoverage=%.4f' % (precision_enhanced, recall_enhanced, coverage))
        return hit10_info

    def load_csv(self, filename):

        df = pd.read_csv(filename, engine="python", error_bad_lines=False, encoding='utf-8')
        return df
    def load_csv2(self, filename):
        df = pd.read_csv(filename, engine="python", error_bad_lines=False, encoding='GBK')
        return df

def load_neighbours(item2id):

    neighbours = {}

    # print(os.getcwd())
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "kg_4.csv"),
                     engine="python", error_bad_lines=False, encoding='utf8')
    for index, row in df.iterrows():
        h = item2id[row['h']]
        t = item2id[row['t']]
        if h not in neighbours.keys():
            neighbours[h] = [t]
        else:
            neighbours[h].append(t)
        if t not in neighbours.keys():
            neighbours[t] = [h]
        else:
            neighbours[t].append(h)
    # for k,v in neighbours.items():
    #     print(k,v)
    return neighbours

def xietongguolu():
# if __name__=="__main__":
    # rating_file = 'D:\\学习资料\\推荐系统\\ml-latest-small\\ratings.csv'

    rating_file_train = os.path.join(os.getcwd(), "data", "rating_train_static.csv")
    rating_file_test = os.path.join(os.getcwd(), "data", "rating_test.csv")
    # rating_file = os.path.join(os.getcwd(), "data", "ratings.dat")
    person2id_file = os.path.join(os.getcwd(), "data", "person2id.csv")
    # print(rating_file_train)
    userCF =  UserBasedCF()
    userCF.load_csv(rating_file_train)
    person2id_df = pd.read_csv(person2id_file, engine="python", error_bad_lines=False, encoding='utf-8', sep='\t')
    person2id = {}
    for index, row in person2id_df.iterrows():
        person2id[row['person']] = row['id']
    train_df = userCF.load_csv(rating_file_train)
    test_df = userCF.load_csv(rating_file_test)
    trainSet_len = 0
    testSet_len = 0
    num = 0
    idx_person = 0
    for index, row in train_df.iterrows():
        user = row['place']
        item = row['person']
        rating = row['rating']
        userCF.trainSet.setdefault(user, {})
        userCF.trainSet[user][person2id[item]] = rating
        trainSet_len += 1
    for index, row in test_df.iterrows():
        user = row['place']
        item = row['person']
        rating = row['rating']
        userCF.testSet.setdefault(user, {})
        userCF.testSet[user][person2id[item]] = rating
        testSet_len += 1
    print('Split trainingSet and testSet success!')
    print('TrainSet = %s' % trainSet_len)
    print('TestSet = %s' % testSet_len)
    # neighbours = load_neighbours(person2id)
    # userCF = UserBasedCF()
    # userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    hit10_info = userCF.evaluate()
    # hit10_info = userCF.calculate_score(place)
    # return hit10_info
if __name__=='__main__':
    xietongguolu()
