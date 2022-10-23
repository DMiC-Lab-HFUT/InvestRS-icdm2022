import os
import networkx as nx
from tqdm import tqdm
import numpy as np
from sympy import symbols,diff

from all_data.UserCF1 import UserCF_my1
from all_data.UserCF import UserCF_my2
from scipy import optimize
from networkx.algorithms.link_prediction import resource_allocation_index
from networkx.algorithms.link_prediction import jaccard_coefficient
from networkx.algorithms.link_prediction import adamic_adar_index
from networkx.algorithms.link_prediction import preferential_attachment
from networkx.algorithms.link_prediction import cn_soundarajan_hopcroft
from networkx.algorithms.link_prediction import ra_index_soundarajan_hopcroft
from networkx.algorithms.link_prediction import within_inter_cluster
from networkx.algorithms.link_prediction import common_neighbor_centrality
import Data
import pandas as pd
import score
import DynamicData
import logging
def get_entity2id(path,filename):
    entity2id = {}
    id2entity = {}
    with open(path+filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split('\t')
            entity2id[entity[0]] = int(entity[1])
            id2entity[int(entity[1])] = entity[0]
    return entity2id,id2entity

def get_test_info(path,filename):
    test_dict = {}
    place_set = set()
    test_triple = set()
    with open(path + filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split(',')
            place_set.add(entity[1])
            if entity[1] not in test_dict:
                test_dict[entity[1]] = []
                test_dict[entity[1]].append(entity[0])
            else:
                test_dict[entity[1]].append(entity[0])
            test_triple.add((entity[0],entity[1],"INVEST"))
    return test_dict,place_set,test_triple
def get_kg_info(path,filename):
    kg_train_triple = set()
    with open(path + filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split(',')
            kg_train_triple.add((entity[0],entity[1],entity[2]))
    return kg_train_triple
def initialize_file(path,filename):
    with open(path + filename,'w',encoding='utf-8') as file:
        file.write('')
def get_validate_test(validate_kg5):
    place_set = set()
    validate_test_dict = {}
    for triple in validate_kg5:
        if triple[1]=="":
            print(triple[0])
        place_set.add(triple[1])
        if triple[1] not in validate_test_dict:
            validate_test_dict[triple[1]] = []
            validate_test_dict[triple[1]].append(triple[0])
        else:
            validate_test_dict[triple[1]].append(triple[0])
    return validate_test_dict,place_set
def get_test_range(validate_test_dict,entity2id):
    test_range = []
    test_range_num = []
    for key,item in validate_test_dict.items():
        if len(item)>10:
            test_range.append(key)
            test_range_num.append(int(entity2id[key]))
    return test_range,test_range_num
def get_validate_test_range(validate_test_dict,entity2id,a_b_real_dict):
    test_range = []
    test_range_num = []
    for key,item in validate_test_dict.items():
        if len(item)>10 and key in a_b_real_dict.keys():
            test_range.append(key)
            test_range_num.append(int(entity2id[key]))
    return test_range,test_range_num

def get_final_test_range(validate_test_dict,entity2id,a_b_real_dict):
    test_range = []
    test_range_num = []
    for key,item in validate_test_dict.items():
        if key in a_b_real_dict.keys():
            if len(item)>10:
                test_range.append(key)
                test_range_num.append(int(entity2id[key]))
    return test_range,test_range_num
def define_test_range(d_test_dict, d_entity2id, score_dict,d_mugonsi_train,sc2ctrain):
    node_list = []
    r = []
    for node,item in d_test_dict.items():
        c = 0
        for i in item:
            if i in sc2ctrain:
                if set(sc2ctrain[i]) & set(d_mugonsi_train):
                    c+=1
        if c >=int(10):
            node_list.append(node)
    for n in node_list:
        if n in score_dict.keys():
            r.append(d_entity2id[n])

    return r
def process_kg2_triple(dkg2_train_triple):
    C2C_dTrain1 = set()
    C2C_dTest1 = set()
    C2C_dTrain_company_set = set()
    for triple in dkg2_train_triple:
        if triple[2] in ['BRANCH','OWN_C']:
            C2C_dTest1.add((triple[0],triple[1],triple[2]))
            C2C_dTrain_company_set.add(triple[0])

        if triple[2]=='INVEST_C':
            C2C_dTrain_company_set.add(triple[0])
            # C2C_dTrain_company_set.add(triple[1])
            C2C_dTrain1.add((triple[0],triple[1],triple[2]))
    return C2C_dTrain1,C2C_dTest1,C2C_dTrain_company_set


def get_c2c_from_kg2(ss1kg2_train_triple,ss1kg2_test_triple):
    branch_set = set()
    invest_set = set()
    train_company_set = set()
    test_company_set = set()

    for triple in ss1kg2_train_triple:
        if triple[2] in ['BRANCH','OWN_C']:
            branch_set.add((triple[0],triple[1],triple[2]))
        if triple[2]=='INVEST_C':
            train_company_set.add((triple[0],triple[1],triple[2]))
            invest_set.add((triple[0],triple[1],triple[2]))
    for triple in ss1kg2_test_triple:
        if triple[2] in ['BRANCH','OWN_C']:
            branch_set.add((triple[0],triple[1],triple[2]))
            test_company_set.add((triple[0],triple[1],triple[2]))
        if triple[2]=='INVEST_C':
            invest_set.add((triple[0],triple[1],triple[2]))
    return invest_set,branch_set,train_company_set,test_company_set
def get_2num(d_mugonsi_train,d_entity2id):
    d_mugonsi_train_num = []
    for company in d_mugonsi_train:
        d_mugonsi_train_num.append(d_entity2id[company])
    return d_mugonsi_train_num
def get_ebunch_from_train(C2C_dTrain1,d_entity2id):
    ebunch_num = []
    for triple in C2C_dTrain1:
        ebunch_num.append(d_entity2id[triple[0]])
    return ebunch_num
def get_c2s_from_c2c(c2c_train):
    Sc2c = {}
    C2Sc = {}
    mugongsi = set()
    for triple in c2c_train:
        mugongsi.add(triple[0])
        if triple[0] not in C2Sc.keys():  # mu->zi
            C2Sc[triple[0]] = []
            C2Sc[triple[0]].append(triple[1])
        else:
            C2Sc[triple[0]].append(triple[1])
        if triple[1] not in Sc2c.keys():#zi->mu
            Sc2c[triple[1]] = []
            Sc2c[triple[1]].append(triple[0])
        else:
            Sc2c[triple[1]].append(triple[0])
    return C2Sc,Sc2c,mugongsi
def get_train_Graph(ss1kg1_train_triple,ss1kg2_train_triple,ss1kg5_train_triple,Graph,ss3_entity2id):
    for triple in ss1kg1_train_triple:
        Graph.add_edge(int(ss3_entity2id[triple[0]]),int(ss3_entity2id[triple[1]]))
    for triple in ss1kg2_train_triple:
        if triple[2] =='INVEST_C':
            Graph.add_edge(int(ss3_entity2id[triple[0]]),int(ss3_entity2id[triple[1]]))
    for triple in ss1kg5_train_triple:
        Graph.add_edge(int(ss3_entity2id[triple[0]]),int(ss3_entity2id[triple[1]]))
    return Graph

def write_rating_train1(path,filename,train_triple,entity2id,c2Sc):
    str_rating = ''
    str_rating+='place'+','+'person'+','+'rating'+'\n'
    for triple in train_triple:
        if triple[0] in c2Sc:
            str_rating += str(entity2id[triple[1]]) + ',' + c2Sc[triple[0]][0] + ',' + str(1) + '\n'
        else:
            str_rating += str(entity2id[triple[1]])+','+triple[0]+','+str(1)+'\n'
    with open(path+filename,'w',encoding='utf-8') as file:
        file.write(str_rating)
def write_person2id(path,filename,company_set,entity2id):
    str_rating = ''
    str_rating+='person'+'\t'+'id'+'\n'
    for company in company_set:
        str_rating += company+'\t'+str(entity2id[company])+'\n'

    with open(path+filename,'w',encoding='utf-8') as file:
        file.write(str_rating)

def write_usercf_file(path,kg5_train_triple,kg5_test_triple,Sc2c_train,Sc2c_test):
    company_set = set()
    place_set = set()
    for triple in kg5_train_triple:
        company_set.add(triple[0])
        place_set.add(triple[1])
    for triple in kg5_test_triple:
        company_set.add(triple[0])
        place_set.add(triple[1])
    for key,item in Sc2c_train.items():
        company_set.add(key)
        for com in item:
            company_set.add(com)
    for key,item in Sc2c_test.items():
        company_set.add(key)
        for com in item:
            company_set.add(com)

    entity_list = []
    for node in place_set:
        entity_list.append(node)
    for node in company_set:
        entity_list.append(node)
    entity2id = {}
    id2entity = {}
    for index, n in enumerate(entity_list):
        entity2id[n] = int(index)
        id2entity[int(index)] = n
    write_rating_train1(path, '/rating_train_static.csv', kg5_train_triple, entity2id, Sc2c_train)
    # write_rating_train1(path, '/rating_test.csv', test_triple_name, entity2id, Sc2c)
    write_rating_train1(path, '/rating_test.csv', kg5_test_triple, entity2id, Sc2c_test)

    write_person2id(path, '/person2id.csv', company_set, entity2id)
    write_person2id(path, '/place2id.csv', place_set, entity2id)

def get_cf_place2id(path,filename):
    place2id = {}
    id2place = {}
    place_set = set()
    with open(path+filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split('\t')
            place2id[entity[0]] = int(entity[1])
            id2place[int(entity[1])] = entity[0]
            place_set.add(entity[0])
    return place2id,id2place,place_set
def get_em_entity2id(path,filename):
    entity2id = {}
    id2entity = {}
    # place_set = set()
    with open(path+filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split('\t')
            entity2id[entity[0]] = int(entity[1])
            id2entity[int(entity[1])] = entity[0]
            # place_set.add(entity[0])
    return entity2id,id2entity
def load_embedding_data(path):
    er_vec, er = {}, []
    with open(path) as fr:
        lines = fr.readlines()
        for line in lines:
            er_str, vec_str = line.strip().split('\t')
            er.append(int(er_str))
            er_vec[int(er_str)] = np.array([float(s) for s in vec_str[1:-1].split(',')])
    return er_vec, er
def load_embedding_data2(path,em_id2entity,ss3_entity2id):
    er_vec, er = {}, []
    with open(path) as fr:
        lines = fr.readlines()
        for line in lines:
            er_str, vec_str = line.strip().split('\t')
            entity_id = ss3_entity2id[em_id2entity[int(er_str)]]
            er.append(int(entity_id))
            er_vec[int(entity_id)] = np.array([float(s) for s in vec_str[1:-1].split(',')])
    return er_vec, er

def write_em_train_file(ss1kg1_train_triple,ss1kg2_train_triple,
                        ss1kg5_train_triple,ss3_entity2id,ss3_relation2id,path):
    # print(ss1kg5_train_triple)
    train_triple = set()
    entity_set = set()
    for triple in ss1kg1_train_triple:
        entity_set.add(triple[0])
        entity_set.add(triple[1])
        train_triple.add((int(ss3_entity2id[triple[0]]),int(ss3_entity2id[triple[1]]),int(ss3_relation2id[triple[2]])))
    for triple in ss1kg2_train_triple:
        if triple[2]=='INVEST_C':
            entity_set.add(triple[0])
            entity_set.add(triple[1])
            train_triple.add((int(ss3_entity2id[triple[0]]),int(ss3_entity2id[triple[1]]),int(ss3_relation2id[triple[2]])))
    for triple in ss1kg5_train_triple:
        entity_set.add(triple[0])
        entity_set.add(triple[1])
        train_triple.add(
            (int(ss3_entity2id[triple[0]]), int(ss3_entity2id[triple[1]]), int(ss3_relation2id[triple[2]])))
    with open(path+'train2id.txt','w',encoding='utf-8') as file:
        file.write(str(len(train_triple))+'\n')
        str0=''
        for triple in train_triple:
            str0 += str(triple[0]) + '\t' + str(triple[1])+'\t'+str(triple[2]) + '\n'
            # file.write(str(triple[0])+'\t'+str(triple[1])+'\t'+str(triple[2])+'\n')
        file.write(str0)
    with open(path+'entity2id.txt','w',encoding='utf-8') as file2:
        file2.write(str(len(entity_set)) + '\n')
        str1 = ''
        for entity in entity_set:
            str1+= entity + '\t' + str(ss3_entity2id[entity]) + '\n'
        file2.write(str1)
            # file2.write(entity+'\t'+str(ss3_entity2id[entity])+'\n')
    with open(path+'relation2id.txt','w',encoding='utf-8') as file3:
        file3.write(str(len(ss3_relation2id)) + '\n')
        for relation,item in ss3_relation2id.items():
            file3.write(relation+'\t'+str(item)+'\n')
def write_em_train_file2(ss1kg1_train_triple,ss1kg2_train_triple,
                        ss1kg5_train_triple,ss3_entity2id,ss3_relation2id,path):
    entity2id = {}

    train_triple = set()
    entity_set = set()

    for triple in ss1kg1_train_triple:
        entity_set.add(triple[0])
        entity_set.add(triple[1])
        train_triple.add((triple[0],triple[1],triple[2]))
    for triple in ss1kg2_train_triple:
        if triple[2]=='INVEST_C':
            entity_set.add(triple[0])
            entity_set.add(triple[1])
            train_triple.add((triple[0], triple[1], triple[2]))

    for triple in ss1kg5_train_triple:
        entity_set.add(triple[0])
        entity_set.add(triple[1])
        train_triple.add((triple[0],triple[1],triple[2]))


    with open(path+'entity2id.txt','w',encoding='utf-8') as file2:
        file2.write(str(len(entity_set)) + '\n')
        str1 = ''
        for index,entity in enumerate(entity_set):
            entity2id[entity] = int(index)
            file2.write(entity + '\t'+str(index) + '\n')

    with open(path+'train2id.txt','w',encoding='utf-8') as file:
        file.write(str(len(train_triple))+'\n')
        str0=''
        for triple in train_triple:
            str0 += str(entity2id[triple[0]]) + '\t' + str(entity2id[triple[1]])+'\t'+str(ss3_relation2id[triple[2]]) + '\n'
            # file.write(str(triple[0])+'\t'+str(triple[1])+'\t'+str(triple[2])+'\n')
        file.write(str0)

            # file2.write(entity+'\t'+str(ss3_entity2id[entity])+'\n')
    with open(path+'relation2id.txt','w',encoding='utf-8') as file3:
        file3.write(str(len(ss3_relation2id)) + '\n')
        for relation,item in ss3_relation2id.items():
            file3.write(relation+'\t'+str(item)+'\n')
def get_ebunch(place,train_person_set):
    ebunch = []
    for node in train_person_set:
        ebunch.append((node,place))
    return ebunch
def calculate_similarity(ebunch, mode,graphUN):
    if mode == "ra":
        preds = resource_allocation_index(graphUN, ebunch)
    elif mode == "jc":
        preds = jaccard_coefficient(graphUN, ebunch)
    elif mode == "aa":
        preds = adamic_adar_index(graphUN, ebunch)
    elif mode == "pa":
        preds = preferential_attachment(graphUN, ebunch)
    elif mode == "cnsh":
        preds = cn_soundarajan_hopcroft(graphUN, ebunch)
    elif mode == "rash":
        preds = ra_index_soundarajan_hopcroft(graphUN, ebunch)
    elif mode == "ic":
        preds = within_inter_cluster(graphUN, ebunch)
    else:
        preds = common_neighbor_centrality(graphUN, ebunch)

    score = {}
    for u, v, p in preds:
        # print(111111)
        score[u]=p
    return score
def resort(mode,ebunch,graphUN,constrain):

    score = calculate_similarity(ebunch, mode,graphUN)
    newTopK = dict(sorted(score.items(), key=lambda x: x[-1], reverse=True)[:constrain])
    ture_top = {}
    for key,value in newTopK.items():
        if value ==0:
            break
        ture_top[key] = value

    return ture_top
def combine_all_new(top_em,topN_CF,topN2):
    # em_socre = 0
    # cf_score = 0
    # jc_score = 0
    k = 0
    top_em1 = {}
    for node,item in top_em.items():
        top_em1[node] = item
        k += 1
        if k >= 40:
            break
    top_em_resort = {}
    for node,item in top_em1.items():
        top_em_resort[node] = 2-item
    # max_num = max(top_em_resort.values())
    # min_num = min(top_em_resort.values())
    # top_em_resort_1 = {}
    # for node,item in top_em_resort.items():
    #     top_em_resort_1[node] = (item-min_num)/(max_num-min_num)
        # em_socre+=(item-min_num)/(max_num-min_num)
    top_em_resort = dict(sorted(top_em_resort.items(),key = lambda x:x[-1],reverse=False))
    em_score = max(top_em_resort.values())

    k = 0
    top_cf_resort = {}
    for node, item in topN_CF.items():
        top_cf_resort[node] = item
        # cf_score += item
        k += 1
        if k >= 40:
            break
    try:
        cf_score = max(top_cf_resort.values())
    except:
        cf_score=0

    k = 0
    top_jc_resort = {}
    for node, item in topN2.items():
        top_jc_resort[node] = item

        k += 1
        if k >= 40:
            break
    try:
        jc_score = max(top_jc_resort.values())
    except:
        jc_score=0
    return em_score,cf_score,jc_score,top_em_resort,top_cf_resort,top_jc_resort

def get_mugongsi_num(mugonsi_train,ss3_entity2id):
    mugongsi_num = []
    for com in mugonsi_train:
        mugongsi_num.append(int(ss3_entity2id[com]))
    return mugongsi_num
def get_embedding_topK(train_person_set,en_vec,re_vec,target,constrain,N,train_relation2id):
    topN = {}
    # top_degree ={}
    for node in train_person_set:
        # for node1 in train_person_set:
        #     if node!=node1:
        en_head = en_vec[int(node)]
        en_tail = en_vec[int(target)]
        relation = re_vec[int(train_relation2id['INVEST'])]
        score = np.sum(np.square((en_head +relation- en_tail)))
        topN[node] = score

    topN_j = dict(sorted(topN.items(), key=lambda x: x[-1], reverse=False)[:constrain])


    return topN_j

def combine_all(a,b,top_em,topN_CF,topN2):
    commen = []
    topN = {}#0.05,0.3-->9
    em_N = a
    cf_N = b
    jc_N = 1-em_N-cf_N

    # top_em_resort = {}
    # for node,item in top_em.items():
    #     top_em_resort[node] = 2-item
    #
    # max_num = max(top_em_resort.values())
    # min_num = min(top_em_resort.values())
    top_resort = {}
    #
    for node,item in top_em.items():
        top_resort[node] = em_N * item

    for node,item in topN_CF.items():
        if node in top_resort:
            score_new = cf_N * item
            if score_new > top_resort[node]:
                top_resort[node] = score_new
        else:
            top_resort[node] = cf_N * item
    for node,item in topN2.items():
        if node in top_resort:
            score_new = jc_N * item
            if score_new > top_resort[node]:
                top_resort[node] = score_new
        else:
            top_resort[node] = jc_N * item

    topN_recommend = dict(sorted(top_resort.items(),key=lambda x:x[-1],reverse=True))

    return topN_recommend

def Dynamic_combine_all(a,b,topN_CF,topN_JC):
    topN = {}#0.05,0.3-->9
    em_N = a
    cf_N = b

    jc_N = 1-em_N-cf_N

    # top_em_resort = {}
    # for node,item in top_em.items():
    #     top_em_resort[node] = 2-item
    #
    # max_num = max(top_em_resort.values())
    # min_num = min(top_em_resort.values())
    top_resort = {}
    #
    for node,item in topN_CF.items():
        top_resort[node] = cf_N * item

    for node,item in topN_JC.items():
        if node in top_resort:
            score_new = jc_N * item
            if score_new > top_resort[node]:
                top_resort[node] = score_new
        else:
            top_resort[node] = jc_N * item

    topN_recommend = dict(sorted(top_resort.items(),key=lambda x:x[-1],reverse=True))

    return topN_recommend
def get_kg5_place(triples):
    place_set = set()
    for triple in triples:
        place_set.add(triple[1])
    return place_set

def get_validate_company(company,sc2c,ss3_entity2id):
    validate_mugongsi = set()
    for com in company:
        if com in sc2c:
            for c in sc2c[com]:
                if c in ss3_entity2id:
                    validate_mugongsi.add(ss3_entity2id[c])
        else:
            if com in ss3_entity2id:
                validate_mugongsi.add(ss3_entity2id[com])

    return validate_mugongsi

# def get_company2place_triple(ss1kg5_train_triple,ss3_id2entity):
#     for triple in s
def combine_all_toList(top_em,topN_CF,topN_JC,company_num):
    all_tolist = {}
    score_mode = {}
    kkk = 0
    # top_em_resort = {}
    # for node, item in top_em.items():
    #     top_em_resort[node] = 2 - item

    for key,item in top_em.items():
        if key in company_num:
            kkk+=1
            all_tolist[key] = item
            score_mode[key] = 'w1'
        else:
            score_mode[key] = 'w1n'
            all_tolist[key] = item

    for key,item in topN_CF.items():
        if key in company_num:
            score = item
            if key in all_tolist:
                if score>all_tolist[key]:
                    all_tolist[key] = score
                    score_mode[key] = 'w2'
            else:
                kkk += 1
                all_tolist[key] = score
                score_mode[key] = 'w2'
        else:
            score_mode[key] = 'w2n'
            all_tolist[key] = item
    for key,item in topN_JC.items():
        if key in company_num:
            score_jc = item
            if key in all_tolist:
                if score_jc>all_tolist[key]:
                    all_tolist[key] = score_jc
                    score_mode[key] = 'w3'
            else:
                kkk += 1
                all_tolist[key] = score_jc
                score_mode[key] = 'w3'
        else:
            score_mode[key] = 'w3n'
            all_tolist[key] = item
    # for node,soc in all_tolist.items():
    #     all_socre+=soc
    count_1=0
    count_2=0
    count_3=0
    min_mode=''

    for node,mode in score_mode.items():
        if mode=='w1':
            count_1+=1
        if mode=='w2':
            count_2+=1
        if mode=='w3':
            count_3 += 1
    if count_1<=count_2 and count_1<=count_3:
        min_mode='w1n'
    elif count_2<=count_1 and count_2<=count_3:
        min_mode='w2n'
    else:
        min_mode='w3n'

    all_tolist = dict(sorted(all_tolist.items(),key=lambda x:x[-1],reverse=True))
    print('====kkk====',kkk)

    return all_tolist,score_mode,min_mode

def combine_CfJc_toList(topN_CF,topN_JC,company_num):
    all_tolist = {}
    score_mode = {}
    kkk = 0

    for key,item in topN_CF.items():
        if key in company_num:
            kkk+=1
            all_tolist[key] = item
            score_mode[key] = 'w2'
        else:
            score_mode[key] = 'w2n'
            all_tolist[key] = item

    for key,item in topN_JC.items():
        if key in company_num:
            score = item
            if key in all_tolist:
                if score>all_tolist[key]:
                    all_tolist[key] = score
                    score_mode[key] = 'w3'
            else:
                kkk += 1
                all_tolist[key] = score
                score_mode[key] = 'w3'
        else:
            score_mode[key] = 'w3n'
            all_tolist[key] = item

    count_cf=0
    count_jc=0

    min_mode=''

    for node,mode in score_mode.items():
        if mode=='w2':
            count_cf += 1
        if mode=='w3':
            count_jc += 1
    if count_cf<=count_jc:
        min_mode='w2n'
    else:
        min_mode='w3n'

    all_tolist = dict(sorted(all_tolist.items(),key=lambda x:x[-1],reverse=True))
    print('====kkk====',kkk)

    return all_tolist,score_mode,min_mode
def get_a_b(path,filename):
    a_b_dict = {}
    score_dict = {}
    with open(path+filename,encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split('\t')
            a_b_dict[entity[0]] = [float(entity[1]),float(entity[2])]

            score_dict[entity[0]] = -float(entity[3])

    return a_b_dict,score_dict

def write_testKG(d_test_range_num,d_id2entity,d_test_dict,path):
    triples = set()
    for num in d_test_range_num:
        coms = d_test_dict[d_id2entity[int(num)]]
        for com in coms:
            triples.add((com,d_id2entity[int(num)],'INVEST'))
    with open(path+'KG_test.txt','w',encoding='utf-8') as file:
        file.write(str(len(triples))+'\n')
        for triple in triples:
            file.write(str(triple[0])+','+str(triple[1])+','+str(triple[2])+'\n')





if __name__=='__main__':
    N=20
    mode = 'ra'
    constrain = 2*N
    hit = 0
    count=0
    Graph = nx.Graph()
    path = os.getcwd()+'\\'
    # print(path)
    usercf_path = os.getcwd()+'/UserCF/'

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logging.info("start loading 49-12 data...")

######################加载1949-2012数据集，相当于去除验证集之后的训练集，

    # ss1_entity2id,ss1_id2entity = get_entity2id(path+'snapshot1/','ss1_entity2id.txt')
    # ss1_test_dict,ss1_test_place,test_triple = get_test_info(path+'snapshot1/','kg5_test.txt')
    ss1kg1_train_triple = get_kg_info(path +'snapshot1/', 'ss1_kg1_train_real.txt')
    ss1kg2_train_triple = get_kg_info(path + 'snapshot1/', 'ss1_kg2_train_real.txt')

    # ss1kg2_test_triple = get_kg_info(path + 'snapshot1/', 'kg2_test.txt')
    ss1kg5_train_triple = get_kg_info(path +'snapshot1/', 'ss1_kg5_train_real.txt')
    train_relation2id,_ = get_entity2id(path,'relation2id.txt')
    # train_place = get_kg5_place(ss1kg5_train_triple)
    logging.info("loading 49-12 data success...")

#######################加载1949-2015数据集
    dynamic_year = 3
    logging.info('start loading 49-1'+str(dynamic_year+1)+' data...')
    d_entity2id, d_id2entity = get_entity2id(path + 'snapshot' + str(dynamic_year) + '/',
                                             'ss' + str(dynamic_year) + '_entity2id_real.txt')
    d_relation2id, d_id2relation = get_entity2id(path + 'snapshot' + str(dynamic_year) + '/',
                                                 'ss' + str(dynamic_year) + '_relation2id.txt')
    d_test_dict, d_test_place, d_test_triple = get_test_info(path + 'snapshot' + str(dynamic_year) + '/',
                                                             'kg5_test_real.txt')
    d_kg2_test_1 = get_kg_info(path + 'snapshot' + str(dynamic_year) + '/', 'kg2_test_real.txt')
    # d_kg2_test_2 = get_kg_info(path + 'dynamic3/', 'kg2_test.txt')#2.12有bug的地方

    # print(len(d_kg2_test_2&d_kg2_test_1))

    # d_kg2_test = d_kg2_test_1|d_kg2_test_2
    d_kg2_test = d_kg2_test_1

    d_kg1_train_triple = get_kg_info(path + 'snapshot' + str(dynamic_year) + '/',
                                     'ss' + str(dynamic_year) + '_kg1_train_real.txt')

    d_kg2_train_triple = get_kg_info(path + 'snapshot' + str(dynamic_year) + '/',
                                     'ss' + str(dynamic_year) + '_kg2_train_real.txt')
    d_kg5_train_triple = get_kg_info(path + 'snapshot' + str(dynamic_year) + '/',
                                     'ss' + str(dynamic_year) + '_kg5_train_real.txt')
    logging.info("loading 49-1"+str(dynamic_year+1)+" data success...")
    # _, score_dict = get_a_b(path, 'hit_info_score_4.txt')#前一个时间步的数据,a,b,以及得分。
#######################生成验证集
    logging.info("process validate Dataset...")
    validate_kg5 = d_kg5_train_triple - ss1kg5_train_triple#验证集的company2place

    validate_test_dict,validate_place = get_validate_test(validate_kg5)

    validate_kg2 = d_kg2_train_triple-ss1kg2_train_triple
    validate_kg1 = d_kg1_train_triple-ss1kg1_train_triple

    # validate_range,validate_range_num = get_validate_test_range(validate_test_dict,d_entity2id,score_dict)##计算出分数之后
    validate_range,validate_range_num = get_test_range(validate_test_dict,d_entity2id)



    v_C2C_Train1, v_C2C_Test1, v_C2C_Train_company_set = process_kg2_triple(ss1kg2_train_triple)
    v_C2C_Train2, v_C2C_Test2, v_C2C_Test_company_set = process_kg2_triple(validate_kg2)

    v_C2C_Test = v_C2C_Test1 | v_C2C_Test2
    v_c2sc_test, v_sc2c_test, v_mugonsi_test = get_c2s_from_c2c(v_C2C_Test)
    v_c2sc_train, v_sc2c_train, v_mugonsi_train = get_c2s_from_c2c(v_C2C_Train1)  # 训练集中的母-》子，子-》母，训练集中的母公司名称集合
    v_mugonsi_train_num = get_2num(v_mugonsi_train, d_entity2id)
    v_Graph = nx.Graph()
    v_train_Graph = get_train_Graph(ss1kg1_train_triple, ss1kg2_train_triple,

                                    ss1kg5_train_triple, v_Graph, d_entity2id)
    logging.info('validate train Graph complete')
    #######################################################################################################################

#     write_usercf_file(usercf_path + 'data/', ss1kg5_train_triple, validate_kg5, v_sc2c_train, v_sc2c_test)  # 重写协同过滤文件
#
#     v_usercf_place2id, v_usercf_id2place, d_cf_place_set = get_cf_place2id(usercf_path + 'data/', 'place2id.csv')
#     v_usercf_company2id, v_usercf_id2company, _ = get_cf_place2id(usercf_path + 'data/', 'person2id.csv')
#     UserCF = UserCF_my2.xietongguolu()
# # #2.12修改
# #
#     en_vec, en = load_embedding_data(path + 'snapshot3/new_ss3_e.transe')
#     re_vec, re = load_embedding_data(path + 'snapshot3/new_ss3_r.transe')
#
#     v_test_range = tqdm(validate_range_num)
#     # a_b_dict = {}
#     for index, place in enumerate(v_test_range):
#         v_test_range.set_description("train processing %s" % index)
#
#         company = validate_test_dict[d_id2entity[place]]
#
#         validate_company = get_validate_company(company, v_sc2c_test,d_entity2id)
#         ###embedding
#         top_em = get_embedding_topK(v_mugonsi_train_num, en_vec, re_vec, place, constrain, N * 4, train_relation2id)
#         # print(top_em)
#         top_em_resort = {}
#         for node, item in top_em.items():
#             top_em_resort[node] = 2 - item
#         top_em_resort = dict(sorted(top_em_resort.items(), key=lambda x: x[-1], reverse=True))
#
#         ###
#
#         info = []
#         topN_CF = {}
#         # topN_CF_score = {}
#         if d_id2entity[place] in v_usercf_place2id:
#             info = UserCF.calculate_score(v_usercf_place2id[d_id2entity[place]])
#         else:
#             continue
#         index=0
#         for index,two in enumerate(info):
#             topN_CF[d_entity2id[v_usercf_id2company[two[0]]]] = two[1]
#             if index >=constrain:
#                 break
#         print(d_id2entity[place])
#         v_ebunch = get_ebunch(place, v_mugonsi_train_num)
#         topN_JC = resort(mode, v_ebunch, v_train_Graph, N*4)
#
# #2.11修改
#         # all_tolist,score_mode,min_mode =combine_CfJc_toList(topN_CF, topN_JC, validate_company)
#         all_tolist, score_mode, min_mode = combine_all_toList(top_em_resort, topN_CF, topN_JC, validate_company)
#         v_data = DynamicData.DDate()
#         v_data.get_score(score_mode,topN_CF,topN_JC)
#         v=0
#
#         for node,item in all_tolist.items():
#             v+=1
#             if v>=10:
#                 if score_mode[node] == min_mode:
#                     v_data.mode = score_mode[node]
#                     v_data.target_score = item
#                     break
#         constraint_ueq = []#
#         ###
#         # if v_data.mode=='w2n':
#         #     constraint_ueq = [
#         #         lambda w: w[0] + w[1] - 1,
#         #         lambda w: w[1] * v_data.target_score - (1 - w[0] - w[1]) * v_data.jc_min,
#         #     ]
#         # else:
#         #     constraint_ueq = [
#         #         lambda w: w[0] + w[1] - 1,
#         #         lambda w: (1 - w[0] - w[1]) * v_data.target_score - w[1] * v_data.cf_min,
#         #     ]
#         if v_data.mode=='w1n':
#             constraint_ueq = [
#                 lambda w: w[0] + w[1] - 1,
#                 lambda w: w[0] * v_data.target_score - w[1] * v_data.cf_min,
#                 lambda w: w[0] * v_data.target_score - (1 - w[0] - w[1]) * v_data.jc_min,
#             ]
#         elif v_data.mode=='w2n':
#             constraint_ueq = [
#                 lambda w: w[0] + w[1] - 1,
#                 lambda w: w[1] * v_data.target_score - w[0] * v_data.em_min,
#                 lambda w: w[1] * v_data.target_score - (1 - w[0] - w[1]) * v_data.jc_min,
#             ]
#         else:
#             constraint_ueq = [
#                 lambda w: w[0] + w[1] - 1,
#                 lambda w: (1 - w[0] - w[1]) * v_data.target_score - w[0] * v_data.em_min,
#                 lambda w: (1 - w[0] - w[1]) * v_data.target_score - w[1] * v_data.cf_min,
#             ]
#         ###
#         # score = score_dict[d_id2entity[place]]
#         # v_data.score = score
#         x, y = v_data.calculate(constraint_ueq)
#         a_f,b_f = x.tolist()[0],x.tolist()[1]
#         score = y.tolist()[0]
#
#         with open(path+'a_b_index.txt','a+',encoding='utf-8') as f:
#             # f.write(str(place)+'\t'+'a:'+str(a_f)+'\t'+'b:'+str(b_f)+'\t'+str(score)+'\t'+str(ht)+'\n')
#             f.write(str(d_id2entity[place])+'\t')
#             f.write(str(a_f)+'\t')
#             f.write(str(b_f)+'\t')
#             # f.write('score:' + str(score)+'\t')
#             f.write(str(score))
#             f.write('\n')
    ####################################################################################################################
    logging.info('validate dataset process complete')
    C2C_dTrain1, C2C_dTest1, C2C_dTrain_company_set = process_kg2_triple(d_kg2_train_triple)#invest，branch
    C2C_dTrain2, C2C_dTest2, C2C_dTest_company_set = process_kg2_triple(d_kg2_test)#invest，branch
    # print(len(C2C_dTest_company_set&C2C_dTrain_company_set))
    ###
    C2C_dTest = C2C_dTest1 | C2C_dTest2
    d_c2sc_test, d_sc2c_test, d_mugonsi_test = get_c2s_from_c2c(C2C_dTest)


    d_c2sc_train, d_sc2c_train, d_mugonsi_train = get_c2s_from_c2c(C2C_dTrain1)  # 训练集中的母-》子，子-》母，训练集中的母公司名称集合
    ###
    d_ebunch_num = get_ebunch_from_train(C2C_dTrain1,d_entity2id)
    d_test_mugongsi_num = get_ebunch_from_train(C2C_dTest1,d_entity2id)
    # print(len(C2C_dTrain_company_set))
    # print(len(C2C_dTest_company_set))
    C2C_d_kg2 = C2C_dTrain1 | C2C_dTest1
    test_c2sc_train, test_sc2c_train, test_mugonsi_train = get_c2s_from_c2c(C2C_dTest)
    #有问题在这里
    c2ctrain = C2C_dTrain1 | C2C_dTrain2
    _, sc2ctrain, _ = get_c2s_from_c2c(c2ctrain)



    d_Graph = nx.Graph()
    d_train_Graph = get_train_Graph(d_kg1_train_triple, d_kg2_train_triple,
                                    d_kg5_train_triple, d_Graph, d_entity2id)
    # print(len(d_train_Graph.nodes()))
    logging.info('训练图构建完成...')

    usercf_path2 = os.getcwd() + '/UserCF1/'
    write_usercf_file(usercf_path2 + 'data/', d_kg5_train_triple, d_test_triple, d_sc2c_train, d_sc2c_test)  # 重写协同过滤文件

    d_usercf_place2id, d_usercf_id2place, d_cf_place_set = get_cf_place2id(usercf_path2 + 'data/', 'place2id.csv')
    d_usercf_company2id, d_usercf_id2company, _ = get_cf_place2id(usercf_path2 + 'data/', 'person2id.csv')
    UserCF1 = UserCF_my1.xietongguolu()
    # a_b_real_dict, _ = get_a_b(path, 'a_b_score' + str(dynamic_year) + '.txt')

    a_b_real_dict, _ = get_a_b(path, 'a_b_real.txt')

    d_test_range_num = define_test_range(d_test_dict, d_entity2id, a_b_real_dict,d_mugonsi_train,sc2ctrain)  # 测试集的地点名称和测试集地点num
    # d_test_range_num = [d_entity2id['潜江市'],d_entity2id['荣成市'],d_entity2id['义乌市'],d_entity2id['盘州市'],
    #                 d_entity2id['东阳市'],d_entity2id['诸城市'],d_entity2id['海门市'],d_entity2id['乐清市']
    #                 ,d_entity2id['诸暨市'],d_entity2id['宜兴市']]
    # print(d_test_range_num)
    # write_testKG(d_test_range_num,d_id2entity,d_test_dict,path)
    # print('write!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    d_test_range = tqdm(d_test_range_num)
    d_hit = 0
    d_count = 0
    d_hit5 = 0
    d_hit10 = 0
    d_hit20 = 0
    ndcg_score = 0
    f1_score = 0
    auc_score = 0

    # a_b_real_dict,_ = get_a_b(path, 'a_b_score'+str(dynamic_year)+'.txt')

    for index, dplace in enumerate(d_test_range):
        d_test_range.set_description("train processing %s" % index)

        d_company = d_test_dict[d_id2entity[dplace]]
        logging.info(d_id2entity[dplace]+'测试集中的命中公司长度'+str(len(d_company)))
        info = []
        d_topN_CF = {}
        # topN_CF_score = {}
        if d_id2entity[dplace] in d_usercf_place2id:
            info = UserCF1.calculate_score(d_usercf_place2id[d_id2entity[dplace]])
        else:
            continue
        for ix,two in enumerate(info):
            d_topN_CF[d_entity2id[d_usercf_id2company[two[0]]]] = two[1]
            if ix>=constrain:
                break

        d_ebunch = get_ebunch(dplace, d_ebunch_num)
        d_topN_JC = resort(mode, d_ebunch, d_train_Graph, N*2)


        # _, _, _,f_top_em_resort,f_top_cf_resort,f_top_jc_resort = combine_all_new(f_top_em, f_topN_CF, f_topN2)
        real_a, real_b = a_b_real_dict[d_id2entity[dplace]][0],a_b_real_dict[d_id2entity[dplace]][1]
        # real_a = 0.03
        # real_b = 0.3

        d_recommend_topN=Dynamic_combine_all(real_a,real_b,d_topN_CF,d_topN_JC)
        # print(d_recommend_topN)
        d_count_N = 0
        ndcg_hit = []
        # hit_score = []
        for top_node in d_recommend_topN.keys():
            if top_node not in d_test_mugongsi_num:  # 判断是不是母公司的企业
                continue
            if (d_id2entity[top_node], d_id2entity[dplace], 'INVEST') in d_kg5_train_triple:
                # print(1111)
                continue

            d_count_N += 1
            c = d_id2entity[top_node]
            # print(c)
            Scompany = test_c2sc_train[d_id2entity[top_node]]  # zi gong si

            if len(set(d_company) & set(Scompany)) > 0:
                ndcg_hit.append(1)
                if d_count_N<=5:
                    d_hit5 += 1
                if d_count_N<=10:
                    d_hit10 += 1
                if d_count_N<=20:
                    d_hit20 += 1
                d_hit += 1
                # ht += 1
                # print(id2entity[top_node]+'->'+id2entity[place])
                paths = nx.shortest_path(d_train_Graph, top_node, dplace)
                for path1 in paths:
                    print(d_id2entity[path1] + '->')
                print('|')
            else:
                ndcg_hit.append(0)
            # hit_score.append(d_recommend_topN[top_node])
        ndcg_hit_k = np.array(ndcg_hit[:20])
        dcg = np.sum((2 ** ndcg_hit_k - 1) / np.log2(np.arange(2, len(ndcg_hit_k) + 2)))
        sorted_hits_k = np.flip(np.sort(np.array(ndcg_hit)))[:20]
        idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, len(sorted_hits_k) + 2)))
        if idcg == 0:
            idcg = np.inf
        ndcg = dcg / idcg
        ndcg_score += ndcg

        # learn_data = pd.DataFrame({"Learn": hit_score[:20], "Real": ndcg_hit_k.tolist()})
        # scoreC = score.Score(learn_data["Learn"], learn_data["Real"], 0.3, 1)
        # f1_score += scoreC.get_f1()
        # auc_score += scoreC.get_auc_by_rank()
            # if d_count_N >= N:
            #     break

        d_count += 1

    precision5 = d_hit5 / (1.0 * d_count * 5)
    precision10 = d_hit10 / (1.0 * d_count * 10)
    precision20 = d_hit20 / (1.0 * d_count * 20)
    ndcg20 = ndcg_score / (1.0 * d_count)
    print('===hit5===')
    print('======topK=====' + mode + '======', precision5)
    print('===hit10===')
    print('======topK=====' + mode + '======', precision10)
    print('===hit20===')
    print('======topK=====' + mode + '======', precision20)

    print('===ndcg20===',ndcg20)
    # print("auc 20:",auc_score/d_count)
    # print("f1 20:",f1_score/d_count)