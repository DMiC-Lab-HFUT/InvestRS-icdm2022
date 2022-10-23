import os


def duplicate_remove3(path,filename,entity2id,relation2id,place_name):
    company2P_name = set()
    company2P = set()
    with open(path + filename, encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split(',')
            if entity[1] in place_name:
                company2P_name.add((entity[0],entity[1],'INVEST'))
                company2P.add((entity2id[entity[0]],entity2id[entity[1]],relation2id['INVEST']))
    return company2P_name,company2P

def duplicate_remove4(path,filename,entity2id,relation2id,place_name):
    company2P_name = set()
    company2P = set()
    with open(path + filename, encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split(',')
            if entity[1] in place_name:
                company2P_name.add((entity[0],entity[1],'INVEST'))
                # company2P.add((entity2id[entity[0]],entity2id[entity[1]],relation2id['INVEST']))
    return company2P_name

def get_entity2id(path,filename):
    id2entity = {}
    with open(path+filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split('\t')
            id2entity[entity[0]] = int(entity[1])
    return id2entity

def get_place(path,filename,entity2id):
    place_list = []
    place_name = []
    with open(path+filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            place = line.strip()
            place_list.append(entity2id[place])
            place_name.append(place)
    return place_list,place_name
def get_id2entity(path,filename):
    id2entity = {}
    with open(path+filename,encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split('\t')
            id2entity[int(entity[1])] = entity[0]
    return id2entity
def write_rating_train(path,filename,train_triple,id2entity,c2Sc):
    str_rating = ''
    str_rating+='place'+','+'person'+','+'rating'+'\n'
    for triple in train_triple:
        if id2entity[triple[0]] in c2Sc:

            str_rating += str(triple[1]) + ',' + c2Sc[id2entity[triple[0]]][0] + ',' + str(1) + '\n'
        else:
            str_rating += str(triple[1])+','+id2entity[triple[0]]+','+str(1)+'\n'
    with open(path+filename,'w',encoding='utf-8') as file:
        file.write(str_rating)

def write_rating_train1(path,filename,train_triple,id2entity,c2Sc):
    str_rating = ''
    str_rating+='place'+','+'person'+','+'rating'+'\n'
    for triple in train_triple:
        if triple[0] in c2Sc:

            str_rating += str(entity2id[triple[1]]) + ',' + c2Sc[triple[0]][0] + ',' + str(1) + '\n'
        else:
            str_rating += str(entity2id[triple[1]])+','+triple[0]+','+str(1)+'\n'
    with open(path+filename,'w',encoding='utf-8') as file:
        file.write(str_rating)
def write_train_graph(path,filename,train_triple,id2entity,c2Sc,entity2id):
    str_rating = ''
    str_rating+='h,r,t'+'\n'
    for triple in train_triple:
        if id2entity[triple[0]] in c2Sc:

            str_rating += str(triple[1]) + ',' + str(entity2id[c2Sc[id2entity[triple[0]]][0]]) + ',' + str(1) + '\n'
        else:
            str_rating += str(triple[1])+','+str(entity2id[id2entity[triple[0]]])+','+str(1)+'\n'
    with open(path+filename,'w',encoding='utf-8') as file:
        file.write(str_rating)
def duplicate_remove2(path,filename,entity2id,relation2id):
    company2P = set()
    with open(path + filename, encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split(',')
            # if entity[2] == 'INVEST_C':
            company2P.add((entity[0], entity[1], entity[2]))
    return company2P
def write_person2id(path,filename,train_triple,entity2id,company2C_train,id2entity):
    str_rating = ''
    str_rating+='person'+'\t'+'id'+'\n'
    company_set = set()
    for triple in train_triple:
        company_set.add(triple[0])
    for triple in company2C_train:
        company_set.add(entity2id[triple[0]])
    for company in company_set:
        str_rating += id2entity[company]+'\t'+str(company)+'\n'

    with open(path+filename,'w',encoding='utf-8') as file:
        file.write(str_rating)

def get_C2C_info_train(company2C_train):

    Sc2c = {}

    C2SC = {}
    for triple in company2C_train:#mu->zi
        if triple[1] not in Sc2c.keys():
            Sc2c[triple[1]] = []
            Sc2c[triple[1]].append(triple[0])
        else:
            Sc2c[triple[1]].append(triple[0])

        if triple[0] not in C2SC.keys():#zi->mu
            C2SC[triple[0]] = []
            C2SC[triple[0]].append(triple[1])
        else:
            C2SC[triple[0]].append(triple[1])

    return C2SC,Sc2c
def process_kg1(path,filename):
    company2C = set()
    with open(path + filename, encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            entity = line.strip().split(',')
            company2C.add((entity[0], entity[2], entity[1]))
    person2company_dict = {}
    train_triple = []
    for entity in company2C:
        if entity[2] not in ['OWN', 'INVEST_H', 'SERVE']:
            continue
        if (entity[0], entity[1]) not in person2company_dict:
            person2company_dict[(entity[0], entity[1])] = []
            person2company_dict[(entity[0], entity[1])].append(entity[2])
        else:
            person2company_dict[(entity[0], entity[1])].append(entity[2])

    for key, item in person2company_dict.items():
        if len(item) > 1:
            if 'OWN' in item:
                train_triple.append((key[0], key[1], 'OWN'))

            elif 'INVEST_H' in item:
                train_triple.append((key[0], key[1], 'INVEST_H'))

            else:
                train_triple.append((key[0], key[1], 'SERVE'))

        elif len(item) == 1:
            train_triple.append((key[0], key[1], item[0]))

        else:
            continue
    return train_triple

def rewrite_kg(path,filename,train_triple):
    with open(path+filename,'w',encoding='utf-8') as file:
        file.write('h,r,t'+'\n')
        for triple in train_triple:
            file.write(triple[0]+','+triple[1]+','+triple[2]+'\n')


if __name__=='__main__':
    path = os.getcwd()
    train_triple_kg1 = process_kg1(path,'/kg_1_2017_2021.csv')
    rewrite_kg(path,'/kg1_test.txt',train_triple_kg1)