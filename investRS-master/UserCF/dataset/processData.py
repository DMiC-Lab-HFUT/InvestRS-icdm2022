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

if __name__=='__main__':
    path = os.getcwd()
    entity2id = get_entity2id(path, '/entity2id_D.txt')
    relation2id = get_entity2id(path, '/relation2id_D.txt')
    place_list, place_name = get_place(path, '/placeSet.txt', entity2id)
    id2entity = get_id2entity(path, '/entity2id_D.txt')



    train_triple_name,train_triple = duplicate_remove3(path, '/kg5_train_D.txt', entity2id, relation2id,place_name)
    test_triple_name= duplicate_remove4(path, '/kg5_test.txt', entity2id, relation2id,place_name)

    company2C_train = duplicate_remove2(path, '/kg2_train_D.txt', entity2id, relation2id)
    company2C_test = duplicate_remove2(path, '/kg2_test.txt', entity2id, relation2id)
    company2C_train_1 = company2C_train|company2C_test
    c2Sc,Sc2c = get_C2C_info_train(company2C_train_1)


    # write_train_graph(path,'/train_graph.txt',train_triple,id2entity,Sc2c,entity2id)

    write_rating_train(path,'/rating_train_static.csv',train_triple,id2entity,Sc2c)
    write_rating_train1(path, '/rating_test.csv', test_triple_name, entity2id, Sc2c)
    write_person2id(path,'/person2id.csv',train_triple,entity2id,company2C_train,id2entity)

    print(1111)