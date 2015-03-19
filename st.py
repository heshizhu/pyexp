import numpy as np
import random
import time
import cPickle


base_path = 'G:/temp/TransX/wn11/TransE/'

read_file_start = time.time()

entity_list = []
relation_list = []

# read train set triples
train_data = open(base_path + '../data/train.txt')
train_list = []
for contend in train_data:
    con = contend.strip('\n')
    train_list.append(con)
    con_list = con.split('\t')
    entity_list.append(con_list[0])
    entity_list.append(con_list[2])
    relation_list.append(con_list[1])
train_data.close()
train_set = set(train_list)

# read dev set triples
dev_triples = []
dev_data = open(base_path + '../data/dev.txt')
for contend in dev_data:
    dev_triples.append(contend.strip('\n'))
    con_list = contend.strip('\n').split('\t')
    entity_list.append(con_list[0])
    entity_list.append(con_list[2])
    relation_list.append(con_list[1])
dev_data.close()

#read test set triples
test_triples = []
test_data = open(base_path + '../data/test.txt')
for contend in test_data:
    test_triples.append(contend.strip('\n'))
    con_list = contend.strip('\n').split('\t')
    entity_list.append(con_list[0])
    entity_list.append(con_list[2])
    relation_list.append(con_list[1])
test_data.close()

rule = []
entity_list = list(set(entity_list))
relation_list = list(set(relation_list))

#save entity and relations
entity_fid = open(base_path + '../data/entity-list.txt', 'w')
for ele in entity_list:
    entity_fid.write(ele + '\n')
entity_fid.close()

relation_fid = open(base_path + '../data/relation-list.txt', 'w')
for ele in relation_list:
    relation_fid.write(ele + '\n')
relation_fid.close()
read_file_end = time.time()

entity2id = {}
id2entity = {}
relation2id = {}
id2relation = {}

count = 0
for e in entity_list:
    entity2id[e] = count
    id2entity[count] = e
    count += 1

count = 0
for r in relation_list:
    relation2id[r] = count
    id2relation[count] = r
    count += 1

#save dev and test set
dev_triple_fid = open(base_path + '../data/dev_triple.pkl', 'wb')
cPickle.dump(dev_triples, dev_triple_fid, -1)
dev_triple_fid.close()

test_triple_fid = open(base_path + '../data/test_triple.pkl', 'wb')
cPickle.dump(test_triples, test_triple_fid, -1)
test_triple_fid.close()
print 'read file time spends %f s' % (read_file_end - read_file_start)

n_relation = len(relation_list)
n_entity = len(entity_list)
head = [{} for i in xrange(n_relation)]
tail = [{} for i in xrange(n_relation)]

for i in xrange(n_relation):
    for j in xrange(n_entity):
        head[i][j] = 0
        tail[i][j] = 0

for ele in train_list:
    e = ele.split('\t')
    h = e[0]
    r = e[1]
    t = e[2]
    h_id = entity2id[h]
    r_id = relation2id[r]
    t_id = entity2id[t]
    head[r_id][h_id] += 1
    tail[r_id][t_id] += 1

head_r = [[] for i in xrange(n_relation)]
tail_r = [[] for i in xrange(n_relation)]
for i in xrange(n_relation):
    entity_head_set = set()
    entity_tail_set = set()
    head_sum = 0
    tail_sum = 0
    for j in xrange(n_entity):
        if (head[i][j] != 0):
            entity_head_set.add(j)
            head_sum += head[i][j]
        if (tail[i][j] != 0):
            entity_tail_set.add(j)
            tail_sum += tail[i][j]
    head_r[i] = 1. * head_sum / len(entity_head_set)
    tail_r[i] = 1. * tail_sum / len(entity_tail_set)

all_triple_set = set(train_list + dev_triples + test_triples)
construct_negative_start = time.time()
train_triple = train_list
train_rule = rule
neg_num = 2000
train_triple_idx = np.zeros((len(train_triple), 3), 'int32')
dev_triple_idx = np.zeros((len(dev_triples), 3), 'int32')
test_triple_idx = np.zeros((len(test_triples), 3), 'int32')
negative_train_triple_idx = np.zeros((len(train_triple), neg_num, 4), 'int32')

for i in xrange(len(dev_triples)):
    ele = dev_triples[i]
    e = ele.split('\t')
    e1_idx = entity2id[e[0]]
    rel_idx = relation2id[e[1]]
    e2_idx = entity2id[e[2]]
    dev_triple_idx[i] = np.asarray([e1_idx, rel_idx, e2_idx], 'int32')

for i in xrange(len(test_triples)):
    ele = test_triples[i]
    e = ele.split('\t')
    e1_idx = entity2id[e[0]]
    rel_idx = relation2id[e[1]]
    e2_idx = entity2id[e[2]]
    test_triple_idx[i] = np.asarray([e1_idx, rel_idx, e2_idx], 'int32')

for i in xrange(len(train_triple)):
    print i
    ele = train_triple[i]
    e = ele.split('\t')
    e1_idx = entity2id[e[0]]
    rel_idx = relation2id[e[1]]
    e2_idx = entity2id[e[2]]
    train_triple_idx[i] = np.asarray([e1_idx, rel_idx, e2_idx], 'int32')
    negative_idx = np.zeros((neg_num, 4), 'int32')
    for j in xrange(neg_num):
        rt = random.random()
        while (True):
            neg_entity_idx = random.randint(0, n_entity - 1)
            tmp_triple_l = entity_list[neg_entity_idx] + '\t' + e[1] + '\t' + e[2]
            tmp_triple_r = e[0] + '\t' + e[1] + '\t' + entity_list[neg_entity_idx]
            if (rt < head_r[rel_idx] / (head_r[rel_idx] + tail_r[rel_idx]) and \
                            tmp_triple_l not in all_triple_set ):
                negative_idx[j] = np.asarray([neg_entity_idx, rel_idx, e2_idx, neg_entity_idx], 'int32')
                break
            elif (rt >= head_r[rel_idx] / (head_r[rel_idx] + tail_r[rel_idx]) and \
                              tmp_triple_r not in all_triple_set ):
                negative_idx[j] = np.asarray([e1_idx, rel_idx, neg_entity_idx, neg_entity_idx], 'int32')
                break
            else:
                continue

    negative_train_triple_idx[i] = negative_idx
construct_negative_end = time.time()
print 'construct 2000 negative for one triple time spends %f s' % (construct_negative_end - construct_negative_start)

#save train_triple_idx, dev_triple_idx, test_triple_idx
train_triple_idx_fid = open(base_path + '../data/train_triple_idx.pkl', 'wb')
cPickle.dump(train_triple_idx, train_triple_idx_fid, -1)
train_triple_idx_fid.close()

dev_triple_idx_fid = open(base_path + '../data/dev_triple_idx.pkl', 'wb')
cPickle.dump(dev_triple_idx, dev_triple_idx_fid, -1)
dev_triple_idx_fid.close()

test_triple_idx_fid = open(base_path + '../data/test_triple_idx.pkl', 'wb')
cPickle.dump(test_triple_idx, test_triple_idx_fid, -1)
test_triple_idx_fid.close()

negative_train_triple_idx_fid = open(base_path + '../data/negative_train_triple_idx_0.pkl', 'wb')
cPickle.dump(negative_train_triple_idx[0:60000], negative_train_triple_idx_fid, -1)
negative_train_triple_idx_fid.close()

negative_train_triple_idx_fid = open(base_path + '../data/negative_train_triple_idx_1.pkl', 'wb')
cPickle.dump(negative_train_triple_idx[60000:], negative_train_triple_idx_fid, -1)
negative_train_triple_idx_fid.close()




