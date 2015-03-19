import cPickle
import time
import random
import numpy as np
from model import train_triple_fn, test_fn
from dev_test import output_result

base_path = 'G:/temp/TransX/wn11/TransE/'

print 'reading data....'
read_data_file_start = time.time()
# read train data
train_triples_idx_fid = open(base_path + '../data/train_triple_idx.pkl')
train_triples_idx = cPickle.load(train_triples_idx_fid)
train_triples_idx_fid.close()

negative_train_triples_idx_fid = open(base_path + '../data/negative_train_triple_idx_0.pkl')
negative_train_triples_idx_0 = cPickle.load(negative_train_triples_idx_fid)
negative_train_triples_idx_fid.close()

negative_train_triples_idx_fid = open(base_path + '../data/negative_train_triple_idx_1.pkl')
negative_train_triples_idx_1 = cPickle.load(negative_train_triples_idx_fid)
negative_train_triples_idx_fid.close()

negative_train_triples_idx = np.concatenate([negative_train_triples_idx_0, negative_train_triples_idx_1], axis=0)

negative_id_fid = open(base_path + '../data/negative_id.pkl')
negative_id = cPickle.load(negative_id_fid)
negative_id_fid.close()

# read dev and test data
dev_triples_idx_fid = open(base_path + '../data/dev_triple_idx.pkl')
dev_triples_idx = cPickle.load(dev_triples_idx_fid)
dev_triples_idx_fid.close()

test_triples_idx_fid = open(base_path + '../data/test_triple_idx.pkl')
test_triples_idx = cPickle.load(test_triples_idx_fid)
test_triples_idx_fid.close()

dev_triple_fid = open(base_path + '../data/dev_triple.pkl')
dev_triples = cPickle.load(dev_triple_fid)
dev_triple_fid.close()

test_triple_fid = open(base_path + '../data/test_triple.pkl')
test_triples = cPickle.load(test_triple_fid)
test_triple_fid.close()

#read entity and relations
entity_fid = open(base_path + '../data/entity-list.txt')
entity_list = []
for ele in entity_fid:
    entity_list.append(ele.strip('\n'))
entity_fid.close()

relation_fid = open(base_path + '../data/relation-list.txt')
relation_list = []
for ele in relation_fid:
    relation_list.append(ele.strip('\n'))
relation_fid.close()

read_data_file_end = time.time()

print 'train triple set size: ', train_triples_idx.shape[0]
print 'dev set size: ', dev_triples_idx.shape[0]
print 'test set size: ', test_triples_idx.shape[0]
print 'entity number: ', len(entity_list)
print 'relation number: ', len(relation_list)
print 'read data file spend %f s.' % (read_data_file_end - read_data_file_start)


def sample_minibatch(minibatch_size, train_triples_idx, negative_train_triples_idx):
    n_train_triples = train_triples_idx.shape[0]
    n_batches = int(n_train_triples / minibatch_size)
    total_idx = np.zeros((n_batches, minibatch_size, 7), 'int32')
    idx_all_triples = range(0, n_train_triples)
    sub_idx_range = range(2000)

    for i in xrange(n_batches):
        idx_sample = random.sample(idx_all_triples, minibatch_size)
        total_idx[i, :, 0:3] = train_triples_idx[idx_sample]

        tmp_neg_idx = negative_train_triples_idx[idx_sample]
        sub_idx = [random.choice(sub_idx_range) for j in xrange(minibatch_size)]
        total_idx[i, :, 3:] = tmp_neg_idx[range(minibatch_size), sub_idx]

    return total_idx


epoch = 2000
minibatch_size = 100
n_dim = 100
lamda = 0.01
n_batches = int(train_triples_idx.shape[0] / minibatch_size)

n_entity = len(entity_list)
n_relation = len(relation_list)
'''
r = 0.01
rng = np.random.RandomState(1234)
ent_embed_initial = np.asarray(rng.uniform(low=-r,high=r,size=(n_entity,n_dim)),dtype=theano.config.floatX)
rel_embed_initial = np.asarray(rng.uniform(low=-r,high=r,size=(n_entity,n_dim)),dtype=theano.config.floatX)

ent_embed = theano.shared(value = ent_embed_initial, \
                          name='ent_embed', borrow = True)
rel_embed = theano.shared(value = rel_embed_initial, \
                          name='rel_embed', borrow = True)
'''
ent_embed_fid = open(base_path + '../data/ent_embed_' + str(1649) + '.pkl')
ent_embed = cPickle.load(ent_embed_fid)
ent_embed_fid.close()
rel_embed_fid = open(base_path + '../data/rel_embed_' + str(1649) + '.pkl')
rel_embed = cPickle.load(rel_embed_fid)
rel_embed_fid.close()

train_triple = train_triple_fn(ent_embed, rel_embed, lamda)
test = test_fn(ent_embed, rel_embed)

best_acc = 0.5
for i in range(1649, 2000):
    print i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    t_idx = sample_minibatch(minibatch_size, train_triples_idx, negative_train_triples_idx)
    cost = [train_triple(t_idx[j]) for j in xrange(n_batches)]
    print 'mean cost:%4.4f:' % ( np.mean(cost))
    ent_embed_fid = open(base_path + '../data/ent_embed_' + str(i) + '.pkl', 'wb')
    cPickle.dump(ent_embed, ent_embed_fid, -1)
    ent_embed_fid.close()
    rel_embed_fid = open(base_path + '../data/rel_embed_' + str(i) + '.pkl', 'wb')
    cPickle.dump(rel_embed, rel_embed_fid, -1)
    rel_embed_fid.close()
    dev_F = test(dev_triples_idx)
    test_F = test(test_triples_idx)
    current_acc = output_result(dev_triples, dev_F, test_triples, test_F, relation_list)
    if (current_acc > best_acc):
        best_acc = current_acc
    print 'The highest acc is:%4.4f' % (best_acc)
