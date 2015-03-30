#encoding:utf-8
import os
import codecs
import numpy as np
import theano
import theano.tensor as T
import random

os.chdir("/Users/hesz/temp/TransX/fb13/TransE/")

ent2id, id2ent = dict(), dict()
rel2id, id2rel = dict(), dict()

rel_subs, rel_objs = dict(), dict()
head_npt, tail_nph = dict(), dict() # head_npt为平均每个tail有多少个head
train_tri, valid_tri, test_tri = list(), list(), list()

# loading data
with codecs.open('../data/entity2id.txt', encoding='utf-8') as f:
    for(ent, id) in [line.strip().split('\t') for line in f.readlines()]:
        ent2id[ent] = int(id)
        id2ent[int(id)] = ent
with codecs.open('../data/relation2id.txt', encoding='utf-8') as f:
    for(rel, id) in [line.strip().split('\t') for line in f.readlines()]:
        rel2id[rel] = int(id)
        id2rel[int(id)] = rel
        rel_subs[int(id)], rel_objs[int(id)] = set(), set()
with codecs.open('../data/train.txt', encoding='utf-8') as f:
    rel_head_tails, rel_tail_heads = dict(), dict()
    for rel in id2rel.keys():
        rel_head_tails[rel] = dict()
        rel_tail_heads[rel] = dict()
    for(t1, t2, t3) in [line.strip().split('\t') for line in f.readlines()]:
        sub, rel, obj = ent2id[t1], rel2id[t2], ent2id[t3]
        rel_subs[rel].add(sub)
        rel_objs[rel].add(obj)
        train_tri.append((sub, rel, obj))
        if not rel_head_tails[rel].has_key(sub):
            rel_head_tails[rel][sub] = set()
        rel_head_tails[rel][sub].add(obj)
        if not rel_tail_heads[rel].has_key(obj):
            rel_tail_heads[rel][obj] = set()
        rel_tail_heads[rel][obj].add(sub)
    for rel in id2rel.keys():
        head_npt[rel] = sum(map(lambda x : len(x), rel_tail_heads[rel].values())) * 1.0 / len(rel_tail_heads[rel])
        tail_nph[rel] = sum(map(lambda x : len(x), rel_head_tails[rel].values())) * 1.0 / len(rel_head_tails[rel])
with codecs.open('../data/valid.txt', encoding='utf-8') as f:
    for(t1, t2, t3) in [line.strip().split('\t') for line in f.readlines()]:
        valid_tri.append((ent2id[t1], rel2id[t2], ent2id[t3]))
with codecs.open('../data/test.txt', encoding='utf-8') as f:
    for(t1, t2, t3) in [line.strip().split('\t') for line in f.readlines()]:
        test_tri.append((ent2id[t1], rel2id[t2], ent2id[t3]))

ent_num, rel_num = len(ent2id), len(rel2id)
train_num, valid_num, test_num = len(train_tri), len(valid_tri), len(test_tri)
print "ent : " + str(ent_num)
print "rel : " + str(rel_num)
print "train : " + str(train_num)
print "valid : " + str(valid_num)
print "test : " + str(test_num)

class Param:
    def __init__(self, epoch = 1000, batch = 120, dim = 50, margin = 1.0,
                 l1_flag = False, neg_method = True, neg_scope = False):
        self.epoch = epoch
        self.batch = batch
        self.l1_flag = l1_flag
        self.neg_method = neg_method
        self.neg_scope = neg_scope
        self.dim = dim
        self.margin = margin
        self.write_param()
    def toItems(self):
        items = dict()
        items['epoch'] = self.epoch
        items['batch'] = self.epoch
        items['l1_flag'] = self.l1_flag
        items['neg_method'] = self.neg_method
        items['neg_scope'] = self.neg_scope
        items['dim'] = self.dim
        items['margin'] = self.margin
        return items
    def write_param(self):
        if(os.path.exists("proc")):
            with codecs.open('proc', encoding='utf-8') as f:
                proc_id = int(f.readline()) + 1
        else:
            proc_id = 1
        with codecs.open('proc', 'w', encoding='utf-8') as f:
            f.write(str(proc_id))
        self.proc_id = proc_id
        with codecs.open('%d.param' % proc_id, 'w', encoding='utf-8') as f:
            f.writelines([('%s\t%s' % str(k), str(v)) for (k, v) in param.toItems().items()])
param = Param()

# 初始化参数
r = 0.01
rng = np.random.RandomState(1234)
ent_embed_initial = np.asarray(rng.uniform(low=-r,high=r,size=(ent_num, param.dim)),dtype=theano.config.floatX)
rel_embed_initial = np.asarray(rng.uniform(low=-r,high=r,size=(rel_num, param.dim)),dtype=theano.config.floatX)
# 向量长度为1

ent_embed_initial = T.sqrt(T.sum(T.sqr(ent_embed_initial), axis=1)) #是什么意思
rel_embed_initial = T.sqrt(T.sum(T.sqr(rel_embed_initial), axis=1))

ent_embed = theano.shared(value = ent_embed_initial, name='ent_embed', borrow = True)
rel_embed = theano.shared(value = rel_embed_initial, name='rel_embed', borrow = True)

def sampleNeg(pos_tri):
    neg_tri = pos_tri
    head_pro = 500
    if param.neg_method:
        head_pro = 1000 * tail_nph[pos_tri[1]] / (tail_nph[pos_tri[1]] + head_npt[pos_tri[1]])
    if random.randint(1, 1000) <= head_pro:
        neg_tri[0] = random.randint(len(id2ent))
    else:
        neg_tri[2] = random.randint(len(id2ent))
    return neg_tri



# 构造负样本
for epoch_id in xrange(param.epoch):
    for batch_id in xrange(int(train_num / param.batch)):
        pos_tris = random.sample(train_tri, param.batch)
        neg_tris = [sampleNeg(pt) for pt in pos_tris]






if __name__ == "__main__":
    print("start")
    for (k, v) in Param().toItems().items():
        print k, v
