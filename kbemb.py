#encoding:utf-8
import os
import codecs


os.chdir("G:/temp/TransX/fb13/TransE/")

ent2id, id2ent = dict(), dict()
rel2id, id2rel = dict(), dict()

rel_subs, rel_objs = dict(), dict()
head_npt, tail_nph = dict(), dict() # head_npt为平均每个tail有多少个head
train_tri, valid_tri, test_tri = list()

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
    def __init__(self, epoch = 1000, batch = 120,
                 l1_flag = 0, neg_method = 1, neg_scope = 0,
                 dim = 50, margin = 1):
        self.epoch = epoch
        self.batch = batch
        self.l1_flag = l1_flag
        self.neg_method = neg_method
        self.neg_scope = neg_scope
        self.dim = dim
        self.margin = margin

#
# def read_param():
#     if(os.path.exists("proc")):
#         with codecs.open('proc', encoding='utf-8') as f:
#             proc_id = int(f.readline()) + 1
#     else:
#         proc_id = 1
#     with codecs.open('proc', 'w', encoding='utf-8') as f:
#         f.write(str(proc_id))


if __name__ == "__main__":
    print("start")
