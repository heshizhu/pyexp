#encoding:utf-8

import os
import codecs

base_path = "G:/temp/TransX/fb15k_380/"

if __name__ == '__main__':
    with codecs.open(base_path + 'data/relations.txt', encoding='utf-8') as f:
        relations = set([l.strip() for l in f.readlines()])
    print len(relations)
    for name in ['train', 'valid', 'test']:
        with codecs.open('%sdata/%s.txt' % (base_path, name), encoding='utf-8') as f:
            lines = [l for l in f.readlines() if l.split('\t')[1].strip() in relations]
        with codecs.open('%sdata/%s_fil.txt' % (base_path, name), 'w', encoding='utf-8') as f:
            f.writelines(lines)



def corpus_statistic():
    triples, relations = list(), list()
    rel_heads, rel_tails, rel_tris = dict(), dict(), dict()
    head_npt, tail_nph = dict(), dict()
    rel_head_tails, rel_tail_heads = dict(), dict()

    with codecs.open(base_path + 'data/train.txt', encoding='utf-8') as f:
        for(t1, t2, t3) in [line.strip().split('\t') for line in f.readlines()]:
            triples.append((t1, t2, t3))
            relations.append(t2)
    relations = list(set(relations))
    for rel in relations:
        rel_heads[rel], rel_tails[rel], rel_tris[rel] = set(), set(), 0
        rel_head_tails[rel], rel_tail_heads[rel] = dict(), dict()
    for(t1, t2, t3) in triples:
        if not rel_head_tails[t2].has_key(t1):
            rel_head_tails[t2][t1] = set()
        if not rel_tail_heads[t2].has_key(t3):
            rel_tail_heads[t2][t3] = set()
        rel_heads[t2].add(t1)
        rel_tails[t2].add(t3)
        rel_tris[t2] += 1
        rel_head_tails[t2][t1].add(t3)
        rel_tail_heads[t2][t3].add(t1)
    for rel in relations:
        head_npt[rel] = sum(map(lambda x : len(x), rel_tail_heads[rel].values())) * 1.0 / len(rel_tail_heads[rel])
        tail_nph[rel] = sum(map(lambda x : len(x), rel_head_tails[rel].values())) * 1.0 / len(rel_head_tails[rel])

    count = 0
    for rel in relations:
        if len(rel_heads[rel]) == 1 or len(rel_tails[rel]) == 1:
            continue
        if rel_tris[rel] < 100:
            continue
        print rel, '\t', rel_tris[rel], '\t', len(rel_heads[rel]), '\t', len(rel_tails[rel]), '\t',\
        head_npt[rel], '\t', tail_nph[rel]
        count += 1
    print '个数：', count


#过滤规则
#关系不同head和tail个数为1的过滤
#三元组个数少于100次
