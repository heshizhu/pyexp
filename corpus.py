#encoding:utf-8

import os
import codecs

base_path = "G:/temp/TransX/fb15k/"

if __name__ == '__main__':
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

    for rel in relations:
        print rel, '\t', rel_tris[rel], '\t', len(rel_heads[rel]), '\t', len(rel_tails[rel]), '\t',\
        head_npt[rel], '\t', tail_nph[rel]