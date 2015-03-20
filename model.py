# encoding:utf-8

import theano.tensor as T
import collections
import numpy as np
import theano

# sgd_adddelta
def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6):
    updates = collections.OrderedDict({})
    exp_sqr_grads = collections.OrderedDict({})
    exp_sqr_ups = collections.OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=np.asarray(empty, theano.config.floatX),
                                             name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=np.asarray(empty, theano.config.floatX),
                                           name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param
    return updates


def norm(matrix):
    return T.sqrt(T.sum(T.sqr(matrix), axis=1))


# given a mini-batch triples idx and entity + relation embeddings, compute |F|
def compute_triple_F(e1, rel, e2):
    return -norm(e1 + rel - e2)


def train_triple_fn(ent_embed, rel_embed, lamda):
    idx = T.imatrix('idx')

    h_pos = ent_embed[idx[:, 0]]
    rel_pos = rel_embed[idx[:, 1]]
    t_pos = ent_embed[idx[:, 2]]

    h_neg = ent_embed[idx[:, 3]]
    rel_neg = rel_embed[idx[:, 4]]
    t_neg = ent_embed[idx[:, 5]]

    pos_F = compute_triple_F(h_pos, rel_pos, t_pos)
    neg_F = compute_triple_F(h_neg, rel_neg, t_neg)

    cost = T.maximum(0, 1 - pos_F + neg_F).mean() + \
           lamda * (T.sqr(norm(rel_embed) - 1).sum() + \
                    T.sqr(norm(ent_embed) - 1).sum())

    updates = sgd_updates_adadelta([ent_embed, rel_embed], cost)
    return theano.function(inputs=[idx], \
                           outputs=cost, \
                           updates=updates, \
                           on_unused_input='ignore')


def test_fn(ent_embed, rel_embed):
    idx = T.imatrix('idx')
    h = ent_embed[idx[:, 0]]
    rel = rel_embed[idx[:, 1]]
    t = ent_embed[idx[:, 2]]

    pos_F = compute_triple_F(h, rel, t)

    return theano.function(inputs=[idx], \
                           outputs=pos_F, on_unused_input='ignore')