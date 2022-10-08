import numpy as np
import random
import copy
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def seq_batch(phrase_id, args, phrase2word):
    phrase_batch = np.ones((len(phrase_id),16),dtype = int)*args.pad_id
    phrase_len = torch.LongTensor(len(phrase_id))
    
    for i,ID in enumerate(phrase_id):
        phrase_batch[i,0:len(phrase2word[ID])] = np.array(phrase2word[ID])
        phrase_len[i] = len(phrase2word[ID])
        
    phrase_batch = torch.from_numpy(phrase_batch)
    phrase_batch = Variable(torch.LongTensor(phrase_batch))
    phrase_len = Variable(phrase_len)
    
    if args.use_cuda:
        phrase_batch = phrase_batch.cuda()
        phrase_len = phrase_len.cuda()
    
    return phrase_batch, phrase_len


def get_neg_samples(pos_samples, unq_ent, unq_rels, args):
    # pdb.set_trace()
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * args.neg_samples
    neg_samples = np.tile(pos_samples, (args.neg_samples, 1))

    values = np.random.choice(unq_ent, size=num_to_generate)
    neg_samples[:, 2] = values

    num_to_generate_rel = size_of_batch * args.rel_neg_samples
    rel_neg_samples = np.tile(pos_samples, (args.rel_neg_samples, 1))

    rel_values = np.random.choice(unq_rels, size=num_to_generate_rel)
    rel_neg_samples[:, 1] = rel_values

    pos_labels = np.ones(len(pos_samples))
    neg_labels = np.zeros(len(neg_samples)+len(rel_neg_samples))

    return np.concatenate((pos_samples, neg_samples, rel_neg_samples)), np.concatenate((pos_labels, neg_labels))


def get_next_batch(id_list, data, args, train, node_id, rel_id):
    # pdb.set_trace()
    pos_samples = train[id_list]
    tail_ents = set()
    rels = set()

    for i in range(len(id_list)):
        trip = train[id_list[i]]
        pos_ids = data.label_graph[(trip[0],trip[1])]

        tail_ents.update(pos_ids)
        tail_ents.add(trip[0])
        tail_ents.add(trip[2])

        pos_ids_rel = data.label_graph_rel[(trip[0], trip[2])]
        rels.update(pos_ids_rel)
        rels.add(trip[1])

    # pdb.set_trace()
    unq_ents = node_id.difference(tail_ents)
    unq_ents = list(unq_ents)

    unq_rels = rel_id.difference(rels)
    unq_rels = list(unq_rels)
    samples, labels = get_neg_samples(pos_samples, unq_ents, unq_rels, args)

    return samples, labels


def get_next_batch_test(head, rel, tail, args, data, node_id, rel_id):
    pos_samples = []
    tail_ents = set()
    rels = set()
    for i in range(len(head)):
        pos_samples.append([head[i], rel[i], tail[i]])

        if (head[i], rel[i]) in data.label_graph:
            pos_ids = data.label_graph[(head[i], rel[i])]
            tail_ents.update(pos_ids)

        tail_ents.add(head[i])
        tail_ents.add(tail[i])

        if (head[i], tail[i]) in data.label_graph_rel:
            pos_ids_rel = data.label_graph_rel[(head[i], tail[i])]
            rels.update(pos_ids_rel)
        rels.add(rel[i])

    unq_ents = node_id.difference(tail_ents)
    unq_ents = list(unq_ents)

    unq_rels = rel_id.difference(rels)
    unq_rels = list(unq_rels)

    samples, labels = get_neg_samples(pos_samples, unq_ents, unq_rels, args)

    return np.array(samples), np.array(labels)

def evaluate_neg(model, entTotal, num_rels, test_trips, args, data):
    ents = torch.arange(0, entTotal, dtype=torch.long)
    head = test_trips[:, 0]
    rel = test_trips[:, 1]
    tail = test_trips[:, 2]
    bs = args.batch_size
    node_id_list = set([m for m in range(args.num_nodes)])
    rel_id_list = set([m for m in range(args.num_rels)])

    if args.use_cuda:
        ents = ents.cuda()

    acc = 0.0
    n_batches = int(test_trips.shape[0]/bs) + 1

    for i in range(n_batches):
        ent_head = head[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        r = rel[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        ent_tail = tail[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        samples, labels = get_next_batch_test(ent_head, r, ent_tail, args, data, node_id_list, rel_id_list)
        samples = Variable(torch.from_numpy(samples))
        labels = Variable(torch.from_numpy(labels).float())
        if args.use_cuda:
            samples = samples.cuda()
            labels = labels.cuda()

        scores, batch_acc = model.get_loss_my(samples, labels, ents)
        acc += batch_acc["acc_1"]

    acc = acc / n_batches
    return acc
