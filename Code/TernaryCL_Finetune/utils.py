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
import pdb

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


def get_next_batch(id_list, data, args, train):
    entTotal = args.num_nodes
    samples = []
    labels = np.zeros((len(id_list),entTotal))
    for i in range(len(id_list)):
        trip = train[id_list[i]]
        samples.append([trip[0],trip[1]])
        pos_ids = list(data.label_graph[(trip[0],trip[1])])
        labels[i][pos_ids] = 1

    samples = np.array(samples)
    # pdb.set_trace()
    e_batch, e_len = seq_batch(samples[:, 0], args, data.ent2word)
    r_batch, r_len = seq_batch(samples[:, 1], args, data.rel2word)

    return samples,labels, e_batch, e_len, r_batch, r_len

def get_next_batch_test(head, rel, args, data):
    samples = []
    for i in range(len(head)):
        # pdb.set_trace()
        trip = [head[i], rel[i]]
        samples.append([trip[0],trip[1]])

    samples = np.array(samples)
    e_batch, e_len = seq_batch(samples[:, 0], args, data.ent2word)
    r_batch, r_len = seq_batch(samples[:, 1], args, data.rel2word)

    return samples, e_batch, e_len, r_batch, r_len

def get_rank_entity(scores, tail, Hits, entid2clustid, filter_clustID):
    hits = np.ones((len(Hits)))
    scores = np.argsort(scores)
    rank = 1
    for i in range(scores.shape[0]):
        if scores[i] == tail:
            break
        else:
            if entid2clustid[scores[i]] not in filter_clustID:
                rank += 1
    for i,r in enumerate(Hits):
        if rank>r: hits[i]=0
        else: break
    return rank,hits

def get_rank_mention(scores,clust,Hits,entid2clustid,filter_clustID):
    hits = np.ones((len(Hits)))
    scores = np.argsort(scores)
    rank = 1
    high_rank_clust = set()
    # pdb.set_trace()
    for i in range(scores.shape[0]):
        if scores[i] in clust: break
        else:
            if entid2clustid[scores[i]] not in high_rank_clust and entid2clustid[scores[i]] not in filter_clustID:
                rank+=1
                high_rank_clust.add(entid2clustid[scores[i]])
    for i,r in enumerate(Hits):
        if rank>r: hits[i]=0
        else: break
    return rank,hits

def evaluate(model, entTotal, num_rels, test_trips, args, data):
    ents = torch.arange(0, entTotal, dtype=torch.long)
    rel_id = torch.arange(0, num_rels, dtype=torch.long)
    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    head = test_trips[:,0]
    rel = test_trips[:,1]
    tail = test_trips[:,2]
    id2ent = data.id2ent
    id2rel = data.id2rel
    true_clusts = data.true_clusts
    entid2clustid = data.entid2clustid
    ent_filter = data.label_filter
    bs = args.batch_size

    if args.use_cuda: 
        ents = ents.cuda()
        rel_id = rel_id.cuda()
    
    test_scores = np.zeros((test_trips.shape[0],entTotal))
    n_batches = int(test_trips.shape[0]/bs) + 1

    for i in range(n_batches):
        ent = head[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        r = rel[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        samples, e_batch, e_len, r_batch, r_len = get_next_batch_test(ent, r, args, data)
        samples = Variable(torch.from_numpy(samples))
        if args.use_cuda:
            samples = samples.cuda()
        scores = model.get_embed_my(samples, ents, e_batch, e_len, r_batch, r_len)
        test_scores[i * bs:min((i + 1) * bs, test_trips.shape[0]), :] = scores.cpu().data.numpy()
        
    for j in range(test_trips.shape[0]):
        print("Evaluation Phase: sample {}/{} total samples".format(j + 1,test_trips.shape[0]),end="\r")
        sample_scores = -test_scores[j,:]
        t_clust = set(true_clusts[tail[j]])
        
        _filter = []
        if (head[j],rel[j]) in ent_filter: _filter = ent_filter[(head[j],rel[j])]

        if j % 2 == 1:
            H_r, H_h = get_rank_entity(sample_scores, tail[j], args.Hits, entid2clustid, _filter)
            H_Rank.append(H_r)
            H_inv_Rank.append(1 / H_r)
            H_Hits += H_h
        else:
            T_r, T_h = get_rank_entity(sample_scores, tail[j], args.Hits, entid2clustid, _filter)
            T_Rank.append(T_r)
            T_inv_Rank.append(1 / T_r)
            T_Hits += T_h

    print("Mean Rank: Head = {}  Tail = {}  Avg = {}"
          .format(np.mean(np.array(H_Rank)),np.mean(np.array(T_Rank)),(np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank)))/2))
    print("MRR: Head = {}  Tail = {}  Avg = {}"
          .format(np.mean(np.array(H_inv_Rank)),np.mean(np.array(T_inv_Rank)),(np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank)))/2))
    
    for i,hits in enumerate(args.Hits):
        print("Hits@{}: Head = {}  Tail={}  Avg = {}"
              .format(hits,H_Hits[i]/len(H_Rank),T_Hits[i]/len(H_Rank),(H_Hits[i] + T_Hits[i])/(2*len(H_Rank))))

    return (H_Hits[0] + T_Hits[0]) / (len(H_Rank)+len(T_Rank)), \
           (H_Hits[1] + T_Hits[1]) / (len(H_Rank)+len(T_Rank)), \
           (H_Hits[2] + T_Hits[2]) / (len(H_Rank)+len(T_Rank))



def evaluate_final(model, entTotal, num_rels, test_trips, args, data):
    ents = torch.arange(0, entTotal, dtype=torch.long)
    rel_id = torch.arange(0, num_rels, dtype=torch.long)
    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))

    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))

    H_Rank_mention = []
    H_inv_Rank_mention = []
    H_Hits_mention = np.zeros((len(args.Hits)))

    T_Rank_mention = []
    T_inv_Rank_mention = []
    T_Hits_mention = np.zeros((len(args.Hits)))


    head = test_trips[:, 0]
    rel = test_trips[:, 1]
    tail = test_trips[:, 2]
    id2ent = data.id2ent
    id2rel = data.id2rel
    true_clusts_ent = data.true_clusts
    entid2clustid = data.entid2clustid
    ent_filter = data.label_filter
    bs = args.batch_size

    if args.use_cuda:
        ents = ents.cuda()
        rel_id = rel_id.cuda()

    test_scores = np.zeros((test_trips.shape[0], entTotal))
    n_batches = int(test_trips.shape[0] / bs) + 1
    for i in range(n_batches):
        ent = head[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        r = rel[i * bs:min((i + 1) * bs, test_trips.shape[0])]

        samples, e_batch, e_len, r_batch, r_len = get_next_batch_test(ent, r, args, data)
        samples = Variable(torch.from_numpy(samples))
        if args.use_cuda:
            samples = samples.cuda()
        scores = model.get_embed_my(samples, ents, e_batch, e_len, r_batch, r_len)
        test_scores[i * bs:min((i + 1) * bs, test_trips.shape[0]), :] = scores.cpu().data.numpy()

    for j in range(test_trips.shape[0]):
        print("Evaluation Phase: sample {}/{} total samples".format(j + 1, test_trips.shape[0]), end="\r")
        sample_scores = -test_scores[j, :]

        _filter = []
        if (head[j], rel[j]) in ent_filter: _filter = ent_filter[(head[j], rel[j])]

        if j % 2 == 1:
            H_r, H_h = get_rank_entity(sample_scores, tail[j], args.Hits, entid2clustid, _filter)
            H_Rank.append(H_r)
            H_inv_Rank.append(1 / H_r)
            H_Hits += H_h

            H_t_clust_mention = set(true_clusts_ent[tail[j]])
            H_r_mention, H_h_mention = get_rank_mention(sample_scores, H_t_clust_mention, args.Hits, entid2clustid, _filter)
            H_Rank_mention.append(H_r_mention)
            H_inv_Rank_mention.append(1 / H_r_mention)
            H_Hits_mention += H_h_mention


        else:
            T_r, T_h = get_rank_entity(sample_scores, tail[j], args.Hits, entid2clustid, _filter)
            T_Rank.append(T_r)
            T_inv_Rank.append(1 / T_r)
            T_Hits += T_h

            T_t_clust_mention = set(true_clusts_ent[tail[j]])
            T_r_mention, T_h_mention = get_rank_mention(sample_scores, T_t_clust_mention, args.Hits, entid2clustid, _filter)
            T_Rank_mention.append(T_r_mention)
            T_inv_Rank_mention.append(1 / T_r_mention)
            T_Hits_mention += T_h_mention

    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write(str(np.mean(np.array(H_Rank))) + '\t')
        f.write(str(np.mean(np.array(H_inv_Rank))) + '\t')
        for i, hits in enumerate(args.Hits):
            f.write(str(H_Hits[i] / len(H_Rank))+ '\t')
        f.write('\n')
        f.write(str(np.mean(np.array(T_Rank))) + '\t')
        f.write(str(np.mean(np.array(T_inv_Rank))) + '\t')
        for i, hits in enumerate(args.Hits):
            f.write(str(T_Hits[i] / len(T_Rank)) + '\t')
        f.write('\n')


        f.write(str(np.mean(np.array(H_Rank_mention))) + '\t')
        f.write(str(np.mean(np.array(H_inv_Rank_mention))) + '\t')
        for i, hits in enumerate(args.Hits):
            f.write(str(H_Hits_mention[i] / len(H_Rank_mention)) + '\t')
        f.write('\n')
        f.write(str(np.mean(np.array(T_Rank_mention))) + '\t')
        f.write(str(np.mean(np.array(T_inv_Rank_mention))) + '\t')
        for i, hits in enumerate(args.Hits):
            f.write(str(T_Hits_mention[i] / len(T_Rank_mention)) + '\t')
        f.write('\n')

        f.write('***********************************************\n')


    return H_Hits[0] / len(H_Rank), H_Hits[1] / len(H_Rank), H_Hits[2] / len(H_Rank)
