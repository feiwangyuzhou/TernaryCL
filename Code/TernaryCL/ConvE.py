from utils import *
from encoder import GRUEncoder
import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import pdb

from collections import OrderedDict

class Similarity(nn.Module):
	"""
	Dot product or cosine similarity
	"""

	def __init__(self, temp):
		super().__init__()
		self.temp = temp
		self.cos = nn.CosineSimilarity(dim=-1)

	def forward(self, x, y):
		return self.cos(x, y) / self.temp

def top_N(labels,logits,n=1):
	"""
	parameters:
		labels:N
		logits:N * (m+1)
	"""
	scores_dict=OrderedDict()
	for _batch_index,scores in enumerate(logits):
		sample_dict=dict()
		for _index,score in enumerate(scores):
			sample_dict[_index]=score
		sample_dict=sorted(sample_dict.items(),key = lambda x:x[1],reverse = True)
		results=[tuple_item[0] for tuple_item in sample_dict]
		scores_dict[_batch_index] = results
	all_scores = 0
	for _key,_value in scores_dict.items():
		n_value=_value[:n]
		if 0 in n_value:
			all_scores +=1
	acc_n = all_scores *1.0 / len(labels)
	return acc_n


def TopMetrics(top_list,label_list,logits_list):
	eval_metrics=dict()
	for num in top_list:
		key_metrics = "acc_%d"%(num)
		eval_metrics[key_metrics] = top_N(label_list, logits_list, num)
	return eval_metrics




class ConvEParam(torch.nn.Module):
	def __init__(self, args, embed_matrix,rel2words, ent2word=None):
		super(ConvEParam, self).__init__()
		self.args = args

		self.rel2words = rel2words
		self.ent2word = ent2word
		self.phrase_embed_model = GRUEncoder(embed_matrix, self.args)

		self.np_embeddings = torch.nn.Embedding(self.args.num_nodes, self.args.nfeats)
		torch.nn.init.xavier_normal_(self.np_embeddings.weight.data)
		self.rp_embeddings = torch.nn.Embedding(self.args.num_rels, self.args.nfeats)
		torch.nn.init.xavier_normal_(self.rp_embeddings.weight.data)

		self.inp_drop = torch.nn.Dropout(self.args.dropout)
		self.hidden_drop = torch.nn.Dropout(self.args.dropout)
		self.feature_map_drop = torch.nn.Dropout2d(self.args.dropout)
		# self.loss = torch.nn.BCELoss()
		self.loss_fct = torch.nn.CrossEntropyLoss()
		self.sim = Similarity(temp=self.args.temp)

		self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.bn2 = torch.nn.BatchNorm1d(self.args.nfeats)
		self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.args.num_nodes)))
		self.fc = torch.nn.Linear(16128,self.args.nfeats)

	def forward(self, x, edges):
		return self.cn(x, edges)

	def get_scores(self,ent,rel,tail_embed,batch_size):
		ent = ent.view(-1, 1, 15, 20)
		rel = rel.view(-1, 1, 15, 20)

		stacked_inputs = torch.cat([ent, rel], 2)

		stacked_inputs = self.bn0(stacked_inputs)
		x = self.inp_drop(stacked_inputs)
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.feature_map_drop(x)
		x = x.view(batch_size, -1)
		x = self.fc(x)
		x = self.hidden_drop(x)
		x = self.bn2(x)
		x = F.relu(x)
		# x = torch.mm(x, ent_embed.transpose(1,0))
		# x += self.b.expand_as(x)

		cos_sim = self.sim(x, tail_embed)
		return cos_sim

	def get_embed_my(self, samples, node_id):
		np_embed = self.np_embeddings(node_id)
		e_batch, e_len = seq_batch(samples[:, 0].cpu().numpy(), self.args, self.ent2word)
		np_embed_batch = self.phrase_embed_model(e_batch, e_len)

		# rp_embed = self.rp_embeddings(rel_id)
		r_batch, r_len = seq_batch(samples[:, 1].cpu().numpy(), self.args, self.rel2words)
		rp_embed_batch = self.phrase_embed_model(r_batch, r_len)

		sub_embed = np_embed[samples[:, 0]]
		sub_embed = sub_embed + np_embed_batch
		rel_embed = rp_embed_batch
		tail_embed = np_embed[samples[:, 2]]
		scores = self.get_scores(sub_embed, rel_embed, tail_embed, len(samples))

		return scores

	def get_loss_my(self, samples, labels, node_id):

		cos_sim= self.get_embed_my(samples, node_id)
		# pred = F.sigmoid(scores)
		# predict_loss = self.loss(pred, labels)
		cos_sim = cos_sim.view(self.args.neg_samples + self.args.rel_neg_samples + 1, -1).t()
		labels = torch.zeros(cos_sim.size(0)).long().cuda()
		predict_loss = self.loss_fct(cos_sim, labels)

		top_list = [1, 2, 5]
		eval_metrics = TopMetrics(top_list, labels, cos_sim)
		return predict_loss, eval_metrics