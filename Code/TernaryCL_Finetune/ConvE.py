from utils import *
from encoder import GRUEncoder
import torch.nn.functional as F
import torch
import time


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
		self.loss = torch.nn.BCELoss()

		self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.bn2 = torch.nn.BatchNorm1d(self.args.nfeats)
		self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.args.num_nodes)))
		self.fc = torch.nn.Linear(16128,self.args.nfeats)

	def forward(self, x, edges):
		return self.cn(x, edges)

	def get_scores(self,ent,rel,ent_embed,batch_size):
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
		x = torch.mm(x, ent_embed.transpose(1,0))
		x += self.b.expand_as(x)
		return x

	def get_embed_my(self, samples, node_id, e_batch, e_len, r_batch, r_len):

		np_embed = self.np_embeddings(node_id)
		np_embed_batch = self.phrase_embed_model(e_batch, e_len)

		rp_embed_batch = self.phrase_embed_model(r_batch, r_len)

		sub_embed = np_embed[samples[:, 0]]
		sub_embed = sub_embed + np_embed_batch
		rel_embed = rp_embed_batch
		scores = self.get_scores(sub_embed, rel_embed, np_embed, len(samples))

		return scores

	def get_loss_my(self, samples, labels, node_id, e_batch, e_len, r_batch, r_len):

		scores= self.get_embed_my(samples, node_id, e_batch, e_len, r_batch, r_len)
		pred = F.sigmoid(scores)

		predict_loss = self.loss(pred, labels)

		return predict_loss