#### Import all the supporting classes

import argparse
import numpy as np

from utils import *
from encoder import GRUEncoder
from data import load_data
from ConvE import ConvEParam
import torch
import time
import pdb

import os

def main(args):
	data = torch.load('pretrained/data.pth')
	self_triples = data.get_self_triples(data.id2ent, data.rel2id)
	data.train_trips, data.label_graph, data.neighbors_incoming, data.neighbors_outgoing = data.get_train_triples(
		data.data_files["train_trip_path"],
		data.entid2clustid, data.rel2id,
		data.id2rel, self_triples)
	args.pad_id = data.word2id['<PAD>']
	args.num_nodes = len(data.ent2id)
	args.num_rels = len(data.rel2id)
	if torch.cuda.is_available(): args.use_cuda = True
	else: args.use_cuda = False

	model = ConvEParam(args,data.embed_matrix,data.rel2word, data.ent2word)
	if args.use_cuda:
		model.cuda()

	model_dict = model.state_dict()
	load_pretrained_dict = torch.load('pretrained/neg_best_model.pth')
	load_pretrained_dict = load_pretrained_dict['state_dict']

	pretrained_dict = {k: v for k, v in load_pretrained_dict.items() if k in model_dict}

	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	model.eval()
	_, _, _ = evaluate(model, args.num_nodes, args.num_rels, data.valid_trips, args, data)
	_, _, _ = evaluate(model, args.num_nodes, args.num_rels, data.test_trips, args, data)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max',factor = 0.5, patience = 2)

	train_pairs = list(data.label_graph.keys())
	train_id = np.arange(len(train_pairs))

	node_id = torch.arange(0, args.num_nodes, dtype=torch.long)
	rel_id = torch.arange(0, args.num_rels, dtype=torch.long)
	if args.use_cuda:
		node_id = node_id.cuda()

	best_epoch = 0
	best_hit1 = 0.0
	best_hit10 = 0.0
	best_hit50 = 0.0
	count = 0
	for epoch in range(args.n_epochs):
		model.train()
		if count >= args.early_stop: break
		epoch_loss = 0
		permute = np.random.permutation(train_id)
		train_id = train_id[permute]
		n_batches = train_id.shape[0]//args.batch_size

		for i in range(n_batches):
			id_list = train_id[i*args.batch_size:(i+1)*args.batch_size]
			samples,labels, e_batch, e_len, r_batch, r_len = get_next_batch(id_list, data, args, train_pairs)

			samples = Variable(torch.from_numpy(samples))
			labels = Variable(torch.from_numpy(labels).float())
			if args.use_cuda:
				samples = samples.cuda()
				labels = labels.cuda()

			optimizer.zero_grad()
			loss = model.get_loss_my(samples, labels, node_id, e_batch, e_len, r_batch, r_len)

			loss.backward()
			print("batch {}/{} batches, batch_loss: {}".format(i,n_batches,(loss.data).cpu().numpy()),end='\r')
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
			optimizer.step()
			epoch_loss += (loss.data).cpu().numpy()
		print("epoch {}/{} total epochs, epoch_loss: {}".format(epoch+1,args.n_epochs,epoch_loss/n_batches))

		if epoch > 5:
			if (epoch + 1)%args.eval_epoch==0:
				model.eval()
				hit1, hit10, hit50 = evaluate(model, args.num_nodes, args.num_rels, data.valid_trips, args, data)
				if hit1 > best_hit1:
					count = 0
				elif (hit1 == best_hit1) and (hit10 > best_hit10):
					count = 0
				elif (hit1 == best_hit1) and (hit10 == best_hit10) and (hit50 > best_hit50):
					count = 0
				else:
					count += 1

				if count == 0:
					best_hit1 = hit1
					best_hit10 = hit10
					best_hit50 = hit50
					best_epoch = epoch
					torch.save(model, 'output/best_model.pth')
					torch.save(data, 'output/data.pth')
					torch.save(optimizer, 'output/optimizer.pth')
				# torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, 'output/best_model.pth')

				print("Best hit1: {}, Best epoch: {}".format(best_hit1, best_epoch))

			# scheduler.step(best_epoch)


	### Get Embeddings
	print("Test Set Evaluation ---")
	model = torch.load('output/best_model.pth')
	model.eval()
	_,_,_ = evaluate_final(model, args.num_nodes, args.num_rels, data.test_trips, args, data)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='TernaryCl')

	### Dataset choice
	parser.add_argument('-dataset', 	    dest='dataset', 	    default='ReVerb45K', help='Dataset Choice')

	### Data Paths
	parser.add_argument('-data_path',       dest='data_path',       default='../Data', 			help='Data folder')

	#### Hyper-parameters
	parser.add_argument('-nfeats',      dest='nfeats',       default=300,   type=int,       help='Embedding Dimensions')
	parser.add_argument('-nheads',      dest='nheads',       default=3,     type=int,       help='multi-head attantion in GAT')
	parser.add_argument('-num_layers',  dest='num_layers',   default=1,     type=int,       help='No. of layers in encoder network')
	parser.add_argument('-bidirectional',  dest='bidirectional',   default=True,     type=bool,       help='type of encoder network')
	parser.add_argument('-poolType',    dest='poolType',     default='last',choices=['last','max','mean'], help='pooling operation for encoder network')
	parser.add_argument('-dropout',     dest='dropout',      default=0.5,   type=float,     help='Dropout')
	parser.add_argument('-reg_param',   dest='reg_param',    default=0.0,   type=float,     help='regularization parameter')
	parser.add_argument('-lr',          dest='lr',           default=0.00005, type=float,     help='learning rate')
	parser.add_argument('-p_norm',      dest='p_norm',       default=1,     type=int,       help='TransE scoring function')
	parser.add_argument('-batch_size',  dest='batch_size',   default=128,   type=int,       help='batch size for training')
	parser.add_argument('-neg_samples', dest='neg_samples',  default=10,    type=int,       help='No of Negative Samples for TransE')
	parser.add_argument('-n_epochs',    dest='n_epochs',     default=500,   type=int,       help='maximum no. of epochs')
	parser.add_argument('-grad_norm',   dest='grad_norm',    default=1.0,   type=float,     help='gradient clipping')
	parser.add_argument('-eval_epoch',  dest='eval_epoch',   default=1,     type=int,       help='Interval for evaluating on validation dataset')
	parser.add_argument('-Hits',        dest='Hits',         default= [1,10,50,100],           help='Choice of n in Hits@n')
	parser.add_argument('-early_stop',  dest='early_stop',   default=50,    type=int,       help='Stopping training after validation performance stops improving')
	parser.add_argument('-my', dest='my', default=True, type=bool)
	

	args = parser.parse_args()


	args.data_files = {
	'ent2id_path'       : args.data_path + '/' + args.dataset + '/ent2id.txt',
	'rel2id_path'       : args.data_path + '/' + args.dataset + '/rel2id.txt',
	'train_trip_path'   : args.data_path + '/' + args.dataset + '/train_trip.txt',
	'test_trip_path'    : args.data_path + '/' + args.dataset + '/test_trip.txt',
	'valid_trip_path'   : args.data_path + '/' + args.dataset + '/valid_trip.txt',
	'gold_npclust_path' : args.data_path + '/' + args.dataset + '/gold_npclust.txt',
	'glove_path'        : '../glove/glove.6B.300d.txt'
	}

	args.model_path = "ConvE" + "-" + args.CN + "_modelpath.pth"

	SEED = 1
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	print(args.my)
	main(args)
