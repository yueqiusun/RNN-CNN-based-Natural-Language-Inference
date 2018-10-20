import numpy as np
import torch

from module import Module as M
import pickle as pkl
import prep
from hyperparameter import Hyperparameter as hp
from moviesdataset import MoviesDataset

def load_raw_data(if_rp = False, if_rb = False, if_l = False, if_rs = False):



	movie_train_data, movie_train_target = prep.read_data(hp.prepath_data + hp.tp_filename, hp.prepath_data + hp.tn_filename, if_rp = if_rp, if_rb = if_rb, if_l = if_l, if_rs = if_rs)
	movie_test_data, movie_test_target = prep.read_data(hp.prepath_data + hp.tep_filename, hp.prepath_data + hp.ten_filename, if_rp = if_rp, if_rb = if_rb, if_l = if_l, if_rs = if_rs)

	train_split = 20000
	train_data = movie_train_data[:train_split]
	train_targets = movie_train_target[:train_split]

	val_data = movie_train_data[train_split:]
	val_targets = movie_train_target[train_split:]

	test_data = movie_test_data
	test_targets = movie_test_target

	print ("Train dataset size is {}".format(len(train_data)))
	print ("Val dataset size is {}".format(len(val_data)))
	print ("Test dataset size is {}".format(len(test_data)))

	return train_data, train_targets, val_data, val_targets, test_data, test_targets


def tokenization(train_data,val_data,test_data, min=1, max=5):
	#val set tokens
	print ("Tokenizing val data")
	val_data_tokens, _ = prep.tokenize_dataset(val_data, min=min, max=max)
	pkl.dump(val_data_tokens, open(hp.prepath_data + "val_data_tokens.p", "wb"))

	#test set tokens
	print ("Tokenizing test data")
	test_data_tokens, _ = prep.tokenize_dataset(test_data, min=min, max=max)
	pkl.dump(test_data_tokens, open(hp.prepath_data + "test_data_tokens.p", "wb"))

	#train set tokens
	print ("Tokenizing train data")
	train_data_tokens, all_train_tokens = prep.tokenize_dataset(train_data, min=min, max=max)
	pkl.dump(train_data_tokens, open(hp.prepath_data + "train_data_tokens.p", "wb"))
	pkl.dump(all_train_tokens, open(hp.prepath_data + "all_train_tokens.p", "wb"))
	print('Tokenization complete')

def load_tokens():

	# load preprocessed train, val and test datasets
	train_data_tokens = pkl.load(open(hp.prepath_data + "train_data_tokens.p", "rb"))
	all_train_tokens = pkl.load(open(hp.prepath_data + "all_train_tokens.p", "rb"))

	val_data_tokens = pkl.load(open(hp.prepath_data + "val_data_tokens.p", "rb"))
	test_data_tokens = pkl.load(open(hp.prepath_data + "test_data_tokens.p", "rb"))

	# double checking
	print ("Train dataset size is {}".format(len(train_data_tokens)))
	print ("Val dataset size is {}".format(len(val_data_tokens)))
	print ("Test dataset size is {}".format(len(test_data_tokens)))

	print ("Total number of tokens in train dataset is {}".format(len(all_train_tokens)))
	print('Token loaded')
	return train_data_tokens, val_data_tokens, test_data_tokens, all_train_tokens

def dataloader(train_data_indices, train_targets, val_data_indices, val_targets, test_data_indices, test_targets):
	hp.BATCH_SIZE = 32
	train_dataset = MoviesDataset(train_data_indices, train_targets)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
	                                           batch_size=hp.BATCH_SIZE,
	                                           collate_fn=M.movies_collate_func,
	                                           shuffle=True)

	val_dataset = MoviesDataset(val_data_indices, val_targets)
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
	                                           batch_size=hp.BATCH_SIZE,
	                                           collate_fn=M.movies_collate_func,
	                                           shuffle=True)

	test_dataset = MoviesDataset(test_data_indices, test_targets)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
	                                           batch_size=hp.BATCH_SIZE,
	                                           collate_fn=M.movies_collate_func,
	                                           shuffle=False)
	return train_loader, val_loader, test_loader





