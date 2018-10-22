# First lets improve libraries that we are going to be used in this lab session
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
from collections import Counter
random.seed(134)

from hyperparameter import Hyperparameter as hp


import re
 
from train_f import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 100
max_vocab_size = 20000
MAX_SENTENCE_LENGTH = 34


kernel_size = 3
learning_rate = 3e-4
hidden_size = 200
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--BATCH_SIZE', type=int, default=100,
                        help='')
    parser.add_argument('--max_vocab_size', type=int, default=20000,
                        help='')
    parser.add_argument('--words_to_load', type=int, default=50000,
                        help='')
    parser.add_argument('--print_freq', type=int, default=200,
                        help='')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='')
    parser.add_argument('--max_vocab_size', type=int, default=20000,
                        help='')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='')
    parser.add_argument('--model'
                        ,help='')
    
    return parser.parse_args()
args = parse_args()

BATCH_SIZE = args.BATCH_SIZE
max_vocab_size = args.max_vocab_size



print("arguments: %s" %(args))


def build_all_tokens(inp1, inp2):
    all_tokens = []
    for sent in inp1:
        for x in sent:
            all_tokens.append(x)   
    for sent in inp2:
        for x in sent:
            all_tokens.append(x)    
    
    return all_tokens
    
def read_data(filename, count = 9999999):
    x1 = []
    x2 = []
    y = []
    x1_w = []
    x2_w = []
    raw_data = []
    filename =hp.prepath_data + filename
    len_list = []
    with open(filename, "r") as f:

        line_num = 0
        line = f.readline()
        line = f.readline()
        while line != None and line != "" and line_num<count:
            line = line.lower()
            arr = line.split('\t')
            #list of length of  sentences
            x1.append(arr[0])
            x2.append(arr[1])
            y.append(hp.dummy2int[arr[2][:-1]])
            line_num += 1
            raw_data.append(line)
            line = f.readline()
        x1_w = [line.split() for line in x1]
        x2_w = [line.split() for line in x2]
        len_list = [len(line.split()) for line in x1]
        len_list += [len(line.split()) for line in x2]
        # take the 99 precentile of the length, let it be maximum sentence length
        if count < 1000:
            MAX_SENTENCE_LENGTH = sorted(len_list, reverse = True)[0]
        else:
            MAX_SENTENCE_LENGTH = sorted(len_list, reverse = True)[1000]
        a = sorted(len_list, reverse = True)
        
        all_tokens = build_all_tokens(x1_w, x2_w)
    return x1_w, x2_w, y, MAX_SENTENCE_LENGTH, all_tokens


def build_vocab(all_tokens):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token



def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data


import numpy as np
import torch
from torch.utils.data import Dataset
class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list1, data_list2,target_list):
        """
        @param data_list: list of SNLI tokens 
        @param target_list: list of SNLI targets 

        """
        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.target_list = target_list
        assert (len(self.data_list1) == len(self.target_list))

    def __len__(self):
        return len(self.data_list1)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx1 = self.data_list1[key][:MAX_SENTENCE_LENGTH]
        token_idx2 = self.data_list2[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx1, token_idx2, len(token_idx1), len(token_idx2),label]


def SNLI_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list1 = []
    data_list2 = []
    label_list = []
    length_list1 = []
    length_list2 = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[4])
        length_list1.append(datum[2])
        length_list2.append(datum[3])
    # padding
    for datum in batch:
        padded_vec1 = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[2])), 
                                mode="constant", constant_values=0)
        padded_vec2 = np.pad(np.array(datum[1]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[3])), 
                                mode="constant", constant_values=0)
        data_list1.append(padded_vec1)
        data_list2.append(padded_vec2)
    
    #transform labels to one hot label to fit into cross-entropy loss
    # define example
    
    
    
    return [torch.from_numpy(np.array(data_list1)), torch.from_numpy(np.array(data_list2)),
            torch.LongTensor(length_list1), torch.LongTensor(length_list2),torch.LongTensor(label_list)]

def create_emb_layer(weights_matrix, non_trainable=False):
    weights_matrix = torch.from_numpy(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
#     if non_trainable:
#         emb_layer.weight.requires_grad = False
    

    return emb_layer, num_embeddings, embedding_dim




    
class RNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, num_classes):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        #self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional = True) #dim1: batch dim2: sequence dim3: emb
        self.linear1 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, num_classes)
    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)

        return hidden
        
    def forward(self, x1,x2):
        # reset hidden state

        batch_size, seq_len = x1.size()
        
        self.hidden = self.init_hidden(batch_size)

        # get embedding of characters
        #embed = self.embedding(x)
        # pack padded sequence
        #embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.numpy(), batch_first=True)
        # fprop though RNN
#         m1 = [mask_vocab[i] for i in x1]
        embed1 = self.embedding(x1)
        embed2 = self.embedding(x2)
#         embed1 = m * e1 + (1-m) * e1.clone().detch()
#         embed2 = m * e2 + (1-m) * e2.clone().detch()
        rnn_out1, self.hidden1 = self.rnn(embed1, self.hidden) #batch_size * maximum sentence length * hidden_size
        rnn_out2, self.hidden2 = self.rnn(embed2, self.hidden)
        rnn_out1_last = rnn_out1[:,-1,:]   #batch_size * hidden_size
        rnn_out2_last = rnn_out2[:,-1,:]
        #print(rnn_out1.size())
        rnn_out = torch.cat((rnn_out1_last, rnn_out2_last),1)  #batch_size * hidden_size*2
        #rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        linear_out1 = self.linear1(rnn_out)
        logits = self.linear2(linear_out1)
        # undo packing
        #rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # sum hidden activations of RNN across time
        #rnn_out = torch.sum(rnn_out, dim=1)
        
        #logits = self.linear(rnn_out)
        return logits


#For the CNN, a 2-layer 1-D convolutional network with ReLU activations will suﬃce. 
#We can perform a max-pool at the end to compress the hidden representation into a single vector.
class CNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, kernel_size, num_classes):

        super(CNN, self).__init__()

        self.kernel_size, self.hidden_size = kernel_size, hidden_size
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)

        self.conv = nn.Sequential(nn.Conv1d(embedding_dim, hidden_size, kernel_size=kernel_size, padding=0),
                                   nn.ReLU(),
                                  nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=0),
                                   nn.ReLU())
        self.linear1 = nn.Linear(hidden_size*2, hidden_size*2)
        self.linear2 = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x1, x2):
        batch_size, seq_len = x1.size()

        embed1 = self.embedding(x1).transpose(2,1) #batch_size * embedding dim * maximum_sentence_length 
        embed2 = self.embedding(x2).transpose(2,1)
        #print('embed.size = ', embed1.size())
        out_conv1 = self.conv(embed1)  #batch_size * embedding dim * maximum_sentence_length - 2*kernel_size + 2
        out_conv2 = self.conv(embed2) 
        #print('out_conv1.size = ', out_conv1.size())
        out_pool1, _ = torch.max(out_conv1,dim = 2)
        out_pool2, _ = torch.max(out_conv2,dim = 2)
        #"print('out_pool1.size = ', out_pool1.size())
        out_pool = torch.cat((out_pool1, out_pool2), dim = 1)
        linear_out1 = self.linear1(out_pool)
        logits = self.linear2(linear_out1)
#         hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
#         hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))

#         hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
#         hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))

#         hidden = torch.sum(hidden, dim=1)
#         logits = self.linear(hidden)
        return logits


def main():
    count = 9999999999
    words_to_load = args.words_to_load


    train_x1, train_x2, train_y, MAX_SENTENCE_LENGTH, all_train_tokens  = read_data('snli_train.tsv', count = count)
    val_x1, val_x2, val_y, _, _ = read_data('snli_val.tsv', count = count)

    token2id, id2token = build_vocab(all_train_tokens)

    train_x1_indices = token2index_dataset(train_x1, token2id)
    train_x2_indices = token2index_dataset(train_x2, token2id)
    val_x1_indices = token2index_dataset(val_x1, token2id)
    val_x2_indices = token2index_dataset(val_x2, token2id)

    # double checking
    print ("Train dataset 1 size is {}".format(len(train_x1_indices)))
    print ("Train dataset 2 size is {}".format(len(train_x2_indices)))
    print ("Val dataset 1 size is {}".format(len(val_x1_indices)))
    print ("Val dataset 2 size is {}".format(len(val_x2_indices)))
    ft_home = './'
    

    with open(ft_home + 'wiki-news-300d-1M.vec') as f:
        loaded_embeddings_ft = np.zeros((len(id2token), 300))
        words_ft = {}
        idx2words_ft = {}
        ordered_words_ft = []
        for i, line in enumerate(f):
            if i >= words_to_load: 
                break
            s = line.split()
            #if the word in vocabulary is in fasttext, we load the embedding for that word.
            if s[0] in token2id:
                idx = token2id[s[0]]
                loaded_embeddings_ft[idx] = np.asarray(s[1:])
    #if the word in vocabulary is not in fasttext(include unk and pad), we initialize a random vector.
    # mask_emb = np.zeros(len(id2token))
    
    for i in range(len(id2token)):
        if loaded_embeddings_ft[i][0] == 0:

            # mask_emb[i] = 1
            loaded_embeddings_ft[i] = np.zeros((emb_dim, ))

    #data loader
    train_dataset = SNLIDataset(train_x1_indices, train_x2_indices,train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=SNLI_collate_func,
                                               shuffle=True)

    val_dataset = SNLIDataset(val_x1_indices, val_x2_indices, val_y)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=SNLI_collate_func,
                                               shuffle=True)
    print('Data Loaded')
    kernel_size = args.kernel_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    def test_model(loader, model):
        """
        Help function that tests the model's performance on a dataset
        @param: loader - data loader for the dataset to test against
        """
        correct = 0
        total = 0
        model.eval()
        for data1,data2,lengths1,length2,labels in loader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            labels = labels.to(device)
            outputs = F.softmax(model(data1, data2), dim=1)
            #print('outputs.size = ', outputs.size())
            predicted = outputs.max(1, keepdim=True)[1]

            total += labels.size(0)
            #print('labels.size = ', labels.size())
            #print('predicted.size = ', predicted.size())
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    #         print('correct = ', correct)
    #         print('total= ', total)
        return (100 * correct / total)

    if args.model == 'RNN'
        model = RNN(weights_matrix=loaded_embeddings_ft, hidden_size=hidden_size, num_layers=1, num_classes=3).to(device)
    elif args.model == 'CNN'
        model = CNN(weights_matrix=loaded_embeddings_ft, hidden_size=hidden_size, kernel_size=kernel_size, num_classes=3).to(device)

    print('Model Built, start trainning')
     # number epoch to train

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    print_freq = args.print_freq
    for epoch in range(num_epochs):
        
        losses = AverageMeter()
        for i, (x1, x2, lengths1, lengths2, labels) in enumerate(train_loader):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            losses.update(loss, x1.size(0))
            # Backward and optimize
            loss.backward()
            optimizer.step()
            #validate every print_freq iterations
            if i > 0 and i % print_freq == 0:
                # validate
                val_acc = test_model(val_loader, model)
                print(' Epoch: [{}/{}], Step: [{}/{}], Training loss: {loss.avg:.4f}, Validation Acc: {}'.format(
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc, loss = losses))


if __name__ == '__main__':
    main()

