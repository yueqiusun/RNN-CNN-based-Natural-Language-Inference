import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from hyperparameter import Hyperparameter as hp

class Module():

    def token2index_dataset(tokens_data, token2id):
        indices_data = []
        for tokens in tokens_data:
            index_list = [token2id[token] if token in token2id else hp.UNK_IDX for token in tokens]
            indices_data.append(index_list)
        return indices_data

    def build_vocab(all_tokens, max_vocab_size):
        # Returns:
        # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
        # token2id: dictionary where keys represent tokens and corresponding values represent indices
        token_counter = Counter(all_tokens)
        vocab, count = zip(*token_counter.most_common(max_vocab_size))
        id2token = list(vocab)
        token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
        id2token = ['<pad>', '<unk>'] + id2token
        token2id['<pad>'] = hp.PAD_IDX 
        token2id['<unk>'] = hp.UNK_IDX
        return token2id, id2token

    

    def movies_collate_func(batch):
        """
        Customized function for DataLoader that dynamically pads the batch so that all 
        data have the same length
        """
        data_list = []
        label_list = []
        length_list = []
        #print("collate batch: ", batch[0][0])
        #batch[0][0] = batch[0][0][:hp.MAX_SENTENCE_LENGTH]
        for datum in batch:
            label_list.append(datum[2])
            length_list.append(datum[1])
        # padding
        for datum in batch:
            padded_vec = np.pad(np.array(datum[0]), 
                                    pad_width=((0,hp.MAX_SENTENCE_LENGTH-datum[1])), 
                                    mode="constant", constant_values=0)
            data_list.append(padded_vec)
        return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]
