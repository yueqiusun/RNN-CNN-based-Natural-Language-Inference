class Hyperparameter:
	# save index 0 for unk and 1 for pad
	PAD_IDX = 0
	UNK_IDX = 1
	prepath_data = './hw2_data/'
	dummy2int = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
	BATCH_SIZE = 32
	MAX_SENTENCE_LENGTH = 800
