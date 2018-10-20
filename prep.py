import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize   
from hyperparameter import Hyperparameter as hp
from module import Module as M

  
def remove_stopwords(l):
	stop_words = set(stopwords.words('english')) 
	l_s = l.split(' ')
	filtered_sentence = [w for w in l_s if not w in stop_words] 
	return filtered_sentence

def prep_data(r, if_rp = False, if_rb = False, if_l = False, if_rs = False):
	#replace punctuations with space
	REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
	#replace line break with space
	REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
	#replace punctuations with space
	if if_rp:
		r = [REPLACE_NO_SPACE.sub("", line) for line in r]
	#replace line break with space
	if if_rb:
		r = [REPLACE_WITH_SPACE.sub(" ", line) for line in r]
	#lowercase
	if if_l:
		r = [line.lower() for line in r]
	#remove stopwords
	if if_rs:
		r = [remove_stopwords(line) for line in r]
	else:
		r = [line.split(' ') for line in r]
	output = r
	return output

def word_grams(sent, min=1, max=4):
	words = sent.split(' ')
	s = []
	for n in range(min, max):
		for ngram in ngrams(words, n):
			s.append(' '.join(str(i) for i in ngram))
	return s

#output list of list of words
def read_data(filename_p, filename_n, count = 9999999,if_rp = False, if_rb = False, if_l = False, if_rs = False):
	x = []
	y = []
	raw_data_p = []
	
	with open(filename_p, "r") as f:

		line_num_p = 0
		line_p = f.readline()
		while line_p != None and line_p != "" and line_num_p<count:
			line_num_p += 1
			raw_data_p.append(line_p)
			line_p = f.readline()
	raw_data_n = []
	with open(filename_n, "r") as f:
		line_num_n = 0
		line_n = f.readline()
		while line_n != None and line_n != "" and line_num_n<count:
			line_num_n += 1
			raw_data_n.append(line_n)
			line_n = f.readline()
			
	x = x + prep_data(raw_data_p, if_rp = if_rp, if_rb = if_rb, if_l = if_l, if_rs = if_rs)
	x = x + prep_data(raw_data_n, if_rp = if_rp, if_rb = if_rb, if_l = if_l, if_rs = if_rs)
	y = y + [1] * line_num_p
	y = y + [0] * line_num_n
	return x, y


def tokenize(sent):
  tokens = tokenizer(sent)
  return [token.text.lower() for token in tokens if (token.text not in punctuations)]


def tokenize_dataset(dataset, min=1, max=5):
	token_dataset = []
	# we are keeping track of all tokens in dataset 
	# in order to create vocabulary later
	all_tokens = []
	
	for sample in dataset:
		tokens = word_grams(sample, min=min, max=max)
		token_dataset.append(tokens)
		all_tokens += tokens

	return token_dataset, all_tokens

