import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os
from nltk import ngrams
import pickle

print("Loading the data : ")
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
test_data = np.load('./data/cb513+profile_split1.npy')
print("Original shape : ", train_data.shape)

def corpus_creation():
	train_data_n = np.reshape(train_data, [-1, 57])
	print(train_data_n.shape, train_data_n.shape[0] == 700 * train_data.shape[0])
	amino_acids = train_data_n[:, 0:21]
	print(amino_acids.shape)
	no_of_amino_acids = np.sum(amino_acids, axis = 0)
	print(no_of_amino_acids)
	t_no_of_amino_acids = np.sum(no_of_amino_acids)
	print(t_no_of_amino_acids)
	no_seq = train_data_n[:, 21]
	t_no_of_no_seq = np.sum(no_seq)
	print(t_no_of_amino_acids, t_no_of_no_seq, t_no_of_amino_acids + t_no_of_no_seq)
	amino_acids_with_no_seq = train_data_n[:, 0:22]
	amino_acids_no = np.argmax(train_data_n, 1)
	no_to_am_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
	am_acids_name = []
	for i in range(amino_acids_with_no_seq.shape[0]):
		amino_acid_no = amino_acids_no[i].tolist()
		am_acids_name.append(no_to_am_acid[amino_acid_no])
	amino_acids_total = 0
	no_seq_total = 0
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			no_seq_total += 1
		else:
			amino_acids_total += 1
	print("amino_acids_total", amino_acids_total)
	print("no_seq_total", no_seq_total)

	seqs = {}
	for i in range(5534):
		seqs[i] = ""
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			continue
		else:
			seqs[i // 700] += am_acid_name

	total_len_of_all_seqs = 0
	for i in range(5534):
		total_len_of_all_seqs += len(seqs[i])

	print("Total len verfn results : ", total_len_of_all_seqs == amino_acids_total)

	def verify(seq_no):
		curr_rec = train_data[seq_no, :]
		amino_acids_no_cr = []
		for i in range(700):
			amino_acids_no_cr.append(np.argmax(curr_rec[i*57 : i*57 + 22], 0))
		curr_rec_seq=""
		for no in amino_acids_no_cr:
			if no == 21:
				break
			curr_rec_seq += no_to_am_acid[no]
		return curr_rec_seq == seqs[seq_no]

	verify(0)
	ans = True
	for i in range(5534):
		ans = ans and verify(i)
	print("Verification results : ", ans)

	def unigram_corpus_creation():
		tot_seq = " dummy" * 12
		for i in range(5534):
			seq = seqs[i]
			spaced_seq = ""
			# print(seq)
			for x in seq:
				spaced_seq += " " + x
			tot_seq += spaced_seq + " dummy" * 12
			# print(tot_seq)
		file_path = "./data/unigram_corpus"
		# print(tot_seq[:2000])
		# print(tot_seq[len(tot_seq)-2000:len(tot_seq)])
		if(os.path.exists(file_path)):
			return
		with open(file_path,'w') as f:
			f.write(tot_seq)

	def trigram_corpus_creation():
		tot_seq_0 = " dummy" * 12
		tot_seq_1 = ""
		tot_seq_2 = ""
		
		for i in range(5534):
			seq = seqs[i]
			seq = "$" + seq + "$"
			spaced_seq = ""
			for x in seq:
				spaced_seq += " " + x
			
			trigrams = ngrams(spaced_seq.split(), 3)
			tri_seq_list_0 = []
			tri_seq_list_1 = []
			tri_seq_list_2 = []

			counter_gram = 0
			for gram in trigrams:
				if(counter_gram%3 == 0):
					tri_seq_list_0.append(gram)
				if(counter_gram%3 == 1):
					tri_seq_list_1.append(gram)
				if(counter_gram%3 == 2):
					tri_seq_list_2.append(gram)
				counter_gram += 1

			tri_seq_str_0 = ""
			tri_seq_str_1 = ""
			tri_seq_str_2 = ""

			for gram in tri_seq_list_0:
				tri_seq_str_0 += " " + gram[0] + gram[1] + gram[2] 
			for gram in tri_seq_list_1:
				tri_seq_str_1 += " " + gram[0] + gram[1] + gram[2] 
			for gram in tri_seq_list_2:
				tri_seq_str_2 += " " + gram[0] + gram[1] + gram[2] 
			
			tot_seq_0 += tri_seq_str_0 + " dummy" * 12		
			tot_seq_1 += tri_seq_str_1 + " dummy" * 12		
			tot_seq_2 += tri_seq_str_2 + " dummy" * 12		
		tot_seq = tot_seq_0 + tot_seq_1 + tot_seq_2
		file_path = "./data/trigram_corpus"
		# print(tot_seq[:2000])
		# print(tot_seq[len(tot_seq)-2000:len(tot_seq)])
		if(os.path.exists(file_path)):
			return
		with open(file_path,'w') as f:
			f.write(tot_seq)

	# trigram_corpus_creation()
	unigram_corpus_creation()
	# for i in range(5534):
	# 	print(len(seqs[i]))

corpus_creation()
