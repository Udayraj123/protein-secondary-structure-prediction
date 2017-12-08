import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os
from nltk import ngrams
import json as pickle

print("Loading the data : ")
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
test_data = np.load('./data/cb513+profile_split1.npy')
print("Original shape : ", train_data.shape)

def save_obj(obj,filename,overwrite=1):
	if(not overwrite and os.path.exists(filename)):
		return
 	with open(filename,'wb') as f:
 		pickle.dump(obj,f,mode="w")
 		print("File saved to " + filename)
#	pickle.dump(obj, filename)#, mode='w')
#	print("File saved to " + filename)
	
def load_obj(filename):
 	with open(filename) as f:
 		obj = pickle.load(f)
 		print("File loaded from " + filename)
 		return obj
# 	obj = pickle.load(filename)
# 	print("File loaded from " + filename)
# 	return obj

def read_glove_vec_files():
	file_path = './data/vectors_u.txt'
	file = open(file_path, 'r')
	word_to_glove = {}
	for line in file:
		line = line.split()	
		word = line[0]
		glove_vec = []
		for i in range(1, 101):
			glove_vec.append(float(line[i]))
		word_to_glove[word] = glove_vec
	# print(word_to_glove['L'])
	# print(word_to_glove['dummy'])
	# print(word_to_glove['X'])
	file.close()
	return word_to_glove


def raw_data_train_to_mini_batches():
	train_data_n = np.reshape(train_data, [-1, 57])
	print("Verifying shape of reshaped data")
	print(train_data_n.shape, train_data_n.shape[0] == 700 * train_data.shape[0])
	amino_acids = train_data_n[:, 0:21]
	amino_acids_seq_profile = train_data_n[:, 35:57]
	# print(amino_acids.shape)
	no_of_amino_acids = np.sum(amino_acids, axis = 0)
	# print(no_of_amino_acids)
	t_no_of_amino_acids = np.sum(no_of_amino_acids)
	# print(t_no_of_amino_acids)
	no_seq = train_data_n[:, 21]
	t_no_of_no_seq = np.sum(no_seq)
	
	print(t_no_of_amino_acids, t_no_of_no_seq, t_no_of_amino_acids + t_no_of_no_seq)
	amino_acids_with_no_seq = train_data_n[:, 0:22]
	amino_acids_str_with_no_seq = train_data_n[:, 22:31]
	str_wise_sum = np.sum(amino_acids_str_with_no_seq, axis = 0)
	amino_acids_str_present = np.sum(str_wise_sum[:8])
	amino_acids_dum_present = np.sum(str_wise_sum[8])
	amino_acids_str_no = np.argmax(amino_acids_str_with_no_seq, 1)
	print("Str wise sum : ", str_wise_sum)
	print("Str present, padded data : ", amino_acids_str_present, amino_acids_dum_present, amino_acids_str_present + amino_acids_dum_present)
	amino_acids_no = np.argmax(train_data_n, 1)
	no_to_am_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
	am_acids_name = []
	for i in range(amino_acids_with_no_seq.shape[0]):
		amino_acid_no = amino_acids_no[i].tolist()
		am_acids_name.append(no_to_am_acid[amino_acid_no])
	amino_acids_total = 0
	no_seq_total = 0
	amino_acids_x = 0
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			no_seq_total += 1
		else:
			if(am_acid_name == 'X'):
			  amino_acids_x += 1
			amino_acids_total += 1
	print("amino_acids_total", amino_acids_total)
	print("no_seq_total", no_seq_total)
	print("amino_acid_x_total", amino_acids_x)

	seqs = {}
	seq_pro = {}
	for i in range(5534):
		seqs[i] = ""
		seq_pro[i] = []

	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			continue
		else:
			seqs[i // 700] += am_acid_name
			seq_pro[i // 700].append(amino_acids_seq_profile[i].tolist())

	total_len_of_all_seqs = 0
	for i in range(5534):
		total_len_of_all_seqs += len(seqs[i])

	print("Total len verfn results : ", total_len_of_all_seqs == amino_acids_total)

	seqs_in_vec = []
	masks = []
	ops = []
	seq_len = []
	zeros_list = [0] * len(seq_pro[0][0])
	word_to_glove = read_glove_vec_files()
	for i in range(5534):
		seq = seqs[i]
		temp_seq = []
		temp_ops = []
		temp_msk = []

		for j in range(50):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove["dummy"])
			glove_and_seq_pro_list.extend(zeros_list)
			temp_seq.append(glove_and_seq_pro_list)
			# temp_seq.append(word_to_glove["dummy"])
			temp_ops.append(-1)
			temp_msk.append(0)

		for j in range(len(seq)):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove[seq[j]])
			glove_and_seq_pro_list.extend(seq_pro[i][j])
			temp_seq.append(glove_and_seq_pro_list)
			temp_ops.append(amino_acids_str_no[i*700 + j])
			temp_msk.append(1)

		for j in range(750 - len(seq)):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove["dummy"])
			glove_and_seq_pro_list.extend(zeros_list)
			temp_seq.append(glove_and_seq_pro_list)
			temp_ops.append(-1)
			temp_msk.append(0)

		seqs_in_vec.append(temp_seq)
		ops.append(temp_ops)
		masks.append(temp_msk)
		seq_len.append(len(seq) + 100)

	print("Reached line 284")
	ans = True
	count_masks_is_one = 0

	for j in range(5534):
		for i in range(800):
			if(masks[j][i] == 1):
				count_masks_is_one += 1
				ans = ans and (ops[j][i] != -1)
			else:
				ans = ans and (ops[j][i] == -1)

	for j in range(5534):
		ops_j = ops[j]
		for i in range(800):
			if(i<50 or i >= 50 + len(seqs[j])):
				ans = ans and (ops_j[i] == -1)
			else:
				ans = ans and (ops_j[i] != -1)

	ans = ans and ( count_masks_is_one == amino_acids_str_present)

	print("Verified the data inp, op and masks creation resuts : ", ans)

	batch_size = 128
	no_of_batches = 5534 // batch_size
	# 5534 // batch_size  = 43 for batch_size = 128
	# 5534 // batch_size  = 1106 for batch_size = 5
	# 0 - 42 batches with batch_size samples
	# 43 batch with 30 samples
	# 5504 + 30 samples in total
	mini_batch_data = {}
	print("Total number of batches : ", no_of_batches)
	for i in range(no_of_batches):
		temp = []
		if(i%5 == 0):
			print("Processing batch no : ", i)
		temp.append(seqs_in_vec[i * batch_size : (i + 1) * batch_size ])
		temp.append(ops[i * batch_size : (i + 1) * batch_size ])
		temp.append(masks[i * batch_size : (i + 1) * batch_size ])
		temp.append(seq_len[i * batch_size : (i + 1) * batch_size ])
		mini_batch_data[i] = temp

	# temp = []
	# temp.append(seqs_in_vec[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	# temp.append(ops[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	# temp.append(masks[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	# temp.append(seq_len[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	# mini_batch_data[no_of_batches] = temp

	# total_samples = 0
	# for i in range(no_of_batches + 1):
	# 	total_samples += len(mini_batch_data[i][0]) 
		# print(len(mini_batch_data[i][0]))
	# print(total_samples)
	
	save_obj(mini_batch_data, './data/batch_wise_train_data_' + str(batch_size) + '.pkl')


def raw_data_test_to_mini_batches():
	print("raw_test_data_to_mini_batches : ")
	test_data_n = test_data[:-1, :]
	test_data_n = np.reshape(test_data_n, [-1, 57])
	amino_acids = test_data_n[:, 0:21]
	amino_acids_seq_profile = test_data_n[:, 35:57]
	print(amino_acids.shape)
	no_of_amino_acids = np.sum(amino_acids, axis = 0)
	print(no_of_amino_acids)
	t_no_of_amino_acids = np.sum(no_of_amino_acids)
	print(t_no_of_amino_acids)
	no_seq = test_data_n[:, 21]
	t_no_of_no_seq = np.sum(no_seq)
	print(t_no_of_amino_acids, t_no_of_no_seq, t_no_of_amino_acids + t_no_of_no_seq)
	amino_acids_with_no_seq = test_data_n[:, 0:22]
	amino_acids_str_with_no_seq = test_data_n[:, 22:31]
	str_wise_sum = np.sum(amino_acids_str_with_no_seq, axis = 0)
	amino_acids_str_present = np.sum(str_wise_sum[:8])
	amino_acids_dum_present = np.sum(str_wise_sum[8])
	amino_acids_str_no = np.argmax(amino_acids_str_with_no_seq, 1)
	print("Str wise sum : ", str_wise_sum)
	print("Str present, padded data : ", amino_acids_str_present, amino_acids_dum_present, amino_acids_str_present + amino_acids_dum_present)
	amino_acids_no = np.argmax(test_data_n, 1)
	no_to_am_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
	am_acids_name = []
	for i in range(amino_acids_with_no_seq.shape[0]):
		amino_acid_no = amino_acids_no[i].tolist()
		am_acids_name.append(no_to_am_acid[amino_acid_no])
	amino_acids_total = 0
	no_seq_total = 0
	amino_acids_x = 0
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			no_seq_total += 1
		else:
			if(am_acid_name == 'X'):
			  amino_acids_x += 1
			amino_acids_total += 1
	print("amino_acids_total", amino_acids_total)
	print("no_seq_total", no_seq_total)
	print("amino_acid_x_total", amino_acids_x)

	seqs = {}
	seq_pro = {}
	for i in range(513):
		seqs[i] = ""
		seq_pro[i] = []

	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			continue
		else:
			seqs[i // 700] += am_acid_name
			seq_pro[i // 700].append(amino_acids_seq_profile[i].tolist())

	total_len_of_all_seqs = 0
	for i in range(513):
		total_len_of_all_seqs += len(seqs[i])

	print("Total len verfn results : ", total_len_of_all_seqs == amino_acids_total)

	seqs_in_vec = []
	masks = []
	ops = []
	seq_len = []
	zeros_list = [0] * len(seq_pro[0][0])
	word_to_glove = read_glove_vec_files()

	for i in range(513):
		seq = seqs[i]
		temp_seq = []
		temp_ops = []
		temp_msk = []

		for j in range(50):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove["dummy"])
			glove_and_seq_pro_list.extend(zeros_list)
			temp_seq.append(glove_and_seq_pro_list)
			# temp_seq.append(word_to_glove["dummy"])
			temp_ops.append(-1)
			temp_msk.append(0)

		for j in range(len(seq)):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove[seq[j]])
			glove_and_seq_pro_list.extend(seq_pro[i][j])
			temp_seq.append(glove_and_seq_pro_list)
			temp_ops.append(amino_acids_str_no[i*700 + j])
			temp_msk.append(1)

		for j in range(750 - len(seq)):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove["dummy"])
			glove_and_seq_pro_list.extend(zeros_list)
			temp_seq.append(glove_and_seq_pro_list)
			temp_ops.append(-1)
			temp_msk.append(0)

		seqs_in_vec.append(temp_seq)
		ops.append(temp_ops)
		masks.append(temp_msk)
		seq_len.append(len(seq) + 100)

	print("Reached line 284")
	ans = True
	count_masks_is_one = 0

	for j in range(513):
		for i in range(800):
			if(masks[j][i] == 1):
				count_masks_is_one += 1
				ans = ans and (ops[j][i] != -1)
			else:
				ans = ans and (ops[j][i] == -1)

	for j in range(513):
		ops_j = ops[j]
		for i in range(800):
			if(i<50 or i >= 50 + len(seqs[j])):
				ans = ans and (ops_j[i] == -1)
			else:
				ans = ans and (ops_j[i] != -1)

	ans = ans and ( count_masks_is_one == amino_acids_str_present)

	print("Verified the data inp, op and masks creation resuts : ", ans)

	batch_size = 128
	no_of_batches = 513 // batch_size
	mini_batch_data = {}
	print("Total number of batches : ", no_of_batches)
	for i in range(no_of_batches):
		temp = []
		if(i%50 == 0):
			print("Processing batch no : ", i)
		temp.append(seqs_in_vec[i * batch_size : (i + 1) * batch_size ])
		temp.append(ops[i * batch_size : (i + 1) * batch_size ])
		temp.append(masks[i * batch_size : (i + 1) * batch_size ])
		temp.append(seq_len[i * batch_size : (i + 1) * batch_size ])
		mini_batch_data[i] = temp
	print("Total len verfn results : ", total_len_of_all_seqs == amino_acids_total)
	save_obj(mini_batch_data, './data/batch_wise_test_data_' + str(batch_size) + '.pkl')

raw_data_train_to_mini_batches()
raw_data_test_to_mini_batches()


# word_to_glove =  read_glove_vec_files()
# print(word_to_glove.keys())
# print(len(word_to_glove.keys())) 23



 
