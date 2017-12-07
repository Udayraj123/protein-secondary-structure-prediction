import numpy as np
import pandas as pd
import sklearn.metrics as skm

print("Loading the data : ")
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
test_data = np.load('./data/cb513+profile_split1.npy') 
flag = 0
print("Printing data shape : ")

print(train_data.shape)
if(train_data.shape != (5534, 39900)):
	flag = 1
	print ("Input Train data shape is not matching with (5534, 39900)")

print(test_data.shape)
if(test_data.shape != (514, 39900)):
	flag = 1
	print ("Input Test data shape is not matching with (514, 39900)")

train_data = np.reshape(train_data, [-1, 700, 57])
test_data = np.reshape(test_data, [-1, 700, 57])

print("Printing data shape : ")
print(train_data.shape)
if(train_data.shape != (5534, 700, 57)):
	flag = 1
	print ("Resized Train data shape is not matching with (5534, 700, 57)")

print(test_data.shape)
if(test_data.shape != (514, 700, 57)):
	flag = 1
	print ("Input Test data shape is not matching with (514, 700, 57)")


train_data_input = train_data[:, :, np.r_[0:21, 36:57]]
train_data_otput = train_data[:, :, 22:30]
test_data_input = test_data[:, :, np.r_[0:21, 36:57]]
test_data_otput = test_data[:, :, 22:30]

print("Printing partitioned data")
print(train_data_input.shape)
if(train_data_input.shape != (5534, 700, 42)):
	flag = 1
	print ("Resized train_data_input shape is not matching with (5534, 700, 42)")

print(train_data_otput.shape)
if(train_data_otput.shape != (5534, 700, 8)):
	flag = 1
	print ("Resized train_data_input shape is not matching with (5534, 700, 8)")

print(test_data_input.shape)
if(test_data_input.shape != (514, 700, 42)):
	flag = 1
	print ("Resized train_data_input shape is not matching with (514, 700, 42)")

print(test_data_otput.shape)
if(test_data_otput.shape != (514, 700, 8)):
	flag = 1
	print ("Resized train_data_input shape is not matching with (514, 700, 8)")

train_data_otput = np.reshape(train_data_otput, [-1, 8])
test_data_otput = np.reshape(test_data_otput, [-1, 8])


# print(pd.DataFrame(test_data_otput).mean(axis = 1))
# print(pd.DataFrame(train_data_otput).mean(axis = 1))

if(flag==0):
	print("All checks passed successfully")
else:
	print("Input data is incorrect!")