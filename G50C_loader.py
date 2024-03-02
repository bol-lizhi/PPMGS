from os import path
import numpy as np

g50c_path=path.join('data','G50C.mat')
g50c0_path=path.join('data','G50C0.npy')
g50c1_path=path.join('data','G50C1.npy')

def loadG50C_seperated_balanced_labeled(num_labeled,num_train,num_sites):
	d0=np.load(g50c0_path)
	d1=np.load(g50c1_path)
	np.random.shuffle(d0)
	np.random.shuffle(d1)
	num_alllabeled=num_sites*num_labeled
	num_l0=int(num_alllabeled/2)
	num_l1=num_alllabeled-num_l0
	#print(d0.shape)
	#print(d1.shape)
	labeled=np.vstack((d0[:num_l0],d1[:num_l1]))
	labels=np.ones(num_alllabeled)
	labels[:num_l0]=0
	shuffle_index=np.random.permutation(np.arange(num_alllabeled))
	labeled=labeled[shuffle_index]
	labels=labels[shuffle_index]
	unlabeled=np.vstack((d0[num_l0:],d1[num_l1:]))
	unlabels=np.ones(550 - num_alllabeled)
	unlabels[:275-num_l0]=0
	shuffle_index=np.random.permutation(np.arange(550 - num_alllabeled))
	unlabeled=unlabeled[shuffle_index]
	unlabels=unlabels[shuffle_index]
	num_alltrain_unlabeled=num_train*num_sites - num_alllabeled
	test_data=unlabeled[num_alltrain_unlabeled:]
	test_labels=unlabels[num_alltrain_unlabeled:]
	train_data=[]
	train_labels=[]
	labeled_datasets=[]
	unlabeled_datasets=[]
	labelsets=[]
	num_unlabeled=num_train - num_labeled
	for i in range(num_sites):
		labeled_datasets.append(labeled[i*num_labeled:(i+1)*num_labeled])
		unlabeled_datasets.append(unlabeled[i*num_unlabeled:(i+1)*num_unlabeled])
		labelsets.append(labels[i*num_labeled:(i+1)*num_labeled])
		train_data.append(labeled_datasets[i])
		train_data.append(unlabeled_datasets[i])
		train_labels.append(labelsets[i])
		train_labels.append(unlabels[i*num_unlabeled:(i+1)*num_unlabeled])
	train_data=np.vstack(train_data)
	train_labels=np.hstack(train_labels)
	return labeled_datasets,unlabeled_datasets,labelsets,train_data,train_labels,test_data,test_labels

def loadG50C_fullsize_balanced_labeled(num_labeled,num_train,num_sites):
	d0=np.load(g50c0_path)
	d1=np.load(g50c1_path)
	np.random.shuffle(d0)
	np.random.shuffle(d1)
	num_alllabeled=num_sites*num_labeled
	num_l0=int(num_alllabeled/2)
	num_l1=num_alllabeled-num_l0
	#print(d0.shape)
	#print(d1.shape)
	labeled=np.vstack((d0[:num_l0],d1[:num_l1]))
	labels=np.ones((num_alllabeled,1))
	labels[num_l0:,0]=2
	shuffle_index=np.random.permutation(np.arange(num_alllabeled))
	labeled=labeled[shuffle_index]
	labels=labels[shuffle_index]
	unlabeled=np.vstack((d0[num_l0:],d1[num_l1:]))
	unlabels=np.ones((550 - num_alllabeled,1))
	unlabels[275-num_l0:,0]=2
	shuffle_index=np.random.permutation(np.arange(550 - num_alllabeled))
	unlabeled=unlabeled[shuffle_index]
	unlabels=unlabels[shuffle_index]
	num_alltrain_unlabeled=num_train*num_sites - num_alllabeled
	test_data=unlabeled[num_alltrain_unlabeled:]
	test_labels=unlabels[num_alltrain_unlabeled:]
	train_data=[]
	train_labels=[]
	datasets=[]
	labelsets=[]
	num_unlabeled=num_train - num_labeled
	for i in range(num_sites):
		datasets.append(np.vstack((labeled[i*num_labeled:(i+1)*num_labeled],unlabeled[i*num_unlabeled:(i+1)*num_unlabeled])))
		#unlabeled_datasets.append(unlabeled[i*num_unlabeled:(i+1)*num_unlabeled])
		labelsets.append(np.vstack((labels[i*num_labeled:(i+1)*num_labeled],np.zeros((num_unlabeled,1)))))
		train_labels.append(np.vstack((labels[i*num_labeled:(i+1)*num_labeled],unlabels[i*num_unlabeled:(i+1)*num_unlabeled])))
	train_data=np.vstack(datasets)
	train_labels=np.vstack(train_labels)
	return datasets,labelsets,train_data,train_labels,test_data,test_labels