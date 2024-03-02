import numpy as np
from scipy.io import loadmat

from simulator import SiteSimulator,LONG_MAXIMUM
from secure_ring_summation import SecureRingSummation
from ppmgs import PPEM,PPMGS

from sklearn.utils import check_random_state
from sklearn.metrics import f1_score

import time

import sys,getopt

from G50C_loader import loadG50C_seperated_balanced_labeled

def test_G50C_PPMG(num_components=2,num_labeled=7,num_train=52,num_sites=7,num_unlabeled=1,sigma=10):
	sites=[SiteSimulator() for i in range(num_sites)]
	sins=SecureRingSummation(2,5)
	#sins=sim.DummySecureSummation()
	ppemins=PPEM(num_components,sins)
	ppmgins=PPMGS(num_components,2,sins,ppemins)
	mls,mus,ls,trd,trl,ted,tel=loadG50C_seperated_balanced_labeled(num_labeled,num_train,num_sites)
	ppmgins.load_data(sites,mus,mls,ls)
	ppmgins.fit(sites,num_unlabeled,sigma)
	gins=sites[0].buff['gins']
	label_func=sites[0].buff['label_func']

	train_predprob=gins.predict_proba(trd)
	train_predlab=np.dot(train_predprob,label_func).argmax(axis=1)
	train_error=np.count_nonzero(train_predlab-trl)
	train_f1micro=f1_score(trl.flatten(),train_predlab.flatten(),average='micro')
	train_f1macro=f1_score(trl.flatten(),train_predlab.flatten(),average='macro')

	test_predprob=gins.predict_proba(ted)
	test_predlab=np.dot(test_predprob,label_func).argmax(axis=1)
	test_error=np.count_nonzero(test_predlab-tel)
	test_f1micro=f1_score(tel.flatten(),test_predlab.flatten(),average='micro')
	test_f1macro=f1_score(tel.flatten(),test_predlab.flatten(),average='macro')
	
	master_time=sites[0].runtime
	master_count_comm=sites[0].count_send
	master_size_comm=sites[0].size_send
	master_backup_size_comm=sites[0].backup_size_send

	member_time=sum((sites[i].runtime for i in range(1,num_sites)))
	member_count_comm=sum((sites[i].count_send for i in range(1,num_sites)))
	member_size_comm=sum((sites[i].size_send for i in range(1,num_sites)))
	member_backup_size_comm=sum((sites[i].backup_size_send for i in range(1,num_sites)))

	return master_time,master_count_comm,master_backup_size_comm,master_size_comm,member_time,member_count_comm,member_backup_size_comm,member_size_comm,train_f1micro,train_f1macro,train_error,test_f1micro,test_f1macro,test_error


if __name__=='__main__':
	opts,args = getopt.getopt(sys.argv[1:],"",['K=','l=','n=','m=','r=','sigma='])
	K=2
	l=7
	n=52
	m=7
	r=1
	sigma=10
	for name,value in opts:
		if name =='--K':
			K=int(value)
		elif name=='--l':
			l=int(value)
		elif name=='--n':
			n=int(value)
		elif name=='--m':
			m=int(value)
		elif name=='--r':
			r=float(value)
		elif name=='--sigma':
			sigma=float(value)
	train_size=n*m
	test_size=550-train_size
	print('Performing PPMGS on G50C')
	print('Settings:')
	print('\t',K,'mixture components')
	print('\t',l,'labeled data per participant')
	print('\t',n,'total training data per participant')
	print('\t',m,'participants')
	print('\thyper-parameter r=',r)
	print('\thyper-parameter sigma=',sigma)
	result=test_G50C_PPMG(K,l,n,m,r,sigma)
	print('Result:')
	print('\tTime Cost Per-participant: ',(result[0]+result[4])/m)
	print('\tData Exchanges Per-participant: ',((result[2]+result[6])*LONG_MAXIMUM+result[3]+result[7])/m)
	print('\tTrain f1-macro: ',result[9])
	print('\tTrain ACC: ',1-result[10]/train_size)
	print('\tTest f1-macro: ',result[12])
	print('\tTest ACC: ',1-result[13]/test_size)