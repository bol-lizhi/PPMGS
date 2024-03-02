import numpy as np
import time
import copy

import warnings

LONG_MAXIMUM=1e5#To avoid memory error during data-exchange counting

TINY_POSITIVE=1e-8#To avoid divide-by-zero error

def count_receive(data):
	dtype=type(data)
	if dtype in (list,tuple):
		count=0
		for d in data:
			count+=count_receive(d)
		return count
	if dtype==np.ndarray:
		return np.prod(data.shape)
	return 1

class SiteSimulator():#It simulates the performance of a participant in computation and communication
	def __init__(self):
		self.count_send=0
		self.size_send=0
		self.count_receive=0
		self.size_receive=0
		self.runtime=0
		self.buff={}
		self.backup_size_receive=0
		self.backup_size_send=0

	def receive(self,index,data):
		global LONG_MAXIMUM
		size_receive_this_time=0
		for i in range(len(data)):
			#dtype=type(data[i])
			self.buff[index[i]]=copy.deepcopy(data[i])
			size_receive_this_time+=count_receive(data[i])#(np.prod(data[i].shape) if dtype==np.ndarray else 1)
		#size_receive_this_time/=1e5
		'''
		if self.size_receive < LONG_MAXIMUM - size_receive_this_time:
			try:
				self.size_receive+=size_receive_this_time
			except:
				print('warnings:',self.size_receive,size_receive_this_time)
		else:
			diff=LONG_MAXIMUM-self.size_receive
			self.backup_size_receive+=1
			self.size_receive=size_receive_this_time - diff
		'''
		total_size_receive=self.size_receive+size_receive_this_time
		backup_size_receive_thie_time=np.floor(total_size_receive/LONG_MAXIMUM)
		self.size_receive=total_size_receive - backup_size_receive_thie_time*LONG_MAXIMUM
		self.backup_size_receive+=backup_size_receive_thie_time
		self.count_receive+=1
		return size_receive_this_time

	def send(self,receivers,index,local_index):
		global LONG_MAXIMUM
		size_send_this_time=0
		data=[self.buff[i] for i in local_index]
		for r in receivers:
			size_send_this_time+=r.receive(index,data)
			self.count_send+=1
		#size_send_this_time/=1e5
		'''
		if self.size_send < LONG_MAXIMUM - size_send_this_time:
			self.size_send+=size_send_this_time
		else:
			diff=LONG_MAXIMUM-self.size_send
			self.backup_size_send+=1
			self.size_send=size_send_this_time - diff
		'''
		total_size_send=self.size_send+size_send_this_time
		backup_size_send_this_time=np.floor(total_size_send/LONG_MAXIMUM)
		self.size_send=total_size_send - backup_size_send_this_time*LONG_MAXIMUM
		self.backup_size_send+=backup_size_send_this_time
		return size_send_this_time

	def opt(self,func,loc_param_indices=None,params=None):
		begin=time.time()
		result=func(self,loc_param_indices,params)
		self.runtime+=time.time()-begin
		return result

class DistributedProtocol():
	def __init__(self):
		self.count_comm=0
		self.size_comm=0