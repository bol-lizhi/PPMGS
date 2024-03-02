from simulator import DistributedProtocol,TINY_POSITIVE
import copy
import numpy as np

'''Protocol Class'''
class SecureRingSummation(DistributedProtocol):
	def __init__(self,ceiling,length,fake_performance=False):
		DistributedProtocol.__init__(self)
		self.ceiling=ceiling
		self.length=length
		self.fake_performance=fake_performance
		#Exceeding of values will lead to wrong results. Turn on 'fake_performance' when this happens.
		#'fake_performance = True' means it still performs following the protocol but returns the result of direct summation.

	def secure_sum(self,part_sites,term_index,receivers,rec_index):
		num_sites=len(part_sites)
		for i in range(num_sites):
			s=part_sites[i]
			if self.fake_performance:
				if i==0:
					result=s.buff[term_index]
				else:
					result+=s.buff[term_index]
			s.opt(srs_initial_partial,[term_index],[rec_index])
		for i in range(num_sites):
			s=part_sites[i]
			for j in range(num_sites):
				if not j==i:
					s.opt(srs_divide_partial,[term_index],['srs_segment',rec_index,self.ceiling,self.length])
					s.send([part_sites[j]],['srs_segment'],['srs_segment'])
					part_sites[j].opt(srs_absorb_partial,['srs_segment'],[rec_index])
		for i in range(num_sites-1):
			part_sites[i].send([part_sites[i+1]],['srs_segment'],[rec_index])
			part_sites[i+1].opt(srs_absorb_partial,['srs_segment'],[rec_index])
		part_sites[num_sites-1].send(part_sites[:-1],[rec_index],[rec_index])
		if self.fake_performance:
			for s in part_sites:
				s.buff[rec_index]=result

'''Local Operation Functions'''
def srs_initial_partial(site,loc_param_indices,params):
	term=site.buff[loc_param_indices[0]]
	rec_index=params[0]
	site.buff[rec_index]=copy.deepcopy(term)

def srs_divide_partial(site,loc_param_indices,params):
	term=site.buff[loc_param_indices[0]]
	segment_index,rec_index,ceiling,length=params
	dtype=type(term)
	shape=term.shape if dtype==np.ndarray else None
	segment=np.random.random(shape)*ceiling-length
	if dtype==int:
		segment=int(segment)
	site.buff[rec_index]-=segment
	site.buff[segment_index]=segment

def srs_absorb_partial(site,loc_param_indices,params):
	segment=site.buff[loc_param_indices[0]]
	rec_index=params[0]
	site.buff[rec_index]+=segment

def srs_determine_segment_range(site,loc_param_indices,params):
	value=site.buff[loc_param_indices[0]]
	ceiling,length,ceiling_index,length_index=params
	order=np.floor(np.log10(abs(value)+1e-300))
	#if type(order)==np.ndarray:
	#	order[np.where(order==-np.inf)]=np.average(order[np.where(order!=-np.inf)])
	'''
	if np.sum(value)==0:
		order=np.random.random()
	else:
		order=np.floor(np.log10(abs(np.average(value))))
	'''
	site.buff[ceiling_index]=ceiling*(10**order)
	site.buff[length_index]=length*(10**order)

def srs_divide_partial2(site,loc_param_indices,params):
	term,ceiling,length=(site.buff[i] for i in loc_param_indices)
	segment_index,rec_index=params
	dtype=type(term)
	shape=term.shape if dtype==np.ndarray else None
	segment=np.random.random(shape)*ceiling-length
	if dtype==int:
		segment=int(segment)
	site.buff[rec_index]-=segment
	site.buff[segment_index]=segment