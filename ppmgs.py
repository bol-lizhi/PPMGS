import numpy as np
import time

from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky as compute_precision_cholesky

from sklearn.utils.validation import check_array
from sklearn.utils.extmath import row_norms
from sklearn.metrics import euclidean_distances

from simulator import DistributedProtocol,TINY_POSITIVE


'''
Sub-Protocol: Privacy-preserving Spherical EM
'''
'''Protocol Class'''
class PPEM(DistributedProtocol):
	def __init__(self,num_components,SecureSum,tol=0.001,max_iter=100):
		DistributedProtocol.__init__(self)
		self.num_components=num_components
		self.SecureSumProtocol=SecureSum
		self.tol=tol
		self.max_iter=max_iter

	def load_data(self,part_sites,datasets,data_index):
		for i in range(len(part_sites)):
			part_sites[i].buff[data_index]=datasets[i]

	def preprocess(self,part_sites,data_index,masters,gmm_index,datasets=None,initial_params=None):
		if datasets:
			self.load_data(part_sites,datasets,data_index)
		for s in part_sites:
			s.buff['gmm_tol']=self.tol
			s.buff['gmm_max_iter']=self.max_iter
			s.buff['num_data_partial']=s.buff[data_index].shape[0]
			s.buff[gmm_index]=GMM_spherical_partial(self.num_components,covariance_type='spherical',tol=self.tol,max_iter=self.max_iter)
			s.opt(gmm_preprocess_data_partial,[data_index,gmm_index],[data_index+'2'])
			if initial_params:
				s.buff['weights']=np.array(initial_params[0])
				s.buff['means']=np.array(initial_params[1])
				s.buff['covs']=np.array(initial_params[2])
		for s in masters:
			s.buff['lower_bound']=-np.infty
		_=self.SecureSumProtocol.secure_sum(part_sites,'num_data_partial',masters,'num_data')

	def fit(self,part_sites,data_index,receiver,gmm_index,cov_scale,random_master=True,datasets=None,initial_params=None,kmeans=None):
		masters=part_sites[:1] if random_master else part_sites
		self.preprocess(part_sites,data_index,masters,gmm_index,datasets,initial_params)
		initial_iter_number=1 if initial_params else 0
		num_sites=len(part_sites)
		for i in range(initial_iter_number,self.max_iter+initial_iter_number):
			for s in part_sites:
				s.opt(gmm_EM_partial,[data_index,data_index+'2',gmm_index,'weights','means','covs'],[i,num_sites,'partial_param_',kmeans])
			_=self.SecureSumProtocol.secure_sum(part_sites,'partial_param_'+'A',masters,'global_param_A')
			_=self.SecureSumProtocol.secure_sum(part_sites,'partial_param_'+'B',masters,'global_param_B')
			_=self.SecureSumProtocol.secure_sum(part_sites,'partial_param_'+'E',masters,'global_param_E')
			for s in masters:
				stop=s.opt(gmm_master_partial,['global_param_A','global_param_B','global_param_E','num_data',gmm_index,'lower_bound','gmm_tol'],['weights','means','covs'])
			if random_master:
				_=masters[0].send(part_sites[1:],['weights','means','covs'],['weights','means','covs'])
			if stop:
				#print('achieved')
				break
		if not cov_scale==1:
			for s in part_sites:
				s.opt(gmm_scale_covs_partial,['weights','means','covs',gmm_index],[cov_scale])
		#print(i)
		return i

'''Local Operation Functions'''
def gmm_preprocess_data_partial(site,loc_param_indices,params):
	X,gins=site.buff[loc_param_indices[0]],site.buff[loc_param_indices[1]],
	X2index=params[0]
	site.buff[loc_param_indices[0]],site.buff[X2index]=gins.preprocess_data(X)
	
def gmm_EM_partial(site,loc_param_indices,params):
	num_iter,num_sites,partial_value_index_initial_,kmeans=params
	if num_iter==0:
		X,X2,gins=(site.buff[loc_param_indices[i]] for i in range(3))
		site.buff[partial_value_index_initial_+'A'],site.buff[partial_value_index_initial_+'B'],site.buff[partial_value_index_initial_+'E']=gins.initial_round_EM(X,X2,num_sites,kmeans)
	else:
		X,X2,gins,weights,means,covs=(site.buff[i] for i in loc_param_indices)
		site.buff[partial_value_index_initial_+'A'],site.buff[partial_value_index_initial_+'B'],site.buff[partial_value_index_initial_+'E']=gins.one_round_EM(X,X2,num_sites,weights,means,covs)

def gmm_master_partial(site,loc_param_indices,params):
	A,B,E,num_data,gins,lower_bound,tol=(site.buff[i] for i in loc_param_indices)
	weights_index,means_index,covs_index=params
	site.buff[weights_index]=B/num_data
	means=A/(B.reshape((B.shape[0],1)))
	site.buff[means_index]=means
	X_means=(means*A).sum(axis=1)
	means2=(means**2).sum(axis=1)
	site.buff[covs_index]=((E-2*X_means)/B+means2+1e-6)/means.shape[1]
	inv_covs=1/site.buff[covs_index]
	D1=inv_covs.sum()*E.sum()/2
	D2=np.dot(A.sum(axis=0).T,np.dot(inv_covs,means))
	D3=(means2*inv_covs).sum()*num_data/2
	D=D2-D1-D3+num_data*(np.log(site.buff[weights_index]).sum()-0.5*means.shape[1]*np.log(2*np.pi*site.buff[covs_index]).sum())
	change=D-lower_bound
	site.buff[loc_param_indices[-2]]=D
	#print(D,change)
	if abs(change)<tol:
		return True
	return False

def gmm_scale_covs_partial(site,loc_param_indices,params):
	weights,means,covs,gins=(site.buff[i] for i in loc_param_indices)
	cov_scale=params[0]
	covs*=cov_scale
	gins.params_update(weights,means,covs)

'''Spherical GMM class'''
class GMM_spherical_partial(GMM):
	def one_round_EM(self,X,X2,num_sites,weights,means,covs):
		self.params_update(weights,means,covs)
		log_prob_norm, log_resp = self._e_step(X)
		resp=np.exp(log_resp)
		B=resp.sum(axis=0)+10*np.finfo(resp.dtype).eps/num_sites
		A=np.dot(resp.T,X)
		E=np.dot(X2,resp)
		return A,B,E

	def initial_round_EM(self,X,X2,num_sites,kmeans=None):
		n_samples=X.shape[0]
		if kmeans:
			resp=np.zeros((n_samples,self.n_components))
			label=kmeans.predict(X)
			resp[np.arange(n_samples),label]=1
			#print('working')
		else:
			resp=np.random.rand(n_samples,self.n_components)
			resp /= resp.sum(axis=1)[:, np.newaxis]
		B=resp.sum(axis=0)+10*np.finfo(resp.dtype).eps/num_sites
		A=np.dot(resp.T,X)
		E=np.dot(X2,resp)
		return A,B,E
	
	def preprocess_data(self,X):
		X=check_array(X,dtype=[np.float64, np.float32],ensure_min_samples=1)
		X2=row_norms(X,squared=True)
		return X,X2

	def params_update(self,weights,means,covs):
		'''
		self.weights_=weights
		self.means_=means
		self.covariances_=covs
		self.precisions_cholesky_ = compute_precision_cholesky(covs,'spherical')
		'''
		self._set_parameters((weights,means,covs,compute_precision_cholesky(covs,'spherical')))



'''
Privacy-preserving MGS Optimization
'''
'''Protocol Class'''
class PPMGS(DistributedProtocol):
	def __init__(self,num_components,num_classes,SecureSum,SecureEM,cov_type='spherical'):
		DistributedProtocol.__init__(self)
		self.num_components=num_components
		self.cov_type=cov_type
		self.SecureSumProtocol=SecureSum
		self.SecureEMProtocol=SecureEM
		self.num_classes=num_classes

	def load_data(self,part_sites,unlabeled_datasets,labeled_datasets,labels):
		for i in range(len(unlabeled_datasets)):
			part_sites[i].buff['unlabeled_data']=unlabeled_datasets[i]
			part_sites[i].buff['labeled_data']=labeled_datasets[i]
			part_sites[i].buff['labels']=labels[i]

	def fit(self,part_sites,num_unlabeled,rbf_sigma=0.5,labeled_weight=1,cov_scale=1):
		self.SecureEMProtocol.fit(part_sites,'unlabeled_data',part_sites,'gins',cov_scale)
		self.fit_with_gmm(part_sites,num_unlabeled,rbf_sigma=rbf_sigma,labeled_weight=labeled_weight,cov_scale=cov_scale)

	def fit_with_gmm(self,part_sites,num_unlabeled,rbf_sigma=0.5,labeled_weight=1,cov_scale=1):
		for s in part_sites:
			s.opt(graph_build_partial,['labeled_data','labels','gins'],[num_unlabeled,rbf_sigma,labeled_weight,self.num_classes,'WUU','WULY',self.cov_type])
		WULY=self.SecureSumProtocol.secure_sum(part_sites,'WULY',part_sites,'WULY')
		label_func=[]
		for s in part_sites:
			label_func.append(s.opt(solve_obj_partial,['WUU','WULY'],[self.num_components,'label_func']))
		return label_func

'''Local Operation Functions'''
def buildGraph_fullCov(MatL,MatU,MatL_weight,MatU_weight,MatL_cov,MatU_cov,rbf_sigma=None,knn=0):
	num_labeled=MatL.shape[0]
	num_unlabeled=MatU.shape[0]
	data_dim=MatL.shape[1]
	affinity_UL = np.zeros((num_unlabeled, num_labeled), np.float32)
	affinity_UU = np.zeros((num_unlabeled, num_unlabeled), np.float32)
	MatA=np.eye(data_dim)*rbf_sigma
	kcomp=num_labeled+num_unlabeled-knn
	for i in range(num_unlabeled):
		for j in range(num_labeled):
			diff=MatU[i]-MatL[j]
			new_sigma=np.matrix(MatU_cov[i]+MatL_cov[j]+MatA)
			new_sigma_I=np.array(new_sigma.I)
			affinity=np.exp(-0.5*(np.dot(np.dot(diff,new_sigma_I),diff)))
			affinity*=np.sqrt(1/np.linalg.det(new_sigma/rbf_sigma))
			affinity*=MatU_weight[i]*MatL_weight[j]
			affinity_UL[i][j]=affinity
		affinity_UU[i][i]=0.0
		for j in range(i+1,num_unlabeled):
			diff=MatU[i]-MatU[j]
			new_sigma=np.matrix(MatU_cov[i]+MatU_cov[j]+MatA)
			new_sigma_I=np.array(new_sigma.I)
			affinity=np.exp(-0.5*(np.dot(np.dot(diff,new_sigma_I),diff)))
			affinity*=np.sqrt(1/np.linalg.det(new_sigma/rbf_sigma))
			affinity*=MatU_weight[i]*MatU_weight[j]
			affinity_UU[i][j]=affinity_UU[j][i]=affinity
		if knn>0:
			affinity_i=np.hstack((affinity_UL[i],affinity_UU[i]))
			inds=np.argpartition(affinity_i,kcomp-1)[:kcomp]
			affinity_i=np.ones(num_labeled+num_unlabeled)
			affinity_i[inds]=0.0
			affinity_UL[i]=affinity_i[:num_labeled]
			affinity_UU[i]=affinity_i[num_labeled:]
	return affinity_UL,affinity_UU

def buildGraph_diagCov(MatL,MatU,MatL_weight,MatU_weight,MatL_cov,MatU_cov,rbf_sigma=None,knn=0):
	num_labeled=MatL.shape[0]
	num_unlabeled=MatU.shape[0]
	data_dim=MatL.shape[1]
	affinity_UL = np.zeros((num_unlabeled, num_labeled), np.float32)
	affinity_UU = np.zeros((num_unlabeled, num_unlabeled), np.float32)
	kcomp=num_labeled+num_unlabeled-knn
	for i in range(num_unlabeled):
		for j in range(num_labeled):
			diff=MatU[i]-MatL[j]
			new_sigma=MatU_cov[i]+MatL_cov[j]+rbf_sigma*np.ones(data_dim)
			affinity=np.exp(-0.5*(np.dot(diff,diff/new_sigma)))
			#affinity*=np.sqrt((rbf_sigma**data_dim)/np.prod(new_sigma))
			affinity*=np.sqrt(np.prod(rbf_sigma/new_sigma))
			affinity*=MatU_weight[i]*MatL_weight[j]
			affinity_UL[i][j]=affinity
		affinity_UU[i][i]=0.0
		for j in range(i+1,num_unlabeled):
			diff=MatU[i]-MatU[j]
			new_sigma=MatU_cov[i]+MatU_cov[j]+rbf_sigma*np.ones(data_dim)
			affinity=np.exp(-0.5*(np.dot(diff,diff/new_sigma)))
			#affinity*=np.sqrt((rbf_sigma**data_dim)/np.prod(new_sigma))
			affinity*=np.sqrt(np.prod(rbf_sigma/new_sigma))
			affinity*=MatU_weight[i]*MatU_weight[j]
			affinity_UU[i][j]=affinity_UU[j][i]=affinity
		if knn>0:
			affinity_i=np.hstack((affinity_UL[i],affinity_UU[i]))
			inds=np.argpartition(affinity_i,kcomp-1)[:kcomp]
			affinity_i=np.ones(num_labeled+num_unlabeled)
			affinity_i[inds]=0.0
			affinity_UL[i]=affinity_i[:num_labeled]
			affinity_UU[i]=affinity_i[num_labeled:]
	return affinity_UL,affinity_UU

def buildGraph_sphericalCov(MatL,MatU,MatL_weight,MatU_weight,MatL_cov,MatU_cov,rbf_sigma=None,knn=0):
	num_labeled=MatL.shape[0]
	num_unlabeled=MatU.shape[0]
	data_dim=MatL.shape[1]
	normUU=row_norms(MatU,squared=True)[:,np.newaxis]
	affinity_UL = -0.5*euclidean_distances(MatU,MatL,squared=True,X_norm_squared=normUU)
	affinity_UU = -0.5*euclidean_distances(MatU,squared=True,X_norm_squared=normUU)
	#return affinity_UL,affinity_UU
	kcomp=num_labeled+num_unlabeled-knn
	sigmaORweight=MatU_cov.reshape((num_unlabeled,1))+MatL_cov+rbf_sigma
	#print(sigmaORweight[0][:10])
	#print(MatL_cov.shape)
	#print(rbf_sigma)
	#return sigmaORweight,None
	affinity_UL/=sigmaORweight
	affinity_UL=np.exp(affinity_UL)
	sigmaORweight=np.sqrt((rbf_sigma/sigmaORweight)**data_dim)
	sigmaORweight*=MatU_weight.reshape((num_unlabeled,1))*MatL_weight
	affinity_UL*=sigmaORweight
	#return affinity_UL,affinity_UU
	for i in range(num_unlabeled):
		uslice=affinity_UU[i,i+1:]
		sigmaORweight=MatU_cov[i]+MatU_cov[i+1:]+rbf_sigma
		uslice/=sigmaORweight
		uslice=np.exp(uslice)
		sigmaORweight=np.sqrt((rbf_sigma/sigmaORweight)**data_dim)
		sigmaORweight*=MatU_weight[i]*MatU_weight[i+1:]
		uslice*=sigmaORweight
		affinity_UU[i+1:,i]=uslice
		affinity_UU[i,i+1:]=uslice
		affinity_UU[i,i]=0.0
		if knn>0:
			affinity_i=np.hstack((affinity_UL[i],affinity_UU[i]))
			inds=np.argpartition(affinity_i,kcomp-1)[:kcomp]
			affinity_i=np.ones(num_labeled+num_unlabeled)
			affinity_i[inds]=0.0
			affinity_UL[i]=affinity_i[:num_labeled]
			affinity_UU[i]=affinity_i[num_labeled:]
	#print(affinity_UL)
	return affinity_UL,affinity_UU

def buildGraph(MatL,MatU,MatU_weight,MatU_cov,rbf_sigma,labeled_weight,knn=0,cov_type='full'):
	num_label=MatL.shape[0]
	data_dim=MatL.shape[1]
	#print(MatU_weight,MatU_cov,rbf_sigma)
	MatL_weight=labeled_weight*np.ones(num_label, np.float32)
	if cov_type=='full':
		MatL_cov=np.zeros([num_label,data_dim,data_dim], np.float32)
		return buildGraph_fullCov(MatL,MatU,MatL_weight,MatU_weight,MatL_cov,MatU_cov,rbf_sigma,0)
	elif cov_type=='diag':
		MatL_cov=np.zeros([num_label,data_dim], np.float32)
		return buildGraph_diagCov(MatL,MatU,MatL_weight,MatU_weight,MatL_cov,MatU_cov,rbf_sigma,0)
	else:
		MatL_cov=np.zeros(num_label)
		return buildGraph_sphericalCov(MatL,MatU,MatL_weight,MatU_weight,MatL_cov,MatU_cov,rbf_sigma,0)
#end

#partial funcs
def graph_build_partial(site,loc_param_indices,params):
	mat_labeled,labels,gins=(site.buff[i] for i in loc_param_indices)
	num_unlabeled,rbf_sigma,labeled_weight,num_classes,WUUindex,WULYindex,cov_type=params
	MatU_cov=gins.covariances_
	MatU_weight=gins.weights_*num_unlabeled
	Mat_Component=gins.means_
	WUL,WUU=buildGraph(mat_labeled,Mat_Component,MatU_weight,MatU_cov,rbf_sigma,labeled_weight,cov_type=cov_type)
	num_labeled=labels.shape[0]
	clamp_labels=np.zeros([num_labeled,num_classes],np.float32)
	site.buff[WUUindex]=WUU
	for i in range(num_labeled):
		clamp_labels[i][int(labels[i])] = 1.0
	site.buff[WULYindex]=np.dot(WUL,clamp_labels)
	'''
	test
	'''
	#site.buff['WUL']=WUL
	'''
	site.buff['uweight']=MatU_weight
	site.buff['ucov']=MatU_cov
	site.buff['umeans']=Mat_Component
	'''

def solve_obj_partial(site,loc_param_indices,params):
	WUU,WULY=site.buff[loc_param_indices[0]],site.buff[loc_param_indices[1]]
	num_components,label_func_index=params
	temp=np.array(WUU)
	rowsum=np.hstack((temp,WULY)).sum(axis=1)
	#rowsum=np.diag(1/rowsum)
	
	temp*=-1
	#print(temp.shape,num_componen)
	temp[np.diag_indices(num_components)]=rowsum
	#return rowsum
	site.buff[label_func_index]=np.dot(np.linalg.inv(temp),WULY)
	return site.buff[label_func_index]