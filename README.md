##Privacy-preserving Mixture-distribution-based Graph Smoothing

	This is a local simulation of the semi-supervised distributed privacy-preserving data mining (DPPDM) protocol, PPMGS.

	All operations of each participant in the distributed system is simulated locally.	

	Each participant simulator maintains a python dict as its local memory.

	When inter-participant commnication happens, it simply assigns the data of the sender to the designated index in the dict of the receiver and records the size of the data.


##Requirements

	*sci-kit learn
	
	*numpy
	
	*scipy



##Run the demo
	
	python demo.py


##Available Parameters:
	
	--K=$integer : The number of mixture components; default: 2.
	
	--l=$integer : The number of labeled data per participant; default:7.
	
	--n=$integer : The number of total training data per participant; default:52.
	
	--m=$integer : The number of participants, default 7.
	
	--r=$float : The hyper-parameter denoting the number of the assumed unlabeled random variables, non-integer allowed; default: 1.
	
	--sigma=$float : The hyper-parameter denoting the bandwidth in similarity computation; default: 10.

	
	*All demos are run on the G50C dataset.
