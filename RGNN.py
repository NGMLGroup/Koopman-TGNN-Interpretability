import numpy as np
from scipy import sparse

class RGNN(object):
    """
    Build a randomized GNN layer and evaluate internal states
    
    Parameters:
        n_internal_units - processing units in the hidden layer
        spectral_radius - divide the hidden layer weight matrix by its largest eigenvalue and then multiply it by spectral_radius.
            If spectral_radius==None, the hidden weights matrix is rescaled according to max_norm.
        max_norm - divide the hidden weights matrix by its L2-norm and then multiply it max_norm.
            This is done only if spectral_radius==None.
        leak - amount of leakage in the hidden state update (optional)
        connectivity - percentage of nonzero connection weights (unused in circle connecitvity).
                       If connectivity is 1 the matrix is dense, otherwise is sparse
        input_scaling - scaling of the input connection weights
        noise_level - deviation of the Gaussian noise injected in the state update
        input_weights_mode - define how the input weights are generated. Options:
            binomial - the weight matrix is dense and its elements are drawn from {-1, 1}
            uniform - the weight matrix is dense and its elements are drawn from [-1, 1]
            sparse_uniform - the weight matrix is sparse (with sparsity 0.1) and its elements are drawn from [-1, 1]
            very_sparse - each hidden units is randomly connected to an input feature (connections weights are all 1)
        circle - generate determinisitc hidden layer weights with circle topology
    """
    
    def __init__(self, 
                 n_internal_units=None, 
                 spectral_radius=None,
                 max_norm=None,
                 leak=None,
                 connectivity=None, 
                 input_scaling=None, 
                 noise_level=None, 
                 input_weights_mode=None,
                 circle=False):
        
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak
        self.input_weights_mode = input_weights_mode

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                    n_internal_units,
                    spectral_radius,
                    max_norm)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius,
                max_norm)


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius, max_norm):
        
        # Build circular connections
        internal_weights = sparse.lil_matrix((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1
                
        # Convert to csr format for efficiency
        internal_weights = sparse.csr_matrix(internal_weights)
        
        if spectral_radius is not None:
            # Adjust the spectral radius
            E, _ = sparse.linalg.eigsh(internal_weights, k=1, which='LM')
            e_max = np.max(E)
            internal_weights /= np.abs(e_max)/spectral_radius
        elif max_norm is not None:
            # Adjust the matrix norm
            current_norm = sparse.linalg.norm(internal_weights, ord=2)
            internal_weights /= current_norm
            internal_weights *= max_norm
        else:
            raise ValueError('provide spectral_radius or max_norm!') 
                
        return internal_weights
    
    
    def _initialize_internal_weights(self, n_internal_units, connectivity,
                                     spectral_radius, max_norm):

        # Generate sparse, uniformly distributed weights
        if connectivity == 1:
            internal_weights = np.random.rand(n_internal_units, n_internal_units)
        else:
            internal_weights = sparse.rand(n_internal_units,
                                           n_internal_units,
                                           density=connectivity,
                                           format='csr')

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights = internal_weights - (internal_weights > 0)*.5
        
        if spectral_radius is not None:
            # Adjust the spectral radius
            E, _ = sparse.linalg.eigsh(internal_weights, k=1, which='LM')
            e_max = np.max(E)
            internal_weights /= np.abs(e_max)/spectral_radius
        elif max_norm is not None:
            # Adjust the matrix norm
            if connectivity == 1:
                current_norm = np.linalg.norm(internal_weights, ord=2)
            else:
                current_norm = sparse.linalg.norm(internal_weights, ord=2)
            internal_weights /= current_norm
            internal_weights *= max_norm
        else:
            raise ValueError('provide spectral_radius or max_norm!')      

        return internal_weights
    
    
    def _initialize_input_weights(self, F, mode, normalize=False):
        
        if mode == 'binomial':
            input_weights = (2.0*np.random.binomial(1, 0.5 , [F, self._n_internal_units]) - 1.0)
            
        elif mode == 'uniform':
            input_weights = np.random.uniform(low=-1, high=1, size=(F, self._n_internal_units))
            
        elif mode == 'sparse_uniform':
            input_weights = sparse.rand(F, self._n_internal_units, density=0.1)*2.0
            input_weights = input_weights - (input_weights > 0)*1.0
            
        elif mode == 'very_sparse':
            I = np.eye(F)
            idx = np.random.randint(low=0, high=F, size=self._n_internal_units)
            input_weights = I[:,idx]
            input_weights *= np.random.uniform(low=-1.0, high=1.0, size=input_weights.shape)
            input_weights = sparse.csr_matrix(input_weights)
            
        else:
            raise NotImplementedError('wrong value for input_weights_mode!') 
            
        if normalize:
            input_weights = input_weights/np.linalg.norm(input_weights)
            
        return input_weights
        

    def _compute_state_matrix(self, A, X, conv_thresh, max_iter):
                
        # Init
        N, F = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)
        WiU = (self._input_weights.T.dot(X.T)).T
        
        converged = False
        current_iter = 0
        while not converged:
            
            # Calculate state
            state_before_tanh = WiU + self._internal_weights.T.dot((A.dot(previous_state)).T).T # use transposes here to call sp dot
            
            # Add noise
            state_before_tanh += np.random.rand(N, self._n_internal_units)*self._noise_level
                        
            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                new_state = np.tanh(state_before_tanh)
            else:
                new_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh)
                                               
            # Check convergence
            if np.linalg.norm(new_state - previous_state) < conv_thresh or current_iter >= max_iter:
                if current_iter >= max_iter:
                    print('not converged')
                    
                return new_state
            else:
                previous_state = new_state
                current_iter += 1
                

    def get_states(
            self, 
            A, 
            X, 
            conv_thresh=1e-5, 
            max_iter=50):
        """
        Compute the new vertex features from the hidden states
        
        A - adjacency matrix NxN
        X - matrix of vertex features 
        conv_tresh - the recursion has converged if the norm of the new hidden state changes less than conv_tresh
        max_iter - does not wait for convergence if the maximum number of iterations max_iter is reached
        """
        
        _, F = X.shape
        if self._input_weights is None:
            self._input_weights = self._initialize_input_weights(F, mode=self.input_weights_mode)
            self._input_weights *= self._input_scaling

        # compute hidden states
        states = self._compute_state_matrix(A, X, conv_thresh=conv_thresh, max_iter=max_iter)
    
        return states