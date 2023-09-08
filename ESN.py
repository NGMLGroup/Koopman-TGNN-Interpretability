import torch

from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg


class ESN(nn.Module):
    def __init__(self,
                 n_internal_units = 100,
                 spectral_radius = 0.9,
                 connectivity = 0.5,
                 input_scaling = 0.5,
                 input_shift = 0.0,
                 teacher_scaling = 0.5,
                 teacher_shift = 0.0,
                 feedback_scaling = 0.01,
                 noise_level = 0.01):
        super().__init__()

        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._spectral_radius = spectral_radius
        self._connectivity = connectivity

        self._input_scaling = input_scaling
        self._input_shift = input_shift
        self._teacher_scaling = teacher_scaling
        self._teacher_shift = teacher_shift
        self._feedback_scaling = feedback_scaling
        self._noise_level = noise_level
        self._dim_output = None

        # The weights will be set later, when data is provided
        self._input_weights = None
        self._feedback_weights = None

        # Regression method and embedding method.
        # Initialized to None for now. Will be set during 'fit'.
        self._regression_method = None
        self._embedding_method = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(n_internal_units, connectivity, spectral_radius)


        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))


    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    

    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):
        # The eigs function might not converge. Attempt until it does.
        convergence = False
        while (not convergence):
            # Generate sparse, uniformly distributed weights.
            internal_weights = sparse.rand(n_internal_units, n_internal_units, density=connectivity).todense()

            # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
            internal_weights[np.where(internal_weights > 0)] -= 0.5

            try:
                # Get the largest eigenvalue
                w,_ = slinalg.eigs(internal_weights, k=1, which='LM')

                convergence = True

            except:
                continue

        # Adjust the spectral radius.
        internal_weights /= np.abs(w)/spectral_radius

        return torch.from_numpy(np.array(internal_weights)