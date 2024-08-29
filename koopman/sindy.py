# Based on https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, TruncatedSVD
from einops import rearrange
from torch_sparse import SparseTensor


class SINDy:
    """Sparse Identification of Nonlinear Dynamics"""
    def __init__(self, Z, edge_index, k, add_self_dependency=True, degree=2, alpha=1.0, emb="PCA"):
        # Z in (nodes, time, features)

        self.k = k
        self.emb = emb
        self.emb_engine = None
        self.alpha = alpha
        self.degree = degree
        self.add_self_dependency = add_self_dependency

        if self.emb == "TruncatedSVD":
            self.emb_engine = TruncatedSVD(n_components=self.k)
        elif self.emb == "PCA":
            self.emb_engine = PCA(n_components=self.k)
        elif self.emb == None:
            self.emb_engine = None
            self.k = self.Z.shape[-1]

        # Compute principal components
        if self.emb == None:
            self.Zp = rearrange(Z, 't n f -> n t f')
        else:
            self.Zp = rearrange(Z, 't n f -> (t n) f')
            self.Zp = self.emb_engine.fit_transform(self.Zp)
            self.Zp = rearrange(self.Zp, '(t n) k -> n t k', t=Z.shape[0], n=Z.shape[1], k=k)
        
        self.num_nodes = self.Zp.shape[0]

        # Concatenate edge indexes
        # and compute adjacency matrix
        adj_matrix = self.from_edge_index_to_adj(edge_index)
        self.num_edges = torch.nonzero(adj_matrix).size(0)

        # Compute SINDy library
        self.Zp = torch.from_numpy(self.Zp)
        self.library = self.library_adj(self.Zp, adj_matrix)
    

    def fit(self):

        num_nodes = self.Zp.shape[0]
        x = rearrange(self.library[:,:-1,:], 'n t f -> t (n f)')
        y = rearrange(self.library[:num_nodes,1:,:], 'n t f -> t (n f)')

        ridge = Ridge(alpha=self.alpha, fit_intercept=False)

        ridge.fit(x, y)
        self.Xi = ridge.coef_

        return self.Xi
    

    def from_edge_index_to_adj(self, edge_index):
        edge_index = torch.cat(edge_index, dim=1)
        edge_index = torch.unique(edge_index.T, dim=0).T

        num_nodes = edge_index.max().item() + 1
        adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        adj_matrix = adj_matrix.to_dense()

        return adj_matrix
    

    def library_adj(self, X, adj):
        if self.degree==2:
            library = self.library_adj_2(X, adj)
        elif self.degree==3:
            library = self.library_adj_3(X, adj)
        else:
            raise ValueError(f'Unknown degree: {self.degree}')
        
        return library

            
    def library_adj_2(self, X, adj):
        N, T, F = X.shape

        L = self.num_edges

        if self.add_self_dependency:
            library = torch.empty((2*N+L,T,F))
            library[:N] = X
            library[N:2*N] = X**2
        else:
            library = torch.empty((L,T,F))
            N = 0

        index = 0
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj[i,j]:
                    library[2*N+index] = X[i]*X[j]
                    index += 1

        return library

            
    def library_adj_3(self, X, adj):
        N, T, F = X.shape

        L = self.num_edges

        if self.add_self_dependency:
            library = torch.empty((3*N+L+2*L,T,F))

            library[:N] = X
            library[N:2*N] = X**2
            library[2*N:3*N] = X**3
        else:
            library = torch.empty((3*L,T,F))
            N = 0

        index = 0
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj[i,j]:
                    library[3*N+index] = X[i]*X[j]
                    library[3*N+index+1] = X[i]**2*X[j]
                    library[3*N+index+2] = X[i]*X[j]**2
                    index += 3

        return library
    

    def compute_weights(self):

        Xi = self.fit()

        # Remove "F" dimension
        sum_Xi = Xi.reshape(-1, self.k, Xi.shape[1]).sum(axis=1)
        sum_Xi = sum_Xi.reshape(sum_Xi.shape[0], -1, self.k).sum(axis=2)
        
        # Compute mask weights
        # If present, remove the self-dependency
        if self.add_self_dependency:
            if self.degree == 2:
                sum_Xi = sum_Xi[:,2*self.num_nodes:]
            elif self.degree == 3:
                sum_Xi = sum_Xi[:,3*self.num_nodes:]
            else:
                raise ValueError(f'Unknown degree: {self.degree}')
        
        # For degree 3, sum over terms corresponding to the same edge
        if self.degree == 3:
            sum_Xi = sum_Xi.reshape(sum_Xi.shape[0], -1, 3).sum(axis=2)

        # Compute weights
        weights = np.abs(sum_Xi).sum(axis=0)
        
        return weights