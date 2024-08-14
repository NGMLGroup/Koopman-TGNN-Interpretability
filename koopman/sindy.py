# Based on https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

import torch
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, TruncatedSVD
from einops import rearrange
from torch_sparse import SparseTensor


class SINDy:
    """Sparse Identification of Nonlinear Dynamics"""
    def __init__(self, Z, edge_index, k, alpha=1.0, emb="PCA"):
        # Z in (nodes, time, features)

        self.k = k
        self.emb = emb
        self.emb_engine = None
        self.alpha = alpha

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

        # Concatenate edge indexes
        # and compute adjacency matrix
        adj_matrix = self.from_edge_index_to_adj(edge_index)

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
        N, T, F = X.shape

        L = torch.nonzero(adj).size(0)
        library = torch.empty((2*N+L,T,F))
        index = 0

        library[:N] = X
        library[N:2*N] = X**2
        
        for i in range(N):
            for j in range(N):
                if adj[i,j]:
                    library[2*N+index] = X[i]*X[j]
                    index += 1

        return library