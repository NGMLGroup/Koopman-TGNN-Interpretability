import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge


class DMD:
    """Koopman Analysis of Sequence Models"""
    def __init__(self, Z, alpha=1.0, k=None, emb="TruncatedSVD"):
        # Z in (batch, sequence, features)

        self.Z = Z
        self.Zp = None
        self.C = None

        self.alpha = alpha
        self.k = k
        self.emb = emb
        self.emb_engine = None

        if self.k is None:
            p = self.Z.shape[0] * (self.Z.shape[1]-1)
            self.k = np.minimum(p, self.Z.shape[-1]) - 1
            # self.k = self.Z.shape[-1]-1

        if self.emb == "TruncatedSVD":
            self.emb_engine = TruncatedSVD(n_components=self.k)
        elif self.emb == "PCA":
            self.emb_engine = PCA(n_components=self.k)
        elif self.emb == None or self.emb == "Identity":
            self.emb_engine = None
            self.k = self.Z.shape[-1]

        # compute principal components
        if self.emb == None or self.emb == "Identity":
            self.Zp = Z
        elif len(self.Z.shape) == 2:
            self.Zp = self.emb_engine.fit_transform(self.Z)
        else:
            bsz, sqsz, hsz = self.Z.shape

            zz = self.Z.reshape(-1, hsz)

            self.Zp = self.emb_engine.fit_transform(zz)
            self.Zp = self.Zp.reshape(bsz, sqsz, self.k)

    def compute_KOP(self, X=None, Y=None, index=None):

        ridge = Ridge(alpha=self.alpha, fit_intercept=False)

        if X is not None and Y is not None:
            # compute the KOP
            Xp = self.emb_engine.transform(X.reshape(-1, X.shape[-1])) # remove batch dim
            Yp = self.emb_engine.transform(Y.reshape(-1, Y.shape[-1])) # remove batch dim
            Xp = np.vstack([x[:idx] for x, idx in zip(Xp.reshape(-1, X.shape[1], self.k), index)])
            Yp = np.vstack([y[:idx] for y, idx in zip(Yp.reshape(-1, Y.shape[1], self.k), index)])
            ridge.fit(Xp, Yp)
            self.C = ridge.coef_

        else:
            # split the data to before and after
            Xp, Yp = self.Zp[:, :-1, :], self.Zp[:, 1:, :]

            # compute the KOP
            ridge.fit(Xp.reshape(-1, self.k), Yp.reshape(-1, self.k))
            self.C = ridge.coef_

        return self.C

    def recover_states(self, proj_states, r=2):
        # proj_states in [batch x seq x k]
        bsz, sqsz = proj_states.shape[0], proj_states.shape[1]

        flat_proj_states = proj_states.reshape(-1, r)
        states = self.emb_engine.inverse_transform(flat_proj_states)
        states = states.reshape(bsz, sqsz, -1)
        return states
    
    def compute_weights(self, mode_idx=0):

        C = self.compute_KOP()

        # Compute eigenvalues and eigenvectors
        E, V = np.linalg.eig(C)
        idx = np.argsort(np.abs(E))[::-1]
        E = E[idx]
        V = V[:, idx]

        # Project states to first Koopman mode
        m = V[:, mode_idx].real
        weights = np.dot(self.Zp, m)

        return weights