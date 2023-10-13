import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD


class KANN:
    """Koopman Analysis of Sequence Models"""
    def __init__(self, Z, k=None, emb="TruncatedSVD"):
        # Z in (batch, sequence, features)
        super(KANN, self).__init__()

        self.Z = Z
        self.Zp = None
        self.C = None

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

        # compute principal components
        if len(self.Z.shape) == 2:
            self.Zp = self.emb_engine.fit_transform(self.Z)

        else:
            bsz, sqsz, hsz = self.Z.shape

            zz = self.Z.reshape(-1, hsz)

            self.Zp = self.emb_engine.fit_transform(zz)
            self.Zp = self.Zp.reshape(bsz, sqsz, self.k)

    def compute_KOP(self, X=None, Y=None, index=None):

        if X is not None and Y is not None:
            # compute the KOP
            Xp = self.emb_engine.transform(X.reshape(-1, X.shape[-1])) # remove batch dim
            Yp = self.emb_engine.transform(Y.reshape(-1, Y.shape[-1])) # remove batch dim
            Xp = np.vstack([x[:idx] for x, idx in zip(Xp.reshape(-1, X.shape[1], self.k), index)])
            Yp = np.vstack([y[:idx] for y, idx in zip(Yp.reshape(-1, Y.shape[1], self.k), index)])
            self.C, rs, _, _ = np.linalg.lstsq(Xp, Yp, rcond=None)

        else:
            # split the data to before and after
            Xp, Yp = self.Zp[:, :-1, :], self.Zp[:, 1:, :]

            # compute the KOP
            self.C, rs, _, _ = np.linalg.lstsq(Xp.reshape(-1, self.k), Yp.reshape(-1, self.k), rcond=None)

        return self.C

    def recover_states(self, proj_states, r=2):
        # proj_states in [batch x seq x k]
        bsz, sqsz = proj_states.shape[0], proj_states.shape[1]

        flat_proj_states = proj_states.reshape(-1, r)
        states = self.emb_engine.inverse_transform(flat_proj_states)
        states = states.reshape(bsz, sqsz, -1)
        return states