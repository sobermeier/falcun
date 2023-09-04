import time

import numpy as np
import pdb
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances

from .strategy import Strategy


# kmeans ++ initialization
def init_centers(X, K, det=False):
    ind = np.random.choice(np.arange(len(X)))
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        if det:
            ind = D2.argmax()
        else:
            D2 = D2.ravel().astype(float)
            Ddist = D2 / sum(D2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class KMeansPP(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
        super(KMeansPP, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.emb = emb

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X[idxs_unlabeled], y=self.Y[idxs_unlabeled])

        centers = init_centers(embedding, n, det=True)
        chosen = idxs_unlabeled[centers]
        end_time = time.time()

        sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
        self.save_stats(sampling_df)
        return chosen, end_time


class KMeansPPSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
        super(KMeansPPSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.emb = emb

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X[idxs_unlabeled], y=self.Y[idxs_unlabeled])

        centers = init_centers(embedding, n, det=False)
        chosen = idxs_unlabeled[centers]
        end_time = time.time()

        sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
        self.save_stats(sampling_df)
        return chosen, end_time
