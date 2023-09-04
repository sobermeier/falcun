import time

import numpy as np
import pdb
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from .strategy import Strategy


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
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
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.emb = emb

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled]).numpy()
        if self.emb == "pca":
            pca = PCA(n_components=self.net_args["output_dim"])
            pca.fit(gradEmbedding)
            gradEmbedding = pca.transform(gradEmbedding)
        centers = init_centers(gradEmbedding, n)
        chosen = idxs_unlabeled[centers]
        end_time = time.time()

        sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
        self.save_stats(sampling_df)
        return chosen, end_time
