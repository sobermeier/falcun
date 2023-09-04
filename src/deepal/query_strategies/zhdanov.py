import time

import numpy as np
import pandas as pd
from .strategy import Strategy
from sklearn.cluster import KMeans


class ZhdanovWeightedKMeans(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent", beta=50):
        super(ZhdanovWeightedKMeans, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.emb = emb
        self.beta = beta

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:, 1]
        sorted_margin, sm_idx = U.sort()

        if self.emb == "output":
            embedding = probs.numpy()
        else:
            embedding = self.get_diversity_embeddings(
                emb_type=self.emb,
                x=self.X[idxs_unlabeled],
                y=self.Y[idxs_unlabeled]
            )
        n_filter = n*self.beta
        idx_filtered = sm_idx[:n_filter]  # idx of top most uncertain samples
        embedding = embedding[idx_filtered]

        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embedding, sample_weight=1-sorted_margin[:n_filter])  # cluster top most uncertain samples
        cluster_idxs = cluster_learner.predict(embedding, sample_weight=1-sorted_margin[:n_filter])
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embedding - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])
        end_time = time.time()

        sampling_df = pd.DataFrame([list(sm_idx[q_idxs])], index=["img_id"]).T
        self.save_stats(sampling_df)
        return idxs_unlabeled[q_idxs], end_time
