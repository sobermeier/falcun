import time

import numpy as np
import pandas as pd
from .strategy import Strategy
from sklearn.cluster import KMeans


class KMeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
        super(KMeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.emb = emb

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X[idxs_unlabeled], y=self.Y[idxs_unlabeled])
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embedding)

        cluster_idxs = cluster_learner.predict(embedding)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embedding - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])
        end_time = time.time()

        sampling_df = pd.DataFrame([list(q_idxs)], index=["img_id"]).T
        self.save_stats(sampling_df)
        return idxs_unlabeled[q_idxs], end_time
