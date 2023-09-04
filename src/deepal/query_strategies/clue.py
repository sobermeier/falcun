import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import euclidean_distances

from .strategy import Strategy
from sklearn.cluster import KMeans


class CLUE(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
        super(CLUE, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.emb = emb

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs += 1e-8
        sample_weights = -(probs * torch.log(probs)).sum(1).numpy()

        if self.emb == "output":
            embedding = probs.numpy()
        else:
            embedding = self.get_diversity_embeddings(
                emb_type=self.emb,
                x=self.X[idxs_unlabeled],
                y=self.Y[idxs_unlabeled]
            )

        km = KMeans(n)
        km.fit(embedding, sample_weight=sample_weights)

        dists = euclidean_distances(km.cluster_centers_, embedding)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        end_time = time.time()

        sampling_df = pd.DataFrame([list(idxs_unlabeled[q_idxs])], index=["img_id"]).T
        self.save_stats(sampling_df)
        return idxs_unlabeled[q_idxs], end_time
