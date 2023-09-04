""" IDEAL Sampling Strategies """
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
import torch
from sklearn.preprocessing import MinMaxScaler

from .strategy import Strategy


class Falcun(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, gamma=10, deterministic=False, custom_dist="distance"):
        super(Falcun, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.deterministic = deterministic
        self.gamma = gamma
        self.custom_dist = custom_dist

    def get_unc(self, probs, uncertainty="margin"):
        probs_sorted, idxs = probs.sort(descending=True)
        if uncertainty == "margin":
            return 1 - (probs_sorted[:, 0] - probs_sorted[:, 1])
        elif uncertainty == "entropy":
            probs += 1e-8
            entropy = -(probs * torch.log(probs)).sum(1)
            return (entropy - entropy.min()) / (entropy.max() - entropy.min())
        else:  # lc
            return 1 - probs_sorted[:, 0]

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

        unc = self.get_unc(probs, uncertainty="margin")

        if self.deterministic and self.custom_dist == "unc":
            _, idx = unc.sort(descending=True)
            sel_ids = idx[:n]
        else:
            sel_ids = self.get_indices(n, probs.numpy(), unc.numpy())
        end_time = time.time()

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(sel_ids)] = True

        probs_sorted, idxs = probs.sort(descending=True)
        margin = 1 - (probs_sorted[:, 0] - probs_sorted[:, 1])
        sampling_df = pd.DataFrame([list(idxs_unlabeled), list(chosen_all), margin.tolist()], index=["img_id", "chosen", "score"]).T
        self.save_stats(sampling_df)
        return idxs_unlabeled[sel_ids], end_time

    def update_gamma(self):
        self.gamma += 1

    def get_indices(self, n, probs, unc):
        ind_selected = []
        vec_selected = []

        dists = unc.copy()
        unlabeled_range = np.arange(len(dists))
        candidate_mask = np.ones(len(dists), dtype=bool)

        while len(vec_selected) < n:
            if len(vec_selected) > 0:
                new_dists = pairwise_distances(probs, [vec_selected[-1]], metric="l1").ravel().astype(float)
                dists = np.array([dists[i] if dists[i] < new_dists[i] else new_dists[i] for i in range(len(probs))])
            if self.deterministic:
                if self.custom_dist == "distance":
                    ind = dists.argmax()
                elif self.custom_dist == "distance_unc_norm":
                    scaler = MinMaxScaler((0, 1))
                    scaler.fit(dists[candidate_mask].reshape(-1, 1))
                    _dists = scaler.transform(dists.reshape(-1, 1)).ravel()
                    _rel = _dists + unc
                    ind = _rel.argmax()
            else:
                if sum(dists[candidate_mask]) > 0:

                    if self.custom_dist == "distance_unc_norm":
                        scaler = MinMaxScaler((0, 1))
                        scaler.fit(dists[candidate_mask].reshape(-1, 1))
                        _dists = scaler.transform(dists[candidate_mask].reshape(-1, 1)).ravel()
                        # x = dists[candidate_mask]
                        # _dists = (x - np.min(x)) / (np.max(x) - np.min(x))
                        dist_probs = (_dists + unc[candidate_mask]) ** self.gamma / sum((_dists + unc[candidate_mask]) ** self.gamma)
                    elif self.custom_dist == "distance":
                        dist_probs = dists[candidate_mask] ** self.gamma / sum(dists[candidate_mask] ** self.gamma)
                    elif self.custom_dist == "unc":
                        dist_probs = unc[candidate_mask] ** self.gamma / sum(unc[candidate_mask] ** self.gamma)
                    else:
                        raise NotImplementedError

                    # print(f"MAX DIST PROB: {np.max(dist_probs)}")
                    ind = np.random.choice(unlabeled_range[candidate_mask], size=1, p=dist_probs)[0]
                else:
                    ind = np.random.choice(unlabeled_range[candidate_mask], size=1)[0]
            candidate_mask[ind] = False
            vec_selected.append(probs[ind])
            ind_selected.append(ind)
        return ind_selected
