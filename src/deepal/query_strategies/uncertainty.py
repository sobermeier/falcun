import time

import numpy as np
import pandas as pd

from .strategy import Strategy
from ..utils import calc_unc


class UncertaintySampling(Strategy):
    unc_type = "entropy"
    deterministic = True

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UncertaintySampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # larger is more uncertain
        uncertainty = calc_unc(probs=probs, unc=self.unc_type)

        if self.deterministic:
            _, idx = uncertainty.sort(descending=True)
            q_idx = idx[:n]
        else:
            unc_probs = (uncertainty.numpy() ** 2) / sum(uncertainty.numpy() ** 2)
            q_idx = np.random.choice(np.arange(len(unc_probs)), replace=False, size=n, p=unc_probs)
        chosen = idxs_unlabeled[q_idx]
        end_time = time.time()

        # save stats
        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(q_idx)] = True
        sampling_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), uncertainty.tolist()], index=["img_id", "chosen", "score"]).T
        self.save_stats(sampling_df)
        return chosen, end_time


class MarginSampling(UncertaintySampling):
    unc_type = "margin"

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(MarginSampling, self).__init__(X, Y, idxs_lb, net, handler, args)


class EntropySampling(UncertaintySampling):
    unc_type = "entropy"
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(EntropySampling, self).__init__(X, Y, idxs_lb, net, handler, args)


class LeastConfidence(UncertaintySampling):
    unc_type = "lc"
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args)


class MarginWeighted(UncertaintySampling):
    unc_type = "margin"
    deterministic = False

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(MarginWeighted, self).__init__(X, Y, idxs_lb, net, handler, args)


class EntropyWeighted(UncertaintySampling):
    unc_type = "entropy"
    deterministic = False

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(EntropyWeighted, self).__init__(X, Y, idxs_lb, net, handler, args)


class LeastConfidenceWeighted(UncertaintySampling):
    unc_type = "lc"
    deterministic = False

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(LeastConfidenceWeighted, self).__init__(X, Y, idxs_lb, net, handler, args)
