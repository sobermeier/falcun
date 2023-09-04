import time

import numpy as np
import torch
import pandas as pd
from .strategy import Strategy


class BALDDropout(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
		super(BALDDropout, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.n_drop = n_drop

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)  # mean probability
		entropy1 = (-pb*torch.log(pb)).sum(1)  # entropy on mean prediction
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)  # mean over entropies
		U = entropy2 - entropy1  # mutual information

		scores, idx = U.sort()
		chosen = idxs_unlabeled[idx[:n]]
		end_time = time.time()

		# save stats
		chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
		chosen_all[list(idx[:n])] = True
		sampling_df = pd.DataFrame(
			[list(idxs_unlabeled), list(chosen_all), U.tolist()], index=["img_id", "chosen", "score"]).T
		self.save_stats(sampling_df)

		return chosen, end_time
