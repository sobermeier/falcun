import time

import numpy as np
import pandas as pd

from .strategy import Strategy


class RandomSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		chosen = np.random.choice(idxs_unlabeled, n, replace=False)
		end_time = time.time()
		sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
		self.save_stats(sampling_df)
		return chosen, end_time
