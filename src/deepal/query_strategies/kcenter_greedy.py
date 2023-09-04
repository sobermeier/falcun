import time

import numpy as np
import pandas as pd
from .strategy import Strategy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


class KCenterGreedy(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
		super(KCenterGreedy, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.emb = emb

	def query(self, n):
		lb_flag = self.idxs_lb.copy()

		embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X, y=self.Y)
		dist_mat = euclidean_distances(embedding, embedding)

		mat = dist_mat[~lb_flag, :][:, lb_flag]

		for i in range(n):
			if i%10 == 0:
				print('greedy solution {}/{}'.format(i, n))
			mat_min = mat.min(axis=1)
			q_idx_ = mat_min.argmax()
			q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
			lb_flag[q_idx] = True
			mat = np.delete(mat, q_idx_, 0)
			mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

		chosen = np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]
		end_time = time.time()
		sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
		self.save_stats(sampling_df)

		return chosen, end_time


class KCenterGreedy2(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
		super(KCenterGreedy2, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.emb = emb

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_lb = np.arange(self.n_pool)[self.idxs_lb]

		embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X, y=self.Y)

		dist_mat = euclidean_distances(embedding[idxs_unlabeled], embedding[idxs_lb])
		min_dists = np.min(dist_mat, axis=-1)
		ind = min_dists.argmax()
		indsAll = [ind]
		features = [embedding[idxs_unlabeled[ind]]]
		while len(indsAll) < n:
			new_dist = pairwise_distances(embedding[idxs_unlabeled], [features[-1]]).ravel().astype(float)
			for i in range(len(embedding[idxs_unlabeled])):
				if min_dists[i] > new_dist[i]:
					min_dists[i] = new_dist[i]
			ind = min_dists.argmax()
			features.append(embedding[idxs_unlabeled[ind]])
			indsAll.append(ind)

		chosen = idxs_unlabeled[indsAll]
		end_time = time.time()
		sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
		self.save_stats(sampling_df)
		return chosen, end_time


class CdalCS(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(CdalCS, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.emb = "output"

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_lb = np.arange(self.n_pool)[self.idxs_lb]

		embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X, y=self.Y)

		dist_mat = euclidean_distances(embedding[idxs_unlabeled], embedding[idxs_lb])
		min_dists = np.min(dist_mat, axis=-1)
		ind = min_dists.argmax()
		indsAll = [ind]
		features = [embedding[idxs_unlabeled[ind]]]
		while len(indsAll) < n:
			new_dist = pairwise_distances(embedding[idxs_unlabeled], [features[-1]]).ravel().astype(float)
			for i in range(len(embedding[idxs_unlabeled])):
				if min_dists[i] > new_dist[i]:
					min_dists[i] = new_dist[i]
			ind = min_dists.argmax()
			features.append(embedding[idxs_unlabeled[ind]])
			indsAll.append(ind)

		chosen = idxs_unlabeled[indsAll]
		end_time = time.time()
		sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
		self.save_stats(sampling_df)
		return chosen, end_time
