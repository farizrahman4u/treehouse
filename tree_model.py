import numpy as np
import numpy.ma as ma
from progressbar import ProgressBar
import math


class Model(object):

	def __init__(self, nodes, num_classes=2, depth=5, lr=0.1):
		self.nodes = nodes
		self.num_classes = num_classes
		self.depth = depth
		self.lr = lr
		self.build()

	def _random_prob_dist(self, shape):
		x = np.random.uniform(-1, 1, shape)
		e = np.exp(x - np.max(x, axis=-1, keepdims=True))
		s = np.sum(e, axis=-1, keepdims=True)
		return e / s

	def build(self):
		expd = 2 ** self.depth
		self.nps = self._random_prob_dist((expd - 1, len(self.nodes)))
		self.lps = self._random_prob_dist((expd, self.num_classes))

	def forward(self, x, return_history=True):
		visited_nps = []
		current_np = 0
		nodes = self.nodes
		nps = self.nps
		lps = self.lps
		num_nps = len(nps)
		while(True):
			if current_np >= num_nps:
				print current_np
				visited_lp = current_np - num_nps
				label = np.argmax(lps[visited_lp])
				break
			visited_nps.append(current_np)
			node = nodes[np.argmax(nps[current_np])]
			node_out = node(x)
			current_np = 2 * current_np + 1 + node_out
		if return_history:
			return label, visited_lp, visited_nps
		return label

	def fit(self, X, Y, epochs=10):
		nps = self.nps
		lps = self.lps
		num_nodes = len(self.nodes)
		lr = self.lr
		for epoch in range(epochs):
			print('Epoch ' + str(epoch) + ':')
			pbar = ProgressBar(len(X))
			updated = False
			for x, y in zip(X, Y):
				y_, lp, nps_ = self.forward(x)
				if y == y_:
					for np_idx in nps_:
						np_ = nps[np_idx]
						mx = np.argmax(np_)
						update = np_.copy()
						update *= lr
						update[mx] = 0
						s = update.sum()
						update *= -1
						update[mx] = s
						np_ += update
					lp_ = lps[lp]
					mx = np.argmax(lp_)
					update = lp_.copy()
					update *= lr
					update[mx] = 0
					s = update.sum()
					update *= -1
					update[mx] = s
					lp_ += update
				else:
					for np_idx in nps_:
						np_ = nps[np_idx]
						mx = np.argmax(np_)
						update = np.zeros_like(np_)
						mx_val = np_[mx]
						delta = mx_val * lr
						update += delta / (num_nodes - 1)
						update[mx] = - delta
						np_ += update
					lp_ = lps[lp]
					mx = np.argmax(lp_)
					update = np.zeros_like(lp_)
					mx_val = lp_[mx]
					update += delta / (num_nodes - 1)
					update[mx] = - delta
					lp_ += update
					updated = True
				pbar.update()
			if not updated:
				break


	def predict(self, X):
		f = self.forward
		Y = [f(x, False) for x in X]
		return np.array(Y)

	def evaluate(self, X, Y):
		Y = np.array(Y, dtype=int)
		Y_ = np.array(self.predict(X), dtype=int)
		return float(np.sum(Y==Y_)) / len(Y)


