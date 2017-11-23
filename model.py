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
		self.nps_lr = np.arange(self.nps.size).reshape(self.nps.shape) + 1
		self.nps_lr = np.cast[float](self.nps_lr)
		self.nps_lr *= self.lr
		self.lps = self._random_prob_dist((expd, self.num_classes))
		self.lps_lr = np.arange(self.lps.size).reshape(self.lps.shape) + 1 + self.nps.size
		self.lps_lr = np.cast[float](self.lps_lr)
		self.lps_lr *= self.lr
		num_total = self.nps.size + self.lps.size
		self.nps_lr /= num_total
		self.lps_lr /= num_total


	def forward(self, x, return_history=True):
		visited_nps = []
		current_np = 0
		nodes = self.nodes
		nps = self.nps
		lps = self.lps
		num_nps = len(nps)
		while(True):
			if current_np >= num_nps:
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
		nps_lr = self.nps_lr
		lps_lr = self.lps_lr
		for epoch in range(epochs):
			print('Epoch ' + str(epoch) + ':')
			pbar = ProgressBar(len(X))
			updated = False
			for x, y in zip(X, Y):
				y_, lp, nps_ = self.forward(x)
				if y == y_:
					for np_idx in nps_:
						np_ = nps[np_idx]
						np_lr = nps_lr[np_idx]
						mx = np.argmax(np_)
						update = np_.copy()
						update *= np_lr
						update[mx] = 0
						s = update.sum()
						update *= -1
						update[mx] = s
						np_ += update
					lp_ = lps[lp]
					lp_lr = lps_lr[lp]
					mx = np.argmax(lp_)
					update = lp_.copy()
					update *= lp_lr
					update[mx] = 0
					s = update.sum()
					update *= -1
					update[mx] = s
					lp_ += update
				else:
					for np_idx in nps_:
						np_ = nps[np_idx]
						np_lr = nps_lr[np_idx]
						mx = np.argmax(np_)
						update = np.zeros_like(np_)
						mx_val = np_[mx]
						delta = mx_val * np_lr[mx]
						update += delta / (num_nodes - 1)
						update[mx] = - delta
						np_ += update
					lp_ = lps[lp]
					lp_lr = lps_lr[lp]
					mx = np.argmax(lp_)
					update = np.zeros_like(lp_)
					mx_val = lp_[mx]
					delta = mx_val * lp_lr[mx]
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

	def _get_node_name(self, node):
		if hasattr	(node, '__name__'):
			return node.__name__
		if hasattr(node, '__class__'):
			node = node.__class__.__name__
		return str(node)

	def get_program(self):
		lines = ['\n']
		nodes = self.nodes
		lps = self.lps
		nps = self.nps
		num_nodes = len(nps)
		previous_node = None
		def _get_code(n_p=0, ind=0, values={}):
			if n_p >= num_nodes:
				n_p -= num_nodes
				label = np.argmax(lps[n_p])
				line = ' ' * ind + 'print(' + str(label) + ')'
				lines.append(line)
			else:
					node = nodes[np.argmax(nps[n_p])]
					node_name = self._get_node_name(node)
					node_val = values.get(node, None)
					if node_val is None:
						line = ' ' * ind + 'if ' + node_name + '(input)' + ':'
						lines.append(line)
						ind += 4
						values[node] = True
						left = 2 * n_p + 1
						right = left + 1
						if_statement_idx = len(lines) - 1
						_get_code(right, ind, values)
						values[node] = False
						ind -= 4
						lines.append(' ' * ind + 'else:')
						else_statement_idx = len(lines) - 1
						ind += 4
						_get_code(left, ind, values)
						values[node] = None
						ind -= 4
						# check for if(cond){x}else{x}; kinda hackish, will fix later
						if_block = lines[if_statement_idx + 1: else_statement_idx]
						else_block = lines[else_statement_idx + 1:]

						if if_block == else_block:
							for _ in range(len(lines) - if_statement_idx):
								lines.pop()
							for s in if_block:
								lines.append(s[4:])
					elif node_val:
						right = 2 * n_p + 2
						_get_code(right, ind, values)
					else:
						left = 2 * n_p + 1
						_get_code(left, ind, values)


		_get_code()
		return '\n'.join(lines)




