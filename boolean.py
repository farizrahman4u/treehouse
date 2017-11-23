
class _Boolean(object):

	def get_data(self):
		raise NotImplemented

	def get_callables(self):
		raise NotImplemented

	def __call__(self, x):
		raise NotImplemented


class Boolean_0(_Boolean):

	def get_data(self):
		X = [(0, 0), (0, 1), (1, 0), (1, 1)]
		Y = [0, 1, 1, 0]
		return X, Y

	def get_callables(self):

		def OR(x, y):
			return int(x or y)

		def AND(x, y):
			return int(x and y)
		return [OR, AND]

	def __call__(self, x):
		return int((x and not y) or (y and not x))
