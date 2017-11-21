from tree_model import *
from dot_node import *

X = [(0, 0), (0, 1), (1,  0), (1, 1)]

#xor

Y = [(x * (1 - y) + y * (1 - x)) for x, y in X]


Y = np.array(Y)
X = np.array(X)

nodes = [DotNode(2) for _ in range(5)]

model = Model(nodes)

#model.fit(X, Y)

