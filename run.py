from model import Model
import numpy as np
import interpreter


# Learn to do XOR using AND and OR gates


X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]


X = np.array(X)
Y = np.array(Y)


def _and(x):
    return int(x[0] and x[1])


def _or(x):
    return int(x[0] or x[1])


nodes = [_and, _or]


model = Model(nodes, depth=5)

model.fit(X, Y, 20)

code = model.get_program()
print(code)

assert model.evaluate(X, Y) == 1.0

# Test generated code

for x, y in zip(X, Y):
    x = tuple(x)
    assert interpreter.get_output(code, x, nodes) == y, interpreter.get_output(code, x, nodes) 
