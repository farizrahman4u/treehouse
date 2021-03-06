# treehouse
Learn programs from data

---------

## Example

Learn to do the XOR function combing AND and OR gates:

```python

# AND gate

def _and(x):
    return int(x[0] and x[1])
 
# OR gate

def _or(x):
    return int(x[0] or x[1])

# Training data

X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]  # XOR

X = np.array(X)
Y = np.array(Y)

# Create model

model = Model([_and, _or])

# Train

model.fit(X, Y)

# Predict

model.predict(X)  # >>> [0, 1, 1, 0]

# Generate python-like code:

code = model.get_program()
print(code)

'''
if _and(input):
    if _or(input):
        output = 0
    else:
        output = 1
else:
    if _or(input):
        output = 1
    else:
        output = 0
'''


# Run the generated code on inputs using the built-in interpreter:

input = (0, 0)
interpreter.get_output(code, input, context=nodes)   # >>> 0

input = (0, 1)
interpreter.get_output(code, input, context=nodes)   # >>> 1

input = (1, 0)
interpreter.get_output(code, input, context=nodes)   # >>> 1

input = (0, 0)
interpreter.get_output(code, input, context=nodes)   # >>> 0

```
