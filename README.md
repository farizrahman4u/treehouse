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

# Get human readable pseudo-code

model.get_program()

'''
if _and(input):
    if _or(input):
        print(0)
    else:
        print(1)
else:
    if _or(input):
        print(1)
    else:
        print(0)
'''
```
