import numpy as np

B = np.random.rand(4, 4)
B = B / np.sum(B, axis=1)

print(np.sum(B, axis=1))

print(B)
