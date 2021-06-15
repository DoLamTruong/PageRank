import numpy as np

x = np.random.randint(500, size=(30000, 2))

np.savetxt('data10.txt', x, fmt="%d", delimiter=" ")
# Nodes: 875713 Edges: 5105039