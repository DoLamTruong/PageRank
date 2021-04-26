import numpy as np

x = np.random.randint(500, size=(20000, 2))

np.savetxt('data2.txt', x, fmt="%d", delimiter=" ")