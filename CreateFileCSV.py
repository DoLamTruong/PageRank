import numpy as np

x = np.random.randint(2, size=(100, 100))

np.savetxt('values.csv', x, fmt="%d", delimiter=",")