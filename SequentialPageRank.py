import timeit
import numpy as np

DAMPING_FACTOR = 0

print("Enter size: ")
SIZE = int(input())
EPSILON = 1e-10

start = timeit.default_timer()
filename = "values.csv"
data = np.loadtxt(filename, delimiter=",")
matrix_nor = np.array([sum(elem) for elem in data])

distribution = np.full(SIZE, 1/SIZE)
stepCounter = 0

while True:
    stepCounter = stepCounter + 1
    temp_distribution = np.array([0.0 for i in range(0, SIZE)])
    for j in range (0, SIZE):
        temp_distribution[j] = DAMPING_FACTOR / SIZE + (1 - DAMPING_FACTOR)*sum(distribution*(data[j] / matrix_nor))
    isClose = list(np.isclose(distribution, temp_distribution, atol = EPSILON))
    if False not in isClose:
        break
    distribution = temp_distribution

stop = timeit.default_timer()

print("Time: ", stop - start)
print("Nums of step: ", stepCounter)
print(distribution)
