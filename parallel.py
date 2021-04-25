import os.path
from mpi4py import MPI
import numpy as np
import timeit
from functools import reduce

DAMPING = 0.8
EPSILON = 1e-10

print("Enter size: ")
size = int(input())

# create initial vector v0
x = np.random.randint(2, size=(size, size))
filename = "data.txt"
normal = 1/size
value = np.full(size, normal)
print(value)
# read data from file
# we will get what
# edgeMatrix = np.zeros((size, size))
# sumRow =  [0 for col in range(size)]
# transitionMatrix = np.zeros((size, size))

# filename = "data.txt"
# f = open(filename)
# #   fromNode,toNode = np.fromfile(f, dtype=int, count=2, sep=" ")
# # with open('filename.txt') as fp:
# for line in f:
#     fromNode,toNode=list(map(int, line.split()))
#     edgeMatrix[fromNode][toNode] = 1
#     sumRow[fromNode] += 1
# # data = np.loadtxt(filename)
# print(edgeMatrix)

# initVectorV = np.full(size, 1/size)
# print(initVectorV)
# print(sumRow)
# for i in range(size):
#     for j in range(size):
#         if edgeMatrix[i][j]:
#             transitionMatrix[i][j] = 1/sumRow[i]

# print(transitionMatrix)

def readFiletogetMatrix(fileName):
    edgeMatrix = np.zeros((size, size))
    sumRow =  [0 for col in range(size)]
    transitionMatrix = np.zeros((size, size))

    
    f = open(filename)
    #   fromNode,toNode = np.fromfile(f, dtype=int, count=2, sep=" ")
    # with open('filename.txt') as fp:
    for line in f:
        fromNode,toNode=list(map(int, line.split()))
        edgeMatrix[fromNode][toNode] = 1
        sumRow[fromNode] += 1
    # data = np.loadtxt(filename)
    # print(edgeMatrix)

    
    # print(sumRow)
    for i in range(size):
        for j in range(size):
            if edgeMatrix[i][j]:
                transitionMatrix[j][i] = 1/sumRow[i]

    # print(transitionMatrix)
    return transitionMatrix

transitionMatrix = readFiletogetMatrix(filename)
step = 0
preventError = (1-DAMPING)/size

start = timeit.default_timer()

while True:
    step += 1
    tempValue = [0 for col in range(size)]
    for i in range (size):
        tempValue[i] = preventError + DAMPING* sum(value*transitionMatrix[i])
    isClose = list(np.isclose(tempValue, value, atol = EPSILON))
    print(tempValue)
    if False not in isClose:
        break
    value = tempValue

stop = timeit.default_timer()
print("Step: ", step)
print("Time: ", stop - start, "=====================")
print(value)
