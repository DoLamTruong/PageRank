import os.path
from mpi4py import MPI
import numpy as np
import timeit
from functools import reduce

DAMPING = 0.8
EPSILON = 1e-10
flatBreak = False

Comm = MPI.COMM_WORLD
rank = Comm.Get_rank()
sizeMPI = Comm.Get_size()

if rank == 0:

    print("Enter size: ")
    size = int(input())

    filename = "data2.txt"
    normal = 1/size
    value = np.full(size, normal)
    chunksValue = []
    lengthEachMPI = int(size/sizeMPI)
    # print("lengthEachMPI: ", lengthEachMPI)
    # print("sizeMPI: ", sizeMPI)
    for i in range(sizeMPI):
        if i == sizeMPI-1:
            chunksValue.append(value[i*lengthEachMPI:])
        else:
            chunksValue.append(value[i*lengthEachMPI:(i+1)*lengthEachMPI])
        # print("chunksValue[", i, "] = ", chunksValue[i])
    
    # readFiletogetMatrix
    edgeMatrix = np.zeros((size, size))
    sumRow =  np.zeros(size)
    transitionMatrix = np.zeros((size, size))

    f = open(filename)
    for line in f:
        fromNode,toNode=list(map(int, line.split()))
        edgeMatrix[fromNode][toNode] = 1
        sumRow[fromNode] += 1

    for i in range(size):
        for j in range(size):
            if edgeMatrix[i][j]:
                # transitionMatrix is saved with collum, row order
                transitionMatrix[j][i] = 1/sumRow[i]
    step = 0
    start = timeit.default_timer()

    preventError = (1-DAMPING)/size
else:
    size =None
    transitionMatrix = None
    preventError= None
    value= None
    lengthEachMPI= None
    chunksValue = None

size = Comm.bcast(size, root = 0)
transitionMatrix = Comm.bcast(transitionMatrix, root = 0)
preventError = Comm.bcast(preventError, root = 0)
# value = Comm.bcast(value, root = 0)
lengthEachMPI = Comm.bcast(lengthEachMPI, root = 0)
if rank == sizeMPI -1:
    lengthEachMPI += size%sizeMPI

tempChunksValue = Comm.scatter(chunksValue, root = 0)
# print("rank ", rank, "chunk ", tempChunksValue)

Comm.Barrier()
while True:
    chunksValue = Comm.gather(tempChunksValue,root=0)
    if rank == 0:
        step += 1
        newValue = []
        for ind in chunksValue:
            newValue.extend(ind)
        # print("step", step-1)
        # print("new: ", newValue)
        # print("value ", value)
        close = list(np.isclose(newValue, value, atol = EPSILON))
        if False not in close and step > 2:
            flatBreak = True
        value = newValue
        tempValue = [0 for col in range(size)]
        tempChunksValue = []
        for i in range(sizeMPI):
            if i == (sizeMPI-1):
                tempChunksValue.append(value[i*lengthEachMPI:])
            else:
                tempChunksValue.append(value[i*lengthEachMPI:(i+1)*lengthEachMPI])
        
        
    value = Comm.bcast(value, root = 0)
    tempChunksValue = Comm.scatter(tempChunksValue, root = 0)
    for i in range (lengthEachMPI):
        tempChunksValue[i] = preventError + DAMPING* sum(value*transitionMatrix[rank*lengthEachMPI + i])
    

    
    # value = tempValue
    Comm.Barrier()
    if flatBreak:
        break
    

if rank == 0:
    stop = timeit.default_timer()
    print("Step: ", step)
    print("Time: ", stop - start, "=====================")
    print(value)