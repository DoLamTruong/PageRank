import os.path
from mpi4py import MPI
import numpy as np
import timeit
from functools import reduce

# start time of program :
start = timeit.default_timer()

#define const 
DAMPING_FACTOR = 0.85 # d= 0.85
LENGTH = None

# MPI.Init()
Comm = MPI.COMM_WORLD
size = Comm.Get_size()
rank = Comm.Get_rank()

# processor 0 : is the main processor
if rank == 0:
    
    # readfile
    filename = input("File : ")
    data = np.loadtxt(filename, delimiter=",")

    LENGTH = len(data)
    # create buffer is Distribution (row) of any node 
    # example : [[0 1], [1,1]] has [1,2]
    sendDistribution = np.array([sum(ele) for ele in data])

    #create a buffer has (size) element from processor 0 to send all processor 
    #element of buffer is lines of file (row of Matrix)
    chunks = [[] for _ in range(size)]
    counts = 0
    for index in range(size - 1):
        for count in range(int(LENGTH/size)):
            chunks[index].append(data[counts])
            counts = counts + 1
    
    while counts < len(data):
        chunks[size - 1].append(data[counts])
        counts = counts + 1
    #end of create

else:
    data = None
    sendDistribution = None
    sendRowBuffer = None
    chunks = None

Comm.Barrier()

# all process has received the data (the rows of Matrix ) from process 0
recvRowBuffer = Comm.scatter(chunks, root = 0)
# distribution : all process has 
recv_Distribution = Comm.bcast(sendDistribution, root = 0)
LENGTH = Comm.bcast(LENGTH, root = 0)
# create a start resOld is all [1/length] or one of node is 1 and any is 0
resOld = np.full(len(recvRowBuffer),1/LENGTH)

# use the loop : 
# define number of loop
for index in range(0,19):

    # update data in processor 0 to use calculator of result
    updateData = Comm.gather(resOld,root = 0)

    if rank == 0:
        # updateData in line 65 has : list[numpy.ndarray+] => numpu.ndarray[element+]
        if type(updateData[0]) is np.ndarray:
            updateData = np.array(reduce(lambda a,b: a + b.tolist(), updateData, []))
    if rank != 0:
        updateData = None
    # send all data from process 0 to all process to calculator result
    resOld = Comm.bcast(updateData, root = 0)
    resNew = []
    # PAGE_RANK :
    for ele in recvRowBuffer:
        resx = DAMPING_FACTOR / LENGTH + (1- DAMPING_FACTOR) *sum( ele/recv_Distribution * resOld)
        resNew.append(resx)
    # UPDATE THE resOld
    resOld = np.array([ele for ele in resNew])
    Comm.Barrier()

#end time
stop = timeit.default_timer()

result = Comm.gather(resOld, root = 0)
if rank == 0:
    result = np.array(reduce(lambda a,b: a + b.tolist(), result, []))
    np.savetxt('results.csv', result, fmt="%f", delimiter=",")
    print("Time running is : ",(stop - start))
    
