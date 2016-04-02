#!/usr/bin/env python


import time

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#Basic Python Implementation
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

#NAME = 'NVIDIA CUDA'
#NAME = 'Apple'
NAME = 'Intel(R) Iris(TM) Graphics 6100'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

print devs

ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)
#vary l for different sizes
l = 80
P = 10
R = P*2*l
C = P*5*l
N = R*C

img = np.random.randint(0, 255, N).astype(np.uint8).reshape(R, C)


#img = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode = 'r',shape=(R,C))

# Kernel implementation for the improved method.This uses temp variable in local memory and then uses an shift which is local_size*no of groups.
#The barrier to local memory fence makes sure that the operation is first carried out. Atomic operations are used when multiple threads modify the same data
#In this case , all the data is first stored onto a local temp variable which is then coalesced and then added to the final histogram
kernel_1 = """
__kernel void histo1(__global unsigned char* img, __global unsigned int* final_bin, const int size) {
    unsigned int i = get_global_id(0);
    unsigned int icol = get_local_id(0);	
    unsigned int gid   = get_group_id(0);
    unsigned int ls = get_local_size(0);
    __local unsigned int temp[256];    
   
    temp[icol] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    int j = icol + gid * ls;
    int shift = ls * get_num_groups(0);
    while ( j < size)
    {
        atomic_add(&temp[img[j]], 1);
        j += shift;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(&final_bin[icol], temp[icol]);
}
"""
#Kernel provided by the TA
func = cl.Program(ctx, """
__kernel void func(__global unsigned char *img, __global unsigned int *bins,
                   const unsigned int P) {
    unsigned int i = get_global_id(0);
    unsigned int k;
    volatile __local unsigned char bins_loc[256];
    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (k=0; k<P; k++)
        ++bins_loc[img[i*P+k]];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k=0; k<256; k++)
        atomic_add(&bins[k], bins_loc[k]);
}
""").build().func

#Uncomment the lines below to get python execution time

start = time.time()
h_py = hist(img)
time_python = time.time() - start
#print h_py
#Using Atomic Increment Function gives a slightly better computation time 
kernel_2 = """
__kernel void histo2(__global unsigned char* img, __global unsigned int* final_bin, const int size) {
    unsigned int i = get_global_id(0);
    unsigned int icol = get_local_id(0);
    unsigned int gid   = get_group_id(0);
    unsigned int ls = get_local_size(0);
    __local unsigned int temp[256];

    temp[icol] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    int j = icol + gid * ls;
    int shift = ls * get_num_groups(0);
    while ( j < size)
    {
        atomic_inc(&temp[img[j]]);
        j += shift;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(&final_bin[icol], temp[icol]);
}
"""

#This is for the basic algorithm provided

func.set_scalar_arg_dtypes([None, None, np.uint32])
img_gpu = cl.array.to_device(queue, img)
bin_gpu = cl.array.zeros(queue, 256, np.uint32)
start = time.time()
var1 = func(queue, (N/32,), (1,), img_gpu.data, bin_gpu.data, np.uint32(P))
bin_basic = bin_gpu.get()
print bin_basic


#This is the algorithm implemented for Atomic_Add.
mf = cl.mem_flags

prg = cl.Program(ctx, kernel_1).build()
img_gpu = cl.array.to_device(queue, img)
new_bin = cl.array.zeros(queue, 256, np.uint32)
start = time.time()
var2 = prg.histo1(queue, (N, ), (256, ), img_gpu.data, new_bin.data, np.uint32(N))
bin_histo1 = new_bin.get()
print bin_histo1

prg = cl.Program(ctx, kernel_2).build()
img_gpu = cl.array.to_device(queue, img)
new_bin2 = cl.array.zeros(queue, 256, np.uint32)
start = time.time()
var3 = prg.histo2(queue, (N, ), (256, ), img_gpu.data, new_bin2.data, np.uint32(N))
bin_histo2 = new_bin2.get()
print bin_histo2

print "The size of the Image is 				  : ", N
print "The size of the image is : 					%d Mb" 	     %(N/1000000)
print "Python  Algorithm and Implemented Algorithm Comparison     : ", np.allclose(h_py, bin_basic)
print "Given Algorithm and Implemented algorithm Matrix comparison: ", np.allclose(bin_basic, bin_histo1)
print "Given Algorithm and Atomic Increment algo Matrix comparison: ", np.allclose(bin_basic, bin_histo2)

M = 3
times = []
# Calculate execution time for basic algo
func.set_scalar_arg_dtypes([None, None, np.uint32])
img_gpu = cl.array.to_device(queue, img)
for i in range(M):
    bin_gpu = cl.array.zeros(queue, 256, np.uint32)
    start = time.time()

    var1 = func(queue, (N/32,), (1,), img_gpu.data, bin_gpu.data, np.uint32(P))
    times.append(time.time() - start)
time_basic = np.average(times)
mf = cl.mem_flags

times = []
# Calculating execution time for the Atomic Add algorithm
prg = cl.Program(ctx, kernel_1).build()
img_gpu = cl.array.to_device(queue, img)
for i in range(M):
    new_bin = cl.array.zeros(queue, 256, np.uint32)
    start = time.time()
    var2 = prg.histo1(queue, (N, ), (256, ), img_gpu.data, new_bin.data, np.uint32(N))
    times.append(time.time() - start)
time_histo1 = np.average(times)


times = []
# Calculating execution time using the Atomic Increment algorithm
prg = cl.Program(ctx, kernel_2).build()
img_gpu = cl.array.to_device(queue, img)
for i in range(M):
    new_bin2 = cl.array.zeros(queue, 256, np.uint32)
    start = time.time()
    var3 = prg.histo2(queue, (N, ), (256, ), img_gpu.data, new_bin2.data, np.uint32(N))
    times.append(time.time() - start)
time_histo2 = np.average(times)
print "Python Execution Time:      ", time_python
print "Given Algorithm Time:            ", time_basic
print "Implemented Algorithm Time:      ", time_histo1
print "Atomic Increment Algorithm Time: ", time_histo2

