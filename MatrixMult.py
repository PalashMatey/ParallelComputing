#!/usr/bin/env python

import time
import csv
import numpy as np
import pyopencl as cl
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['savefig.dpi'] = 100

from pylab import *

ctx = cl.create_some_context()
queue=cl.CommandQueue(ctx)			#COMMAND QUEUE



print "###############################################################"
print "Multiplication of y=A*B"
print ""

#### Defining the Naive Kernel ####
					
func_mult_naive= cl.Program(ctx,"""					
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void mat_mult(__global float* A, __global float* B, __global float* D, unsigned int SIZE) {
        unsigned int i = get_global_id(0);
        unsigned int j = get_global_id(1);
	unsigned int k;	
	float temp=0.0;
	for (k=0; k<SIZE; k++) {
	
		temp += (A[i*SIZE + k] * B[k*SIZE + j]);	
	}
	D[i*SIZE + j] = temp;
}
""").build().mat_mult						#KERNEL
func_mult_naive.set_scalar_arg_dtypes([None, None, None, np.uint32])

def mult_op_naive(a_buf, b_buf, d_buf, siz):
    start = time.time()
    func_mult_naive(queue, (siz,siz), None, a_buf, b_buf, d_buf, np.uint32(siz))
    return time.time()-start


def cl_op_mult_naive(a,b,d,siz):
        a_buf,b_buf,d_buf = mem_alloc(a,b,d)
        t=mult_op_naive(a_buf,b_buf,d_buf, siz)
        d=mem_transfer(d,d_buf)
        return t, d



#############################################################################################


#### Defining the Tiled Kernel ####

func_mult_tiling= cl.Program(ctx,"""
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void mat_mult(__global float* A, __global float* B, __global float* D, unsigned int SIZE, unsigned int m, unsigned int n, unsigned int p) {
    __local float AS[1024];
    __local float BCS[1024];
    int i = get_global_id(1);
    int j = get_global_id(0);
	
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int aBegin = n* SIZE * by;
    int aEnd   = aBegin + n - 1;
    int aStep  = SIZE;
    int bBegin = SIZE * bx;
    int bStep  = SIZE * p;
    float temp = 0.0f;
    for (int a = aBegin, b = bBegin; a <= aEnd;a += aStep, b += bStep) 
    {
        AS[tx + ty*SIZE] = A[a + n * ty + tx];
        BCS[tx + ty*SIZE] = B[b + p*ty + tx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < SIZE; ++k)
            temp += AS[ty*SIZE + k] * BCS[k*SIZE + tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    D[i * p + j] = temp;
}
""").build().mat_mult                                              #KERNEL
func_mult_tiling.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32, np.uint32])

def mult_op_tiling(a_buf, b_buf, d_buf, siz, m, n, p):
    start = time.time()
    func_mult_tiling(queue, (m,p), (siz,siz), a_buf, b_buf, d_buf, np.uint32(siz), np.uint32(m), np.uint32(n), np.uint32(p))
    return time.time()-start


def cl_op_mult_tiling(a,b,d,siz,m,n,p):
        a_buf,b_buf,d_buf = mem_alloc(a,b,d)
        t=mult_op_tiling(a_buf,b_buf,d_buf, siz, m,n,p)
        d=mem_transfer(d,d_buf)
        return t, d



#############################################################################################



def create_arrays(size):
	A=np.random.random((size,size)).astype(np.float32)
	B=np.random.random((size,size)).astype(np.float32)
	#C=np.random.random((size,size)).astype(np.float32)
	D=np.zeros((size,size)).astype(np.float32)
	return A, B, D

def create_arrays_2(m,n,p):
	A=np.random.random((m,n)).astype(np.float32)
        B=np.random.random((n,p)).astype(np.float32)
        #C=np.random.random((n,p)).astype(np.float32)
        D=np.zeros((m,p)).astype(np.float32)
        return A, B, D


def mem_alloc(A, B, D):
	mf=cl.mem_flags								#MEMORY_FLAG allocation
	a_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
	b_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
	#c_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
	d_buf=cl.Buffer(ctx, mf.WRITE_ONLY, D.nbytes)
	init_arr=np.zeros(D.shape).astype(np.float32)
	cl.enqueue_copy(queue,d_buf,init_arr)				#Initializing the Memory of the Output Buffers 
	return a_buf, b_buf, d_buf	



#prg=cl.Program(ctx,kernel).build()					#PROGRAM
#prg.mat_transpose(queue,A_trans.shape,None,a_buf,atrans_buf,np.uint32(height_A),np.uint32(width_A)) 	#KERNEL LAUNCH
def mem_transfer(D, d_buf):
	cl.enqueue_copy(queue,D,d_buf)						#Copying Final Data into Python Buffers
	return D

#print "A\n", A
#print "A_transpose_OpenCL\n", A_trans
#print "A_transpose_python\n", A_trans2
#print 'equal:        ', np.allclose(A_trans, A_trans2)
 
def py_calc(A,B,y):
	start=time.time()
	y=np.dot(A,B)
	t= time.time()-start
	return t,y

def py_calc_time(siz,M=4):
	times = []
	a, b, d =create_arrays(siz)
        a_buf, b_buf, d_buf = mem_alloc(a, b, d)
	for i in xrange(M):
    		t,y=py_calc(a,b,d)
    		times.append(t)
	#print 'python time:  ', np.average(times)
	return np.average(times)
 
def cl_op_naive_time(siz, M=4):
	times = []
	a, b, d =create_arrays(siz)
	a_buf, b_buf, d_buf = mem_alloc(a, b, d)
	for i in xrange(M):
		t=mult_op_naive(a_buf,b_buf,d_buf, siz)
		times.append(t)
		d=mem_transfer(d,d_buf)
	#print 'opencl time:  ', np.average(times)
	return np.average(times)

def cl_op_tiling_time(siz,m,n,p, M=4):
        times = []
        a, b, d =create_arrays_2(m,n,p)
        a_buf, b_buf, d_buf = mem_alloc(a, b, d)
        for i in xrange(M):
                t=mult_op_tiling(a_buf,b_buf, d_buf, siz, m, n, p)
                times.append(t)
                d=mem_transfer(d,d_buf)
        #print 'opencl time:  ', np.average(times)
	return np.average(times)


#### Testing & Running the code


### Initialising the parameters ####
SIZE=32
m=SIZE
n=SIZE
p=SIZE

a, b, d=create_arrays(SIZE)
#print "A\n", a
#print "B\n", b
#print "C\n", c
a_buf, b_buf, d_buf=mem_alloc(a,b,d)
d=mem_transfer(d, d_buf)


python_time,D_py =py_calc(a, b, d)
#print "D Python:\n", D_py

pyopencl_time4, D_cl4=cl_op_mult_naive(a,b,d,SIZE)
#print "A' op0\n",A_trans_cl0
print "Output for Python-CPU and Naive-Kernel-GPU are equal:\t",np.allclose(D_py,D_cl4)

pyopencl_time5, D_cl5=cl_op_mult_tiling(a,b,d,SIZE,m,n,p)
#print "A' op0\n",A_trans_cl0
print "Output for Python-CPU and Tiling-Kernel-GPU are  equal:\t",np.allclose(D_py,D_cl5)


#########################################################

### Comparing python & pyopenCL timings ###

python_times=[]
pyopencl_op_naive_times=[]
pyopencl_op_tiling_times=[]

param=np.arange(1,40,1).astype(np.int32)

for i in param:
        python_times.append(py_calc_time(i*SIZE,4))
        pyopencl_op_naive_times.append(cl_op_naive_time(i*SIZE,4))
        pyopencl_op_tiling_times.append(cl_op_tiling_time(SIZE,i*SIZE,i*SIZE,i*SIZE,4))

l_index=len(python_times)-1
naive_speedup=python_times[l_index]/pyopencl_op_naive_times[l_index]
tiling_speedup=python_times[l_index]/pyopencl_op_tiling_times[l_index]


print "\nDim\t", "\tPython_time\t", "\tNaive_time\t", "Tiling_time\t" 
for i in param:
        print "(",i*SIZE, ",",i*SIZE,")\t", python_times[i-1],"\t", pyopencl_op_naive_times[i-1], "\t", pyopencl_op_tiling_times[i-1], "\t"#, pyopencl_op2_times[i-1],"\t"#, pyopencl_op3_times[i]

for i in param:
	if pyopencl_op_tiling_times[i-1]<python_times[i-1]:
		print "\nAfter (", i*SIZE, ",",i*SIZE, ") pyopenCL Tiling is faster than python."
		break
for i in param:
	if pyopencl_op_naive_times[i-1]<python_times[i-1]:
                print "After (", i*SIZE, ",", i*SIZE,") pyopenCL Naive is faster than python."
                break

print "Avg speedup factor for multiplication is:", (tiling_speedup +naive_speedup)/2

plt.clf()
plt.plot(param*SIZE, python_times, 'bo-',
         param*SIZE, pyopencl_op_naive_times, 'r*-',
         param*SIZE, pyopencl_op_tiling_times, 'go-')

plt.xlabel('elements in square matrix A,B')
plt.ylabel('$t$')
plt.title('Time vs Size for different Multiplication Implementations')
plt.legend(('Python-CPU', 'Naive-Kernel-GPU', 'Tiling-Kernel-GPU'), loc='upper left')
plt.grid(True)
plt.gca().set_xlim((min(param*SIZE), max(param*SIZE)))
plt.gca().set_ylim((0, 1.2*max(python_times)))
#plt.draw()
plt.savefig('Multiplication_scaling.png')



