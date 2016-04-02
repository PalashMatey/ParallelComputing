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
queue=cl.CommandQueue(ctx)          #COMMAND QUEUE


print "##############################################"
print "Matrix Transpose\n"

#### Defining the Initial Kernel ####

###func0: Naive Implementation ###
                    
func0= cl.Program(ctx,"""                   
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void mat_transpose(__global float* A, __global float *A_trans, unsigned int H_A, unsigned int W_A) {
        unsigned int i = get_global_id(0);
        unsigned int j = get_global_id(1);
    A_trans[i*H_A+j]=0;
        A_trans[i*H_A + j]= A[j*W_A +i];
}
""").build().mat_transpose                      #KERNEL

""" """
func0.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

def trans_naive(a_buf, atrans_buf, H_A, W_A):
    start = time.time()
    func0(queue, (W_A, H_A), None, a_buf, atrans_buf, np.uint32(H_A), np.uint32(W_A))
        return time.time()-start

def cl_naive_trans(a,a_trans,HA,WA):
    a_buf, atrans0_buf = mem_alloc(a, a_trans)
        t=trans_naive(a_buf,atrans0_buf, HA, WA)
        a_trans0=mem_transfer(a_trans,atrans0_buf)
    return t, a_trans0

###func1: Row Optimisation (All Global)###

func1= cl.Program(ctx,"""
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void mat_transpose(__global float* A, __global float *A_trans, unsigned int H_A, unsigned int W_A) {
        unsigned int i = get_global_id(0);
        unsigned int j;
    
    for (j=0;j<H_A;j++) {
                A_trans[i*H_A + j]=0.0;
        }
    for (j=0;j<H_A;j++) {
            A_trans[i*H_A + j]= A[j*W_A +i];
    }
}
""").build().mat_transpose                                              #KERNEL
""" """
func1.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

def trans_row_opt(a_buf, atrans_buf, H_A, W_A):
    start = time.time()
    func1(queue, (W_A, ), None, a_buf, atrans_buf, np.uint32(H_A), np.uint32(W_A))
    return time.time()-start

def cl_row_opt_trans(a,a_trans,HA,WA):
        a_buf, atrans1_buf = mem_alloc(a, a_trans)
        t=trans_row_opt(a_buf,atrans1_buf, HA, WA)
        a_trans1=mem_transfer(a_trans,atrans1_buf)
    return t, a_trans1

""" """

##########################################################################################


def create_arrays(height_A,width_A):
    A=np.random.random((height_A,width_A)).astype(np.float32)
    A_trans=np.zeros((width_A,height_A)).astype(np.float32)
    return A, A_trans

def mem_alloc(A, A_trans):
    mf=cl.mem_flags                             #MEMORY_FLAG allocation
    a_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    atrans_buf=cl.Buffer(ctx, mf.WRITE_ONLY, A_trans.nbytes)
    init_arr=np.zeros((W_A,H_A)).astype(np.float32)
    cl.enqueue_copy(queue,atrans_buf,init_arr)              #Initializing the Memory of the Output Buffers 
    return a_buf, atrans_buf    

def mem_transfer(A_trans, atrans_buf):
    cl.enqueue_copy(queue,A_trans,atrans_buf)                       #Copying Final Data into Python Buffers
    return A_trans

 
def py_trans(A,y):
    start=time.time()
    y=np.transpose(A)
    t= time.time()-start
    return t,y

def py_time(HA,WA,M=4):
    times = []
    A,y=create_arrays(HA,WA)
    for i in xrange(M):
            t,y=py_trans(A,y)
            times.append(t)
    #print 'python time:  ', np.average(times)
    return np.average(times)
 
def cl_naive_time(HA, WA, M=4):
    times = []
    a, atrans =create_arrays(HA,WA)
    a_buf, atrans_buf = mem_alloc(a, atrans)
    for i in xrange(M):
        t=trans_naive(a_buf,atrans_buf, HA, WA)
        times.append(t)
        atrans=mem_transfer(atrans,atrans_buf)
    #print 'opencl naive time:  ', np.average(times)
    return np.average(times)

def cl_row_opt_time(HA, WA, M=4):
        times = []
        a, atrans =create_arrays(HA,WA)
        a_buf, atrans_buf = mem_alloc(a, atrans)
        for i in xrange(M):
                t=trans_row_opt(a_buf,atrans_buf, HA, WA)
                times.append(t)
                atrans=mem_transfer(atrans,atrans_buf)
        #print 'opencl row_opt time:  ', np.average(times)
    return np.average(times)



######################################################################

### Initialising the parameters ####
H_A=6
W_A=8


a, atrans=create_arrays(H_A,W_A)
#print "A\n", a
a_buf, atrans_buf=mem_alloc(a,atrans)
atrans=mem_transfer(atrans, atrans_buf)


python_time,A_trans_py =py_trans(a, atrans)
#print "A' Python:\n", A_trans_py

### Verifying that the results are equal ###

pyopencl_time0, A_trans_cl0=cl_naive_trans(a,atrans,H_A,W_A)
#print "A' naive\n",A_trans_cl0
print "Output for Python-CPU and Naive-Kernel-GPU are equal:\t",np.allclose(A_trans_py,A_trans_cl0)

pyopencl_time1, A_trans_cl1=cl_row_opt_trans(a,atrans,H_A,W_A)
#print "A' row_opt\n",A_trans_cl1
print "Output for Python-CPU and RowOpt-Kernel-GPU are equal:\t",np.allclose(A_trans_py,A_trans_cl1)



#############################################################################################

### Comparing differnet pyopenCL kernel timings ###

python_times=[]
pyopencl_naive_times=[]
pyopencl_row_opt_times=[]


param=np.arange(1,201,1).astype(np.int32)

for i in param:
    python_times.append(py_time(i*H_A,i*W_A,4))
    pyopencl_naive_times.append(cl_naive_time(i*H_A,i*W_A,4))
    pyopencl_row_opt_times.append(cl_row_opt_time(i*H_A,i*W_A,4))

print "\nDim\t", "Python_time\t", "Naive_transpose\t", "Row Optimisation\t" 
for i in param:
    print "(",i*H_A, ",",i*W_A,")\t", python_times[i-1],"\t", pyopencl_naive_times[i-1], "\t", pyopencl_row_opt_times[i-1], "\t"

for i in param:
    if pyopencl_row_opt_times[i-1] < pyopencl_naive_times[i-1]:
        print "\nAt a dimension size of (", i*H_A, ",", i*W_A, "), Row Optimization beats Naive Transpose implementations" 
        break



 
plt.clf()
plt.plot(param*H_A*param*W_A, pyopencl_naive_times, 'r*-',
     param*H_A*param*W_A, pyopencl_row_opt_times, 'b*-',
     param*H_A*param*W_A, python_times, 'k*-')

plt.xlabel('# elements in matrix A')
plt.ylabel('$t$')
plt.title('Time vs Size for different Transpose Implementations')
plt.legend(('Naive-Kernel-GPU', 'RowOpt-Kernel-GPU', 'Python-CPU'), loc='upper left')
plt.grid(True)
#plt.draw()
plt.savefig('Transpose_scaling.png')



