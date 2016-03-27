import time
import pyopencl as cl
import numpy as np



NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
	if platform.name == NAME:
		devs = platform.get_devices()


ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

#Defining Values for L,M & N. Change these to observe change in Execution times.
L=50
M=40
N=60
max = 10
#Generating the Two matrices. Making sure they consist of complex numbers.
x = np.random.randint(0, max, size = (L,M)).astype(np.complex64) + 1j*np.random.randint(0, max, size = (L,M)).astype(np.complex64)
c = np.random.randint(0, max, size = (M,N)).astype(np.complex64) + 1j*np.random.randint(0, max, size = (M,N)).astype(np.complex64)
#Getting the python multiplication output.
py_val = np.dot(x,c)

print x
print c
#Defining the Kernel for Basic Matrix Multiplication
kernel1 = """
#include <pyopencl-complex.h>
__kernel void matmul1(__global cfloat_t* X, __global cfloat_t* C, __global cfloat_t* Y , const int L, const int M, const int N) {
	
	unsigned int Row = get_global_id(0);
	unsigned int Col = get_global_id(1);
	cfloat_t temp = 0;
	for (int i = 0; i < M; i++)
	{
		temp = temp + cfloat_mul(X[Row*M + i], C[i*N+Col]);
	}
	Y[Row*N + Col] = temp;
}
"""

#Defining the Kernel for Optimization 1. Reduce work item overhead. Whole row of Y per work item.
kernel2 = """
#include <pyopencl-complex.h>
__kernel void matmul2(__global cfloat_t* X, __global cfloat_t* C, __global cfloat_t* Y , const int L, const int M, const int N) {
	
	unsigned int Col, k;
	unsigned int Row = get_global_id(0);
	for (Col = 0; Col < N ; Col++)
	{
		cfloat_t temp = 0;
		for (k=0; k < M ; k++)
		{
			temp = temp + cfloat_mul(X[Row*M + k], C[k*N + Col]);
		}
		Y[Row*N + Col] = temp;
	}
}
"""

#Defining the Kernel for Optimization 2. Setup a work array for A in private memory and copy into it from global memory before we start with the matrix multiplications.

kernel3 = """
#include <pyopencl-complex.h>
__kernel void matmul3(__global cfloat_t* X, __global cfloat_t* C, __global cfloat_t* Y , const int L, const int M, const int N) {
	
	unsigned int Col, k;
	cfloat_t Awrk[1024];
	unsigned int Row = get_global_id(0);
	for (k = 0; k < M; k++)
	{
		Awrk[k] = X[Row*M + k];
	}
	for (Col = 0; Col < N ; Col++)
	{
		cfloat_t temp = 0;
		for (k=0; k < M ; k++)
		{
			temp = temp + cfloat_mul(Awrk[k], C[k*N + Col]);
		}
		Y[Row*N + Col] = temp;
	}
}
"""

#Size of the buffers to be used must be equal to python Matrix dimensions
c1 = np.zeros_like(py_val)
c2 = np.zeros_like(py_val)
c3 = np.zeros_like(py_val)

#Defining buffers and clearing the memory flags
mf = cl.mem_flags
x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
c1_buf =cl.Buffer(ctx, mf.WRITE_ONLY, c1.nbytes)
c2_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c2.nbytes)
c3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c3.nbytes)


#Building the kernel and passing values to the kernel followed by getting a copy of the output from the GPU
prg = cl.Program(ctx, kernel1).build()
prg.matmul1(queue, (L, N), None, x_buf, c_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
cl.enqueue_copy(queue, c1, c1_buf)
prg = cl.Program(ctx, kernel2).build()
prg.matmul2(queue, (L, N), None, x_buf, c_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
cl.enqueue_copy(queue, c2, c2_buf)
prg = cl.Program(ctx, kernel3).build()
prg.matmul3(queue, (L, N), None, x_buf, c_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
cl.enqueue_copy(queue, c3, c3_buf)


#Printing
print '\nNumpy Matrix Multiplication :  ', py_val
print '\nOpencl Matrix Multiplication basic          :\n ', c1
print '\nOpencl Matrix Multiplication Optimization 1 :\n ', c2
print '\nOpencl Matrix Multiplication Optimization 2 :\n ', c3

#Comparing
print 'PyopenCL matrix multiply basic and python are equal:                 ', np.allclose(py_val, c1, rtol = 1e-02)
print 'PyopenCL matrix multiply Optimization 1 and python are equal:        ', np.allclose(py_val, c2, rtol = 1e-02)
print 'PyopenCL matrix multiply Optimization 2 and python are equal:        ', np.allclose(py_val, c3, rtol = 1e-02)


#Execution time

M=3
times = []
for i in xrange(M):
	start = time.time()
	py_val = np.dot(x,c)
	times.append(time.time()-start)
times_py = np.average(times)
print 'python time:               ', times_py


M=3
times = []
for i in xrange(M):
	start = time.time()
	prg = cl.Program(ctx, kernel1).build()
	prg.matmul1(queue, (L, N), None, x_buf, c_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times1 = np.average(times)
print 'OpenCL Basic time 1 time:  ', times1

M=3
times = []
for i in xrange(M):
        start = time.time()
	prg = cl.Program(ctx, kernel2).build()
        prg.matmul2(queue, (L, N), None, x_buf, c_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
        times.append(time.time()-start)
times2 = np.average(times)
print 'OpenCL Optimization 1 time:', times2

M=3
times = []
for i in xrange(M):
        start = time.time()
        prg = cl.Program(ctx, kernel3).build()
        prg.matmul3(queue, (L, N), None, x_buf, c_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
        times.append(time.time()-start)
times3 = np.average(times)
print 'OpenCL Optimization 2 time:', times3


#Plotting Functions
val = True
def plotfunctions(MAKE_PLOT) :
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt
	if MAKE_PLOT == True :
		







		elements = []
		times = []
		prg=cl.Program(ctx,kernel1).build()
		L = 1
		M = 2
		N = 400
		#Keeping N constant and varying L and M in the ratio of 1:2 for Basic Matrix Multiplication
		while L < N and M < N :
			elements.append(L*M)
			start = time.time()
			prg.matmul1(queue, (L, N), None, x_buf, c_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			times1 = np.average(times)
			
			 
			print "Basic Execution time  L(%d) *M(%d)   *  N(%d)   %f" %(L,M,N,times1)
			L = L + 20
			M = L*2
		plt.figure(1)
		plt.gcf()
		plt.plot(elements, times,'b')
		plt.xlabel('Matrix X for Basic')
		plt.ylabel('Time Taken')
		plt.gca().set_ylim((0.00006, 0.00025))
		plt.savefig('plot-basic1.png')


		elements = []
                times = []
                prg = cl.Program(ctx, kernel1).build()
                L = 400
                M = 1
                N = 2
                #Keeping L constant and varying M and N in the ratio of 1:2 for Basic Matrix Multiplication
                while M < L and N < L :
                        elements.append(N*M)
                        start = time.time()
                        prg.matmul1(queue, (L, N), None, x_buf, c_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
                        times.append(time.time() - start)
			times_c1 = np.average(times)
			print "Basic Execution time  M(%d) * N(%d)   *  L(%d)    %f" %(M,N,L,times_c1)
                        M = M + 20
                        N = M*2
                plt.figure(2)
                plt.gcf()
                plt.plot(elements, times,'b')
                plt.xlabel('Matrix C for Basic')
                plt.ylabel('Time Taken')
                plt.gca().set_ylim((0.00006, 0.00025))
                plt.savefig('plot-basic2.png')



		elements = []
                times = []
                prg = cl.Program(ctx, kernel2).build()
                L = 1
                M = 2
                N = 400
                #Keeping N constant and varying L and M in the ratio of 1:2 for Basic Matrix Multiplication
                while L < N and M < N :
                        elements.append(L*M)
                        start = time.time()
                        prg.matmul2(queue, (L, N), None, x_buf, c_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
                        times.append(time.time() - start)
			time2 = np.average(times)
			
			
			print "Optimization 1 time for  L(%d) * M(%d)   * N(%d)  %f" %(L,M,N,time2)
                        L = L + 20
                        M = L*2
                plt.figure(1)
                plt.gcf()
                plt.plot(elements, times, 'g')
                plt.xlabel('Matrix X for Optimization 1')
                plt.ylabel('  Time Taken  ')
                plt.gca().set_ylim((0.00006, 0.00025))
                plt.savefig('plot-optimized 1.png')


		elements = []
                times = []
                prg = cl.Program(ctx, kernel2).build()
                L = 400
                M = 1
                N = 2
                #Keeping L constant and varying M and N in the ratio of 1:2 for Basic Matrix Multiplication
                while M < L and N < L :
                        elements.append(N*M)
                        start = time.time()
                        prg.matmul2(queue, (L, N), None, x_buf, c_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
                        times.append(time.time() - start)
			times_c2 = np.average(times)
                        print "Optimization 1 time   M(%d) * N(%d)   *  L(%d)    %f" %(M,N,L,times_c2)
                        M = M + 20
                        N = M*2
                plt.figure(2)
                plt.gcf()
                plt.plot(elements, times,'g')
                plt.xlabel('Matrix C for Optimization 1 ')
                plt.ylabel('Time Taken')
                plt.gca().set_ylim((0.00006, 0.00025))
                plt.savefig('plot-optimized_1.png')

		elements = []
                times = []
                L = 1
                M = 2
                N = 400
                #Keeping N constant and varying L and M in the ratio of 1:2 for Basic Matrix Multiplication
                while L < N and M < N :
                        elements.append(L*M)
                        start = time.time()
                        py_val2 = np.dot(x,c)
                        times.append(time.time() - start)
                        times_py1 = np.average(times)
			print "Python Execution time L(%d) * M(%d)  *  N(%d) %f" %(L,M,N,times_py1)
                        L = L + 20
                        M = L*2
                plt.figure(1)
                plt.gcf()
                plt.plot(elements, times,'r')
                plt.xlabel('Matrix X for Python')
                plt.ylabel('Time Taken')
                plt.gca().set_ylim((0.00006, 0.00025))
                plt.savefig('plot-basic1.png')


		elements = []
                times = []
                prg = cl.Program(ctx, kernel3).build()
                L = 1
                M = 2
                N = 400
                #Keeping N constant and varying L and M in the ratio of 1:2 for Basic Matrix Multiplication
                while L < N and M < N :
                        elements.append(L*M)
                        start = time.time()
                        prg.matmul3(queue, (L, N), None, x_buf, c_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
                        times.append(time.time() - start)
                        L = L + 20
                        M = L*2
			time3 = np.average(times)
			
			print "Optimizatoin 2 time for L(%d) * M(%d)  * N(%d)  %f" %(L,M,N,time3)	
	        plt.figure(1)
        	plt.gcf()
                plt.plot(elements, times,'k')
                plt.xlabel('Matrix X for Optimization 2')
                plt.ylabel('  Time Taken  ')
                plt.gca().set_ylim((0.00006, 0.00025))
                plt.savefig('plot-optimized 2.png')

		elements = []
                times = []
                L = 400
                M = 1
                N = 2
                #Keeping L constant and varying M and N in the ratio of 1:2 for Basic Matrix Multiplication
                while M < L and N < L :
                        elements.append(N*M)
                        start = time.time()
                        py_val3 = np.dot(x,c)
                        times.append(time.time() - start)
                        times_py2 = np.average(times)
                        print "Python Execution time M(%d) * N(%d) * L(%d) %f" %(M,N,L,times_py2)
                        M = M + 20
                        N = M*2
                plt.figure(2)
                plt.gcf()
                plt.plot(elements, times,'r')
                plt.xlabel('Matrix C for Python')
                plt.ylabel('Time Taken')
                plt.gca().set_ylim((0.00001, 0.00100))
                plt.savefig('plot-basic2.png')

		elements = []
                times = []
                prg = cl.Program(ctx, kernel3).build()
                L = 400
                M = 1
                N = 2
                #Keeping L constant and varying M and N in the ratio of 1:2 for Basic Matrix Multiplication
                while M < L and N < L :
                        elements.append(N*M)
                        start = time.time()
                        prg.matmul3(queue, (L, N), None, x_buf, c_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
                        times.append(time.time() - start)
			times_c3 = np.average(times)
                        print "Optimization 2 time M %d * N(%d)  * L(%d)  %f" %(M,N,L,times_c3)
                        M = M + 20
                        N = M*2
                plt.figure(2)
                plt.gcf()
                plt.plot(elements, times,'k')
                plt.xlabel('Matrix C for Optimized 2')
                plt.ylabel('Time Taken')
                plt.gca().set_ylim((0.00006, 0.00025))
                plt.savefig('plot-optimized_2.png')


plotfunctions(val)
