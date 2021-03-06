import csv
import time
import pyopencl as cl
import numpy as np
import numpy.matlib

# Selecting OpenCL platform.
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
	if platform.name == NAME:
		devs = platform.get_devices()

# Setting Command Queue.
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

blocks = 25
workgroup = 25
range_taken = 10

#the transpose kernel, taking A row in private memory in Dwrk, did not include basic transpose as it would not give a faster time.
kernel_1 = """
__kernel void transpose1(__global float* D, __global float* output, const int P, const int Q) {

        unsigned int row = get_global_id(0);
        float Dwrk[1024];
        for (int i=0; i<Q ; i++)
        {
        Dwrk[i]= D[row*Q + i];
        }
        for (int j= 0 ; j< Q ; j++)
        {
        output[j*P + row] = Dwrk[j];
        }

}
"""

#taking an entire row in private memory, and also taking a entire column of the matrices B and C  in local mem(shared memory) and adding it
#We further multiply this result of a Column of B and C with the row obtained from Matrix A. Then we store the resulting value into the output. which is our matrix where the
#result is stored.
kernel_2 = """
__kernel void algo1(__global float* A, __global float* B, __global float* C, __global float* output, const int L, const int M, const int N) {
	
	float Awrk[1024]; // We define private memory
	unsigned int row = get_global_id(0); 
	unsigned int icol = get_local_id(0);
	unsigned int nCol = get_local_size(0); 
	unsigned int col, k;
	__local float Bloc[1024];	
	

	for (int k = 0; k < M; k++)
	{
		Awrk[k] = A[row*M + k];
	}
	for (int col = 0; col < N ; col++)
	{	
		
		for (int k = icol; k < M; k += nCol)
		{
			Bloc[k] = B[k*N + col] + C[k*N + col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		float temp = 0;
		for (int k=0; k < M ; k++)
		{
			temp = temp + Awrk[k]*Bloc[k];
		}
		output[row*N + col] = temp;
	}
}
"""


#Matrix multiplication using Tiling, Here we define the Tile_Width similar to the block size which must be  multiples of the input matrices. This way we do not have to worry about padding
#We first load the tiles into the local memory using variables A_tile and B_tile. We add the column of C similar to the above example and store the result in B_tile.
#The multiplication result is stored in the output matrix.
kernel_3 = """
#define TILE_WIDTH 25
__kernel void algo2(__global float* A, __global float* B, __global float* C, __global float* output, const int L, const int M, const int N) {


		unsigned int tx = get_local_id(0);
		unsigned int bx = get_group_id(0);
		unsigned int ty = get_local_id(1); 
		unsigned int by = get_group_id(1); 
		__local float A_tile[TILE_WIDTH][TILE_WIDTH];
		__local float B_tile[TILE_WIDTH][TILE_WIDTH];
		
		unsigned int row = by*get_local_size(1) + ty;
		unsigned int col = bx*get_local_size(0) + tx;
		float Cwrk = 0;
		for (int k = 0; k < M/TILE_WIDTH; k++)
		{
			A_tile[ty][tx] = A[row*M + k*TILE_WIDTH + tx];
			B_tile[ty][tx] = B[(k*TILE_WIDTH + ty)*N + col] + C[(k*TILE_WIDTH + ty)*N + col];
			barrier(CLK_LOCAL_MEM_FENCE);
			for (int j = 0; j < TILE_WIDTH; j++)
			{
				Cwrk = Cwrk + A_tile[ty][j]*B_tile[j][tx];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		output[row*N + col] = Cwrk;
}
"""



d = 1
while d < 10:
	L= 100*d
	M = 100*d
	N = 100*d
	d = d + 1
	P = 400
	Q= 10
#L = 300
#M = 300
#N = 300
#P = 400
#Q = 10

# Generate random numbers for matrix values
	a = np.random.randint(0, range_taken, size = (L,M)).astype(np.float32)
	b = np.random.randint(0, range_taken, size = (M,N)).astype(np.float32)
	c = np.random.randint(0, range_taken, size = (M,N)).astype(np.float32)
	d = np.random.randint(0, range_taken, size = (P,Q)).astype(np.float32)
#print a
#print b
#print c
#print d


	py_val = np.dot(a,(b+c))

	py_val1 = np.transpose(d)

# Store for both methods 1 and 2 , also for the trasnpose.
	out1 = np.zeros_like(py_val)
	out2 = np.zeros_like(py_val)
#just passing out3=np.zeros_like(py_val1) does not work. Maybe a type casting issue or maybe with the way matrices are treated by numpy
	out3 = np.zeros(shape=(Q,P)).astype(np.float32)

#Defining buffers and clearing the memory flags
	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
	d_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
	out1_buf =cl.Buffer(ctx, mf.WRITE_ONLY, out1.nbytes)
	out2_buf =cl.Buffer(ctx, mf.WRITE_ONLY, out2.nbytes)
	out3_buf =cl.Buffer(ctx, mf.WRITE_ONLY, out3.nbytes)

# Kernel build ,calling the appropriate kernel function followed by getting the values from the GPU
	prg = cl.Program(ctx, kernel_2).build()
	prg.algo1(queue, (L, ), (L/workgroup, ), a_buf, b_buf, c_buf, out1_buf, np.int32(L), np.int32(M), np.int32(N))
	cl.enqueue_copy(queue, out1, out1_buf)
	prg = cl.Program(ctx, kernel_3).build()
	prg.algo2(queue, (L,N), (blocks,blocks), a_buf, b_buf, c_buf, out2_buf, np.int32(L), np.int32(M), np.int32(N))	
	cl.enqueue_copy(queue, out2, out2_buf)
	prg = cl.Program(ctx, kernel_1).build()
	prg.transpose1(queue,d.shape,None, d_buf,  out3_buf, np.int32(P), np.int32(Q))
	cl.enqueue_copy(queue, out3, out3_buf)


	# Print values of all  matrices and compare
	print '\nNumpy Matrix Multiplication :  ', py_val
	print '\nNumpy Matrix Transpose :\n       ', py_val1
	print '\nOpencl Matrix Multiplication Algorithm 1 :', out1
	print '\nOpencl Matrix Multiplication Algorithm 2 :', out2
	print '\nOpencl Matrix Transpose  Algorithm 1\n     :', out3

	# Compare the calculated values for algorithms and numpy
	print 'OpenCL matrix multiply optimization  algorithm 1 and numpy are equal:        ', np.allclose(py_val, out1)
	print 'OpenCL matrix multiply optimization  algorithm 2 and numpy are equal:        ', np.allclose(py_val, out2)
	print 'OpenCL matrix transpose optimization algorithm 1 and numpy are equal:        ', np.allclose(py_val1, out3)


	M = 3

	# Measure time taken by Python 
	times = []
	for i in xrange(M):
		start = time.time()
		py_val = np.dot(a,b)
		times.append(time.time()-start)
	times_py = np.average(times)
	print 'Python Time for the Operation A * (B + C):  ', times_py


	times = []
	for i in xrange(M):
        	start = time.time()
        	py_val1 = np.transpose(b)
        	times.append(time.time()-start)
	times_py1 = np.average(times)
	print 'Python Time for Transpose Operation :      ', times_py1

	# Measure time taken by Algorithm 1
	prg = cl.Program(ctx, kernel_2).build()
	times = []
	for i in xrange(M):
		start = time.time()
		prg.algo1(queue, (L, ), (L/workgroup, ), a_buf, b_buf, c_buf, out1_buf, np.int32(L), np.int32(M), np.int32(N))
		times.append(time.time()-start)
	times1 = np.average(times)
	print 'OpenCL Algorithm-1 time:  ', times1

	# Measure time taken by Algorithm 2
	prg = cl.Program(ctx, kernel_3).build()
	times = []
	for i in xrange(M):
		start = time.time()
		prg.algo2(queue, (L,N), (blocks,blocks), a_buf, b_buf, c_buf, out2_buf, np.int32(L), np.int32(M), np.int32(N))
		times.append(time.time()-start)
	times2 = np.average(times)
	print 'OpenCL Algorithm-2 time:  ', times2

	#Measure time taken by Transpose Algorithm 1

	prg = cl.Program(ctx, kernel_1).build()
	times = []
	for i in xrange(M):
        	start = time.time()
		prg.transpose1(queue, d.shape,None, d_buf,  out3_buf, np.int32(P), np.int32(Q))
        	times.append(time.time()-start)
	times3 = np.average(times)
	print 'OpenCL Transpose-Algorithm time: ', times3

	#Plotting Functions

	#Plotted in Excel
