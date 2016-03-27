import time
 
import pyopencl as cl
import pyopencl.array
import numpy as np
#include <pyopencl-complex.h> 
# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
 
# Set up a command queue; we need to enable profiling to time GPU operations:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
N=3
x= np.random.randint(1,9, (3,3)).astype(np.uint32);

print x
y= np.transpose(x)

print y
kernel = """
kernel void trans(__global int* c,__global int* x) {
	unsigned int Row = get_global_id(0);
	unsigned int Col = get_global_id(1);
	
	

 	
	c[Row*3+Col]=x[Col*3+Row];
		
	
	
	
	
}
"""


c_gpu = cl.array.zeros(queue,x.shape,x.dtype)

x_gpu = cl.array.to_device(queue, x)
 
prg = cl.Program(ctx, kernel).build()
prg.trans(queue , x.shape, None , c_gpu.data, x_gpu.data)
print 'original',c_gpu
c= c_gpu.get()

print 'input x\n', x

print 'transpose c\n', c
