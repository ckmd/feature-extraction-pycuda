import pycuda.autoinit
import pycuda.driver as drv
import numpy, time

# __global__ void conv33(float *r33r, float *r33i, float *r17r, float *r17i, float *r9r, float *r9i, float *r5r, float *r5i, float *a33, float *a17, float *a9, float *a5, float *f33r, float *f33i, float *f17r, float *f17i, float *f9r, float *f9i, float *f5r, float *f5i)
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void conv33(float *r33r, float *r33i, float *a33, float *f33r, float *f33i)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  int Idx = i + j * blockDim.x * gridDim.x;
  r33r[Idx] = a33[Idx] * f33r[Idx];
  r33i[Idx] = a33[Idx] * f33i[Idx];
}

__global__ void conv17(float *r17r, float *r17i, float *a17, float *f17r, float *f17i)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  int Idx = i + j * blockDim.x * gridDim.x;
  r17r[Idx] = a17[Idx] * f17r[Idx];
  r17i[Idx] = a17[Idx] * f17i[Idx];
}

__global__ void conv9(float *r9r, float *r9i, float *a9, float *f9r, float *f9i)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  int Idx = i + j * blockDim.x * gridDim.x;
  r9r[Idx] = a9[Idx] * f9r[Idx];
  r9i[Idx] = a9[Idx] * f9i[Idx];
//  r5r[Idx] = a5[Idx] * f5r[Idx];
//  r5i[Idx] = a5[Idx] * f5i[Idx];
}

__global__ void conv5(float *r5r, float *r5i, float *a5, float *f5r, float *f5i)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  int Idx = i + j * blockDim.x * gridDim.x;
  r5r[Idx] = a5[Idx] * f5r[Idx];
  r5i[Idx] = a5[Idx] * f5i[Idx];
}
""")

# a = image, f = filter, r = result
a33 = numpy.ones(296208).astype(numpy.float32)
a17 = numpy.ones(78608).astype(numpy.float32)
a9 = numpy.ones(22032).astype(numpy.float32)
a5 = numpy.ones(6800).astype(numpy.float32)

f33r = a33 * 3.31
f33i = a33 * 3.32
f17r = a17 * 1.71
f17i = a17 * 1.72
f9r = a9 * 0.91
f9i = a9 * 0.92
f5r = a5 * 0.51
f5i = a5 * 0.52

r33r = numpy.zeros_like(a33)
r33i = numpy.zeros_like(a33)
r17r = numpy.zeros_like(a17)
r17i = numpy.zeros_like(a17)
r9r = numpy.zeros_like(a9)
r9i = numpy.zeros_like(a9)
r5r = numpy.zeros_like(a5)
r5i = numpy.zeros_like(a5)

waktu = []
# while True:
for lol in range(1000):
  start = time.time()
  # max thread per block is 1024, and max block per grid is 304, so be careful
  # calculating parallel using GPU
  conv33 = mod.get_function("conv33")
  conv17 = mod.get_function("conv17")
  conv9 = mod.get_function("conv9")
  conv5 = mod.get_function("conv5")
  conv33(drv.Out(r33r), drv.Out(r33i), drv.In(a33), drv.In(f33r), drv.In(f33i), block=(68,4,1), grid=(33,33))
  conv17(drv.Out(r17r), drv.Out(r17i), drv.In(a17), drv.In(f17r), drv.In(f17i), block=(68,4,1), grid=(17,17))
  conv9(drv.Out(r9r), drv.Out(r9i), drv.In(a9), drv.In(f9r), drv.In(f9i), block=(68,4,1), grid=(9,9))
  conv5(drv.Out(r5r), drv.Out(r5i), drv.In(a5), drv.In(f5r), drv.In(f5i), block=(68,4,1), grid=(5,5))
  # Using Numpy
  # r33r = a33 * f33r
  # r33r = a33 * f33r
  # r17r = a17 * f17r
  # r17r = a17 * f17r
  # r9r = a9 * f9r
  # r9r = a9 * f9r
  # r5r = a5 * f5r
  # r5r = a5 * f5r
  end = time.time()
  waktu.append(end-start)
  print(end - start)
  # print(r33r, len(r33r), sum(r33r))
  # print(r33i, len(r33i), sum(r33i))
  # print(r17r, len(r17r), sum(r17r))
  # print(r17i, len(r17i), sum(r17i))
  # print(r9r, len(r9r), sum(r9r))
  # print(r9i, len(r9i), sum(r9i))
  # print(r5r, len(r5r), sum(r5r))
  # print(r5i, len(r5i), sum(r5i))

print("average time in second : ",sum(waktu)/len(waktu))