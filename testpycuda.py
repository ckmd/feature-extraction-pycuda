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
conv33 = mod.get_function("conv33")
conv17 = mod.get_function("conv17")
conv9 = mod.get_function("conv9")
conv5 = mod.get_function("conv5")

# while True:
for lol in range(100):
  start = time.time()
  # max thread per block is 1024, and max block per grid is 304, so be careful
  # calculating parallel using GPU
  conv33(drv.Out(r33r), drv.Out(r33i), drv.In(a33), drv.In(f33r), drv.In(f33i), block=(68,4,1), grid=(33,33))
  conv17(drv.Out(r17r), drv.Out(r17i), drv.In(a17), drv.In(f17r), drv.In(f17i), block=(68,4,1), grid=(17,17))
  conv9(drv.Out(r9r), drv.Out(r9i), drv.In(a9), drv.In(f9r), drv.In(f9i), block=(68,4,1), grid=(9,9))
  conv5(drv.Out(r5r), drv.Out(r5i), drv.In(a5), drv.In(f5r), drv.In(f5i), block=(68,4,1), grid=(5,5))

  splr33r = numpy.sum(numpy.split(r33r,272),axis = 1)
  splr33i = numpy.sum(numpy.split(r33i,272),axis = 1)
  splr17r = numpy.sum(numpy.split(r17r,272),axis = 1)
  splr17i = numpy.sum(numpy.split(r17i,272),axis = 1)
  splr9r = numpy.sum(numpy.split(r9r,272),axis = 1)
  splr9i = numpy.sum(numpy.split(r9i,272),axis = 1)
  splr5r = numpy.sum(numpy.split(r5r,272),axis = 1)
  splr5i = numpy.sum(numpy.split(r5i,272),axis = 1)

  mag33 = numpy.sqrt(splr33r**2 + splr33i**2)
  mag17 = numpy.sqrt(splr17r**2 + splr17i**2)
  mag9 = numpy.sqrt(splr9r**2 + splr9i**2)
  mag5 = numpy.sqrt(splr5r**2 + splr5i**2)

  phase33 = numpy.arctan(splr33i / splr33r)
  phase17 = numpy.arctan(splr17i / splr17r)
  phase9 = numpy.arctan(splr9i / splr9r)
  phase5 = numpy.arctan(splr5i / splr5r)

  # Normalisasi ke 0 dan 1 sebelum masuk ke NN

  end = time.time()
  waktu.append(end-start)
  print(end - start)
  print(mag33)

print("average time in second : ",sum(waktu)/len(waktu))