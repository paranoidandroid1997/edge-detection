#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <hip/hip_runtime.h>

__global__ void shConvolveKernel(const float*, const int, const int, const float*, const int, const int, float*);
__global__ void convolveKernel(const float*, const int, const int, const float*, const int, const int, float*);
void convolve(const float*, const int, const int, const float*, const int, const int, float*);
void shConvolve(const float*, const int, const int, const float*, const int, const int, float*);

#endif