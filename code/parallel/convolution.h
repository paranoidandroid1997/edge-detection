#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <hip/hip_runtime.h>

__global__ void convolveKernel();
void convolve(const float*, const int, const int, const float*, const int, const int, float*);

#endif