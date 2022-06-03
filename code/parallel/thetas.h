#ifndef THETAS_H
#define THETAS_H

#include <hip/hip_runtime.h>

__global__ void findThetasKernel(float*, float*, float*, int);
void findThetas(float*, float*, float*, int);

#endif