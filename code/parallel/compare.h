#ifndef COMPARE_H
#define COMPARE_H

#include <hip/hip_runtime.h>

__global__ void compareKernel(float*, float*, float*, int, int);
void compare(float*, float*, float*, int, int);

#endif