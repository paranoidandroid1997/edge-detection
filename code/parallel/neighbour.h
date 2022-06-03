#ifndef NEIGHBOUR_H
#define NEIGHBOUR_H

#include <hip/hip_runtime.h>

__global__ void neighbourCheckKernel(float*, float*, int*, int, int);
void neighbourCheck(float*, float*, int*, int, int);

#endif