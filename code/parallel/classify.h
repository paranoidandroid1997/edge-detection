#ifndef CLASSIFY_H
#define CLASSIFY_H

#include <hip/hip_runtime.h>

__global__ void classifyKernel(int*, float*, float, float, int);
void classify(int*, float*, float, float, int);

#endif