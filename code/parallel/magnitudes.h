#ifndef MAGNITUDES_H
#define MAGNITUDES_H

#include <hip/hip_runtime.h>

__global__ void findMagnitudesKernel(float*, float*, float*, int);
void findMagnitudes(float*, float*, float*, int);

#endif