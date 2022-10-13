#include <iostream>

#include "magnitudes.h"

__global__ void findMagnitudesKernel(float d_Gx[], float d_Gy[], float d_magnitudes[], int imSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= imSize) return;

    d_magnitudes[tid] = sqrt(d_Gx[tid] * d_Gx[tid] + d_Gy[tid] * d_Gy[tid]);
}

void findMagnitudes(float Gx[], float Gy[], float magnitudes[], int imSize){
    float *d_Gx,
          *d_Gy,
          *d_magnitudes;

    int BLOCK_SIZE = 1024;
    
    hipMalloc(&d_Gx, (imSize)*sizeof(float));
    hipMalloc(&d_Gy, (imSize)*sizeof(float));
    hipMalloc(&d_magnitudes, (imSize)*sizeof(float));

    hipMemcpy(d_Gx, Gx, (imSize)*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_Gy, Gy, (imSize)*sizeof(float), hipMemcpyHostToDevice);


    std::cout << "Launching Magnitudes Kernel" << std::endl; 
    std::cout << "The Block Size is " << BLOCK_SIZE << std::endl; 
    std::cout << "The Grid Dimension is " << ((BLOCK_SIZE - 1) + imSize)/ BLOCK_SIZE << std::endl; 

    hipEvent_t start, stop; 
    float elapsed_msecs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);

    for (int i = 0; i < 10000; i++){
        findMagnitudesKernel<<<((BLOCK_SIZE - 1) + imSize)/ BLOCK_SIZE, BLOCK_SIZE>>>(d_Gx, d_Gy, d_magnitudes, imSize);
    }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_msecs, start, stop); 
    std::cout<< "GPU magnitudes time = " << elapsed_msecs << "ms" << std::endl;

    hipMemcpy(magnitudes, d_magnitudes, (imSize)*sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_Gx);
    hipFree(d_Gy);
    hipFree(d_magnitudes);
}