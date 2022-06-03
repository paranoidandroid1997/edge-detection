#include <iostream>

#include "classify.h"

__global__ void classifyKernel(int d_evals[], float d_magnitudes[], float highThresh, float lowThresh, int imSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= imSize) return;

    if (d_magnitudes[tid] < highThresh && d_magnitudes[tid] > lowThresh){
        d_evals[tid] = 1;
    }
    else if (d_magnitudes[tid] < lowThresh){
        d_evals[tid] = 0;
        d_magnitudes[tid] = 0;
    }
    else {
        d_evals[tid] = 2;
    }
}

void classify(int evals[], float magnitudes[], float highThresh, float lowThresh, int imSize){
    int *d_evals;
    float *d_magnitudes;

    int BLOCK_SIZE = 1024;
    
    hipMalloc(&d_evals, (imSize)*sizeof(int));
    hipMalloc(&d_magnitudes, (imSize)*sizeof(float));

    hipMemcpy(d_magnitudes, magnitudes, (imSize)*sizeof(float), hipMemcpyHostToDevice);


    std::cout << "Launching Classify Kernel" << std::endl; 
    std::cout << "The Block Size is " << BLOCK_SIZE << std::endl; 
    std::cout << "The Grid Dimension is " << ((BLOCK_SIZE - 1) + imSize)/ BLOCK_SIZE << std::endl; 

    hipEvent_t start, stop; 
    float elapsed_msecs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);

    for (int i = 0; i < 1; i++){
        classifyKernel<<<((BLOCK_SIZE - 1) + imSize)/ BLOCK_SIZE, BLOCK_SIZE>>>(d_evals, d_magnitudes, highThresh, lowThresh, imSize); 
    }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_msecs, start, stop); 
    std::cout<< "GPU classify time = " << elapsed_msecs << "ms" << std::endl;

    hipMemcpy(evals, d_evals, (imSize)*sizeof(int), hipMemcpyDeviceToHost);
    
    hipFree(d_evals);
    hipFree(d_magnitudes);
}