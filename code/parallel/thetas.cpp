#include <iostream>

#include "thetas.h"

#define pi 3.14159

__global__ void findThetasKernel(float d_Gx[], float d_Gy[], float d_thetas[]){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = atan2(d_Gy[tid], d_Gx[tid]);
    if (           (temp <= pi/8 && temp > -pi/8)
                || (temp >= -pi && temp <= -7*pi/8)
                || (temp <= pi && temp > 7*pi/8)){d_thetas[tid] = 0.0f;}
    else if (      (temp <= 3*pi/8 && temp > pi/8)
                || (temp > -3*pi/8 && temp <= -pi/8)){d_thetas[tid] = 45.0f;}
    else if (      (temp <= 5*pi/8 && temp > 3*pi/8)
                || (temp > -5*pi/8 && temp <= -3*pi/8) ){d_thetas[tid] = 90.0f;}
    else if (      (temp <= 7*pi/8 && temp > 5*pi/8)
                || (temp > -7*pi/8 && temp <= -5*pi/8)){d_thetas[tid] = 135.0f;}
    else {d_thetas[tid] = 0;} // For some reason 3.14159 is not counted
}

void findThetas(float Gx[], float Gy[], float thetas[], int imSize){
    float *d_Gx,
          *d_Gy,
          *d_thetas;

    int BLOCK_SIZE = 1024;
    
    hipMalloc(&d_Gx, (imSize)*sizeof(float));
    hipMalloc(&d_Gy, (imSize)*sizeof(float));
    hipMalloc(&d_thetas, (imSize)*sizeof(float));

    hipMemcpy(d_Gx, Gx, (imSize)*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_Gy, Gy, (imSize)*sizeof(float), hipMemcpyHostToDevice);


    std::cout << "Launching Thetas Kernel" << std::endl; 
    std::cout << "The Block Size is " << BLOCK_SIZE << std::endl; 
    std::cout << "The Grid Dimension is " << ((BLOCK_SIZE - 1) + imSize)/ BLOCK_SIZE << std::endl; 

    hipEvent_t start, stop; 
    float elapsed_msecs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);

    for (int i = 0; i < 1; i++){
        findThetasKernel<<<((BLOCK_SIZE - 1) + imSize)/ BLOCK_SIZE, BLOCK_SIZE>>>(d_Gx, d_Gy, d_thetas);
    }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_msecs, start, stop); 
    std::cout<< "GPU thetas time = " << elapsed_msecs << "ms" << std::endl;

    hipMemcpy(thetas, d_thetas, (imSize)*sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_Gx);
    hipFree(d_Gy);
    hipFree(d_thetas);
}
