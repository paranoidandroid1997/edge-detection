#include <iostream>

#include "compare.h"

__global__ void neighbourCheckKernel(float d_finalPixels[], float d_newMagnitudes[], int d_evals[], int imHeight, int imWidth){
    int imRow = blockIdx.y * blockDim.y + threadIdx.y;
    int imCol = blockIdx.x * blockDim.x + threadIdx.x;

    if(d_evals[imRow*imWidth + imCol]==1){
        if(  (d_evals[(imRow - 1)*imWidth + (imCol - 1)] == 2)
            ||(d_evals[(imRow - 1)*imWidth + (imCol)] == 2)
            ||(d_evals[(imRow - 1)*imWidth + (imCol + 1)] == 2)
            ||(d_evals[(imRow)*imWidth + (imCol - 1)] == 2)
            ||(d_evals[(imRow)*imWidth + (imCol + 1)] == 2)
            ||(d_evals[(imRow + 1)*imWidth + (imCol - 1)] == 2)
            ||(d_evals[(imRow + 1)*imWidth + (imCol)] == 2)
            ||(d_evals[(imRow + 1)*imWidth + (imCol+1)] == 2)
            ){
                d_finalPixels[imRow*imWidth + imCol] = 255;
            }
        else{
            d_finalPixels[imRow*imWidth + imCol] = 0;
        }
    }
    else if (d_evals[imRow*imWidth + imCol] == 2){
            d_finalPixels[imRow*imWidth + imCol] = 255; 
    }
}

void neighbourCheck(float finalPixels[], float newMagnitudes[], int evals[],  int imHeight, int imWidth){
    float *d_finalPixels,
          *d_newMagnitudes;
    int *d_evals;

    int BLOCK_SIZE = 32;
    
    hipMalloc(&d_finalPixels, (imWidth * imHeight)*sizeof(float));
    hipMalloc(&d_newMagnitudes, (imWidth * imHeight)*sizeof(float));
    hipMalloc(&d_evals, (imWidth * imHeight)*sizeof(int));

    hipMemcpy(d_newMagnitudes, newMagnitudes, (imWidth * imHeight)*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_evals, evals, (imWidth * imHeight)*sizeof(int), hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(((dimBlock.x - 1) + imWidth) / dimBlock.x, ((dimBlock.y - 1) + imHeight)/ dimBlock.y);

    std::cout << "Launching Neighbour Check Kernel" << std::endl; 
    std::cout << "The Block Dimension is " << dimBlock.x << " X " << dimBlock.y << std::endl; 
    std::cout << "The Grid Dimension is " << dimGrid.x << " X " << dimGrid.y << std::endl;  

    hipEvent_t start, stop; 
    float elapsed_msecs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);

    for (int i = 0; i < 1; i++){
        neighbourCheckKernel<<<dimGrid, dimBlock>>>(d_finalPixels, d_newMagnitudes, d_evals, imHeight, imWidth); 
    }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_msecs, start, stop); 
    std::cout<< "GPU neighbour check time = " << elapsed_msecs << "ms" << std::endl;

    hipMemcpy(finalPixels, d_finalPixels, (imWidth * imHeight)*sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_finalPixels);
    hipFree(d_newMagnitudes);
    hipFree(d_evals);
}