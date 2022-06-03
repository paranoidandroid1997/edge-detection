#include <iostream>

#include "compare.h"

__global__ void compareKernel(float d_newMagnitudes[], float d_magnitudes[], float d_thetas[], int imHeight, int imWidth){
    int imRow = blockIdx.y * blockDim.y + threadIdx.y;
    int imCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (imRow >= imHeight || imCol >= imWidth) return;

    if (d_thetas[imRow*imWidth + imCol] == 0){
        if (   (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow)*imWidth + (imCol + 1)])
            && (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow)*imWidth + (imCol - 1)])){
                d_newMagnitudes[imRow*imWidth + imCol] = d_magnitudes[imRow*imWidth + imCol];
            }
        else {
            d_newMagnitudes[imRow*imWidth + imCol] = 0;
        }
    }
    else if (d_thetas[imRow*imWidth + imCol] == 45){
        if (   (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow + 1)*imWidth + (imCol + 1)])
            && (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow - 1)*imWidth + (imCol - 1)])){
                d_newMagnitudes[imRow*imWidth + imCol] = d_magnitudes[imRow*imWidth + imCol];
            }
        else {
            d_newMagnitudes[imRow*imWidth + imCol] = 0;
        }

    }
    else if (d_thetas[imRow*imWidth + imCol] == 90){
        if (   (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow + 1)*imWidth + (imCol)])
            && (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow - 1)*imWidth + (imCol)])){
                d_newMagnitudes[imRow*imWidth + imCol] = d_magnitudes[imRow*imWidth + imCol];
            }
        else {
            d_newMagnitudes[imRow*imWidth + imCol] = 0;
        }     
    }
    else if (d_thetas[imRow*imWidth + imCol] == 135){
        if (   (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow - 1)*imWidth + (imCol + 1)])
            && (d_magnitudes[imRow*imWidth + imCol] > d_magnitudes[(imRow + 1)*imWidth + (imCol - 1)])){
            d_newMagnitudes[imRow*imWidth + imCol] = d_magnitudes[imRow*imWidth + imCol];
            }
        else {
            d_newMagnitudes[imRow*imWidth + imCol] = 0;
        }
        
    }
}

void compare(float newMagnitudes[], float magnitudes[], float thetas[],  int imHeight, int imWidth){
    float *d_newMagnitudes,
          *d_magnitudes,
          *d_thetas;

    int BLOCK_SIZE = 32;
    
    hipMalloc(&d_newMagnitudes, (imWidth * imHeight)*sizeof(float));
    hipMalloc(&d_magnitudes, (imWidth * imHeight)*sizeof(float));
    hipMalloc(&d_thetas, (imWidth * imHeight)*sizeof(float));

    hipMemcpy(d_magnitudes, magnitudes, (imWidth * imHeight)*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_thetas, thetas, (imWidth * imHeight)*sizeof(float), hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(((dimBlock.x - 1) + imWidth) / dimBlock.x, ((dimBlock.y - 1) + imHeight)/ dimBlock.y);

    std::cout << "Launching Compare Kernel" << std::endl; 
    std::cout << "The Block Dimension is " << dimBlock.x << " X " << dimBlock.y << std::endl; 
    std::cout << "The Grid Dimension is " << dimGrid.x << " X " << dimGrid.y << std::endl;  

    hipEvent_t start, stop; 
    float elapsed_msecs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);

    for (int i = 0; i < 1; i++){
        compareKernel<<<dimGrid, dimBlock>>>(d_newMagnitudes, d_magnitudes, d_thetas, imHeight, imWidth); 
    }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_msecs, start, stop); 
    std::cout<< "GPU compare time = " << elapsed_msecs << "ms" << std::endl;

    hipMemcpy(newMagnitudes, d_newMagnitudes, (imWidth * imHeight)*sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_newMagnitudes);
    hipFree(d_magnitudes);
    hipFree(d_thetas);
}