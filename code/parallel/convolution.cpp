#include "convolution.h"
#include <iostream>

__global__ void convolveKernel(const float* d_A, const int d_AWidth, const int d_AHeight,
                               const float* d_K, const int d_KWidth, const int d_KHeight, float* d_C){
    int imRow = blockIdx.y * blockDim.y + threadIdx.y;
    int imCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (imRow >= d_AHeight - d_KHeight/2 || imCol >= d_AWidth - d_KHeight/2) return;
    if (imRow <= d_KHeight/2 || imCol <= d_KHeight/2 ) return;

    float newVal = 0.0f;
    for (int kerRow = 0; kerRow < d_KHeight; kerRow++){
        for (int kerCol = 0; kerCol < d_KWidth; kerCol++){
            newVal += d_A[(imRow + (1 - kerRow))*d_AWidth + (imCol + (1 - kerCol))]
                      * d_K[(kerRow)*d_KWidth + (kerCol)];
        }
    }
    d_C[imRow*d_AWidth + imCol] = newVal;
}

void convolve(const float* A, const int AWidth, const int AHeight,
              const float* K, const int KWidth, const int KHeight, float* C){
    float *d_A,
          *d_K,
          *d_C;

    int BLOCK_SIZE = 32;
    
    hipMalloc(&d_A, (AWidth*AHeight)*sizeof(float));
    hipMalloc(&d_K, (KWidth*KHeight)*sizeof(float));
    hipMalloc(&d_C, (AWidth*AHeight)*sizeof(float));

    hipMemcpy(d_A, A, (AWidth*AHeight)*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, (KWidth*KHeight)*sizeof(float), hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(((dimBlock.x - 1) + AWidth) / dimBlock.x, ((dimBlock.y - 1) + AHeight)/ dimBlock.y);

    std::cout << "Launching Convolve Kernel" << std::endl; 
    std::cout << "The Block Dimension is " << dimBlock.x << " X " << dimBlock.y << std::endl; 
    std::cout << "The Grid Dimension is " << dimGrid.x << " X " << dimGrid.y << std::endl; 

    hipEvent_t start, stop; 
    float elapsed_msecs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);

    for (int i = 0; i < 1; i++){
        convolveKernel<<<dimGrid, dimBlock>>>(d_A, AWidth, AHeight, d_K, KWidth, KHeight, d_C);
    }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_msecs, start, stop); 
    std::cout<< "GPU convolve time = " << elapsed_msecs << "ms" << std::endl;

    hipMemcpy(C, d_C, (AWidth*AHeight)*sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_A);
    hipFree(d_K);
    hipFree(d_C);
}

//  __global__ void shConvolveKernel(const float* d_A, const int d_AWidth, const int d_AHeight,
//                                 const float* d_K, const int d_KWidth, const int d_KHeight, float* d_C){
//      extern __shared__ float Ks[];

//     int imRow = blockIdx.y * blockDim.y + threadIdx.y;
//     int imCol = blockIdx.x * blockDim.x + threadIdx.x;

//     if (threadIdx.x == 0 && threadIdx.y == 0){
//         for (int kerRow = 0; kerRow < d_KHeight; kerRow++){
//             for (int kerCol = 0; kerCol < d_KWidth; kerCol++){
//                     Ks[(kerRow)*d_KWidth + (kerCol)] = d_K[(kerRow)*d_KWidth + (kerCol)];
//             }
//         }
//     }
//     __syncthreads();

//     float newVal = 0.0f;
//     for (int kerRow = 0; kerRow < d_KHeight; kerRow++){
//         for (int kerCol = 0; kerCol < d_KWidth; kerCol++){
//             newVal += d_A[(imRow + (1 - kerRow))*d_AWidth + (imCol + (1 - kerCol))]
//                       * Ks[(kerRow)*d_KWidth + (kerCol)];
//         }
//     }
//     d_C[imRow*d_AWidth + imCol] = newVal;
// }

// void shConvolve(const float* A, const int AWidth, const int AHeight,
//               const float* K, const int KWidth, const int KHeight, float* C){
//     float *d_A,
//           *d_K,
//           *d_C;

//     int BLOCK_SIZE = 32;
    
//     hipMalloc(&d_A, (AWidth*AHeight)*sizeof(float));
//     hipMalloc(&d_K, (KWidth*KHeight)*sizeof(float));
//     hipMalloc(&d_C, (AWidth*AHeight)*sizeof(float));

//     hipMemcpy(d_A, A, (AWidth*AHeight)*sizeof(float), hipMemcpyHostToDevice);
//     hipMemcpy(d_K, K, (KWidth*KHeight)*sizeof(float), hipMemcpyHostToDevice);

//     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 dimGrid(((dimBlock.x - 1) + AWidth) / dimBlock.x, ((dimBlock.y - 1) + AHeight)/ dimBlock.y);

//     std::cout << "Launching Convolve Kernel" << std::endl; 
//     std::cout << "The Block Dimension is " << dimBlock.x << " X " << dimBlock.y << std::endl; 
//     std::cout << "The Grid Dimension is " << dimGrid.x << " X " << dimGrid.y << std::endl; 

//     hipEvent_t start, stop; 
//     float elapsed_msecs; 
//     hipEventCreate(&start); 
//     hipEventCreate(&stop); 
//     hipEventRecord(start, 0);

//     for (int i = 0; i < 100000; i++){
//         shConvolveKernel<<<dimGrid, dimBlock, (KWidth * KHeight) * sizeof(float)>>>(d_A, AWidth, AHeight, d_K, KWidth, KHeight, d_C);
//     }

//     hipEventRecord(stop, 0); 
//     hipEventSynchronize(stop); 
//     hipEventElapsedTime(&elapsed_msecs, start, stop); 
//     std::cout<< "GPU shared convolve time = " << elapsed_msecs << "ms" << std::endl;

//     hipMemcpy(C, d_C, (AWidth*AHeight)*sizeof(float), hipMemcpyDeviceToHost);
    
//     hipFree(d_A);
//     hipFree(d_K);
//     hipFree(d_C);
// }
