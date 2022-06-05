#include "convolution.h"

void convolve(const float* A, const int AWidth, const int AHeight,
              const float* K, const int KWidth, const int KHeight, float* C){

    for (int imRow = KHeight/2; imRow < AHeight - KHeight/2; imRow++){
        for (int imCol = KHeight/2; imCol < AWidth - KHeight/2; imCol++){
            float newVal = 0.0f;
            for (int kerRow = 0; kerRow < KHeight; kerRow++){
                for (int kerCol = 0; kerCol < KWidth; kerCol++){
                    newVal += A[(imRow + (1 - kerRow))*AWidth + (imCol + (1 - kerCol))] * K[(kerRow)*KWidth + (kerCol)];
                }
            }
            C[imRow*AWidth + imCol] = newVal;
        }
    }
}