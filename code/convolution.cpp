#include "convolution.h"

void convolve(float* A, const int AWidth, const int AHeight,
              const float* K, const int KWidth, const int KHeight){

    float* C = new float[AWidth * AHeight];
    for (int imRow = 1; imRow < AHeight - 1; imRow++){
        for (int imCol = 1; imCol < AWidth - 1; imCol++){
            float newVal = 0.0f;

            for (int kerRow = 0; kerRow < KHeight; kerRow++){
                for (int KerCol = 0; kerCol < KWidth; kerCol++){
                    newVal += A[imRow*AWidth + imCol] * K[kerRow *KWidth + kerCol];
                }
            }
        }
        C[imRow*AWidth + imCol] = newVal;
    }

    delete A;
    A = C;
}