#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>

#include "CImg.h"
#include "convolution.h"
using namespace cimg_library;

#include "convolution.h"

#define pi 3.14159

int main(){
    // Load in the image
    CImg<float> image("../images/input/test-image-4.pgm");

    // Get useful data about the image
    int imWidth = image.width();
    int imHeight = image.height();
    int imSpectrum = image.spectrum();
    int imSize = image.size();

    // Print out this useful data for debugging
    std::cout << "Dimension of Image: " << '(' << imWidth  << " X " << imHeight << ')' << std::endl;
    std::cout << "Number of Channels: " << imSpectrum << std::endl;
    std::cout << "Image Size: " << imSize << std::endl;

    // Make an array that holds the pixel data to perform computations on
    float* rawData = new float[imSize];
    for (int row = 0; row < imHeight; row++){
        for (int col = 0; col < imWidth; col++){
            rawData[row*imWidth + col] = image(col, row); 
        }
    }

    // Define a 3X3 gaussian kerenl
    float gaussianKernel[25] = {2.0f,4.0f,5.0f,4.0f,2.0f,
                                4.0f,9.0f,12.0f,9.0f,4.0f,
                                5.0f,12.0f,15.0f,12.0f,5.0f,
                                4.0f,9.0f,12.0f,9.0f,4.0f,
                                2.0f,4.0f,5.0f,4.0f,2.0f};

    for (int i = 0; i < 25; i++){
        gaussianKernel[i] *= 1.0f/159.0f;
    }

    // Blur the image and store that in blurredData and then delete the unblurred data
    float* blurredData = new float[imWidth * imHeight];
    convolve(rawData, imWidth, imHeight, gaussianKernel, 5, 5, blurredData);
    delete rawData;

    // Define sobel operators to find the gradients in the X and Y directions
    float sobelGx[9] = {1.0f, 0.0f, -1.0f,
                         2.0f, 0.0f, -2.0f,
                         1.0f, 0.0f, -1.0f};
    
    float sobelGy[9] = {1.0f, 2.0f, 1.0f,
                         0.0f, 0.0f, 0.0f,
                         -1.0f, -2.0f, -1.0f};
    

    // Apply sobel operators and store the values
    float* Gx = new float[imWidth * imHeight];
    float* Gy = new float[imWidth * imHeight];
    convolve(blurredData, imWidth, imHeight, sobelGx, 3, 3, Gx);
    convolve(blurredData, imWidth, imHeight, sobelGy, 3, 3, Gy);

    // Use the derivatives in the x and y direction to compute the magnitudes of the gradients
    float* magnitudes = new float[imWidth * imHeight];
    for (int i = 0; i < imWidth * imHeight; i++){
        magnitudes[i] = sqrt(Gx[i]*Gx[i] + Gy[i]*Gy[i]);
    }

    // Calculate gradient direction to be 1 of four directions:
    // horizontal, vertical, left to right diagonal, right to left diagonal
    float* thetas = new float[imWidth * imHeight];
    for (int i = 0; i < imWidth * imHeight; i++){
        thetas[i] = atan2(Gy[i], Gx[i]);
        if (        (thetas[i] <= pi/8 && thetas[i] > -pi/8)
                 || (thetas[i] >= -pi && thetas[i] <= -7*pi/8)
                 || (thetas[i] <= pi && thetas[i] > 7*pi/8)){thetas[i] = 0.0f;}
        else if (   (thetas[i] <= 3*pi/8 && thetas[i] > pi/8)
                 || (thetas[i] > -3*pi/8 && thetas[i] <= -pi/8)){thetas[i] = 45.0f;}
        else if (   (thetas[i] <= 5*pi/8 && thetas[i] > 3*pi/8)
                 || (thetas[i] > -5*pi/8 && thetas[i] <= -3*pi/8) ){thetas[i] = 90.0f;}
        else if (   (thetas[i] <= 7*pi/8 && thetas[i] > 5*pi/8)
                 || (thetas[i] > -7*pi/8 && thetas[i] <= -5*pi/8)){thetas[i] = 135.0f;}
        else {thetas[i] = 0;} // For some reason 3.14159 is not counted
    }

    for (int row = 1; row < imHeight - 1; row++){
        for (int col = 0; col < imWidth - 1; col++){
            if (thetas[row*imWidth + col] == 0){
                if (   (magnitudes[row*imWidth + col] > magnitudes[(row)*imWidth + (col + 1)])
                    && (magnitudes[row*imWidth + col] > magnitudes[(row)*imWidth + (col - 1)])){
                        continue;
                    }
                else {
                    magnitudes[row*imWidth + col] = 0;
                }
            }
            else if (thetas[row*imWidth + col] == 45){
                if (   (magnitudes[row*imWidth + col] > magnitudes[(row + 1)*imWidth + (col + 1)])
                    && (magnitudes[row*imWidth + col] > magnitudes[(row - 1)*imWidth + (col - 1)])){
                        continue;
                    }
                else {
                    magnitudes[row*imWidth + col] = 0;
                }

            }
            else if (thetas[row*imWidth + col] == 90){
                if (   (magnitudes[row*imWidth + col] > magnitudes[(row + 1)*imWidth + (col)])
                    && (magnitudes[row*imWidth + col] > magnitudes[(row - 1)*imWidth + (col)])){
                        continue;
                    }
                else {
                    magnitudes[row*imWidth + col] = 0;
                }     
            }
            else if (thetas[row*imWidth + col] == 135){
                if (   (magnitudes[row*imWidth + col] > magnitudes[(row - 1)*imWidth + (col + 1)])
                    && (magnitudes[row*imWidth + col] > magnitudes[(row + 1)*imWidth + (col - 1)])){
                        continue;
                    }
                else {
                    magnitudes[row*imWidth + col] = 0;
                }
                
            }
        }
    }

    for (int i = 0; i < imHeight * imWidth; i++){
        std::cout << magnitudes[i] << std::endl;
    }

    float highThresh = 20.0f;
    float lowThresh = 10.0f;

    int* evals = new int[imWidth * imHeight];

    for (int i = 0; i < imWidth * imHeight; i++){
        if (magnitudes[i] < highThresh && magnitudes[i] > lowThresh){
            evals[i] = 1;
        }
        else if (magnitudes[i] < lowThresh){
            evals[i] = 0;
            magnitudes[i] = 0;
        }
        else {
            evals[i] = 2;
        }
    }


    for (int row = 1; row < imHeight - 1; row++){
        for (int col = 1; col < imWidth - 1; col++){
            if(evals[row*imWidth + col]==1){
                if(  (evals[(row - 1)*imWidth + (col - 1)] == 2)
                   ||(evals[(row - 1)*imWidth + (col)] == 2)
                   ||(evals[(row - 1)*imWidth + (col + 1)] == 2)
                   ||(evals[(row)*imWidth + (col - 1)] == 2)
                   ||(evals[(row)*imWidth + (col + 1)] == 2)
                   ||(evals[(row + 1)*imWidth + (col - 1)] == 2)
                   ||(evals[(row + 1)*imWidth + (col)] == 2)
                   ||(evals[(row + 1)*imWidth + (col+1)] == 2)
                   ){
                       continue;
                   }
                else{
                    magnitudes[row*imWidth + col] = 0;
                }
            }
        }
    }
    delete Gx;
    delete Gy;

    // Feed array of pixels back into CImg and save the new image
    CImg <float>  output(magnitudes, imWidth, imHeight);
    output.save("../images/output/output.bmp");

    delete magnitudes;
    delete thetas;

    return 0;
}
