#include <iostream>
#include <cmath>

#include "CImg.h"
using namespace cimg_library;

#include "convolution.h"

#define pi 3.14159;

int main(){
    // Load in the image
    CImg<float> image("../../images/input/test-image-2.pgm");

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

    // Define a 5X5 gaussian kerenl
    float gaussianKernel[25] = {2.0f,4.0f,5.0f,4.0f,2.0f,
                                4.0f,9.0f,12.0f,9.0f,4.0f,
                                5.0f,12.0f,15.0f,12.0f,5.0f,
                                4.0f,9.0f,12.0f,9.0f,4.0f,
                                2.0f,4.0f,5.0f,4.0f,2.0f};

    for (int i = 0; i < 25; i++){
        gaussianKernel[i] *= 1.0f/159.0f;
    }

    float* blurredData = new float[imWidth * imHeight];
    convolve(rawData, imWidth, imHeight, gaussianKernel, 5, 5, blurredData);
    delete rawData;

    // Feed array of pixels back into CImg and save the new image
    CImg <float>  outputf(blurredData, imWidth, imHeight);
    outputf.save("../../images/output/final-output.bmp");

    delete blurredData;

    return 0;
}