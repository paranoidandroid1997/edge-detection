#include <iostream>
#include <hip/hip_runtime.h>

#include "CImg.h"
#include "convolution.h"
using namespace cimg_library;

#include "convolution.h"

int main(){
    // Load in the image
    CImg<float> image("../images/input/test-image-2.pgm");

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
    float gaussianKernel[9] = {1.0f,2.0f,1.0f,2.0f,4.0f,2.0f,1.0f,2.0f,1.0f};
    for (int i = 0; i < 9; i++){
        gaussianKernel[i] *= 1.0f/16.0f;
    }

    convolve(rawData, imWidth, imHeight, gaussianKernel, 3, 3);

    // Feed array of pixels (rawData) back into CImg and save the new image
    CImg <float>  output(rawData, imWidth, imHeight);
    output.save("../images/output/output.bmp");

    delete rawData;
    return 0;
}
