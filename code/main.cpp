#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#include "CImg.h"
using namespace cimg_library;

int main(){
    // Load in the image
    CImg<unsigned char> image("../images/input/test-image-2.pgm");

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
    unsigned char* rawData = new unsigned char[imSize];
    for (int row = 0; row < imHeight; row++){
        for (int col = 0; col < imWidth; col++){
            rawData[row*imWidth + col] = image(col, row); 
        }
    }

    CImg <unsigned char>  output(rawData, imWidth, imHeight);
    output.save("../images/output/output.bmp");

    delete rawData;
    return 0;
}
