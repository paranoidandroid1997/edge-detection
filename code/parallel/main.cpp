#include <iostream>
#include <cmath>

#include "CImg.h"
using namespace cimg_library;

#include "convolution.h"
#include "magnitudes.h"
#include "thetas.h"
#include "classify.h"
#include "compare.h"
#include "neighbour.h"

int main(){
    for (int y = 0; y < 1; y++){

    // Load in the image
    CImg<float> image("../../images/input/test-image-6.pgm");

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

    CImg <float>  output1(rawData, imWidth, imHeight);
    output1.save("../../images/output/original-image.bmp");

    for (int i = 0; i < 1; i++){


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
    hipDeviceSynchronize();
    delete[] rawData;

    CImg <float>  output2(blurredData, imWidth, imHeight);
    output2.save("../../images/output/blurred-image.bmp");

    // Define sobel operators to find the gradients in the x and y directions
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
    hipDeviceSynchronize();
    convolve(blurredData, imWidth, imHeight, sobelGy, 3, 3, Gy);
    hipDeviceSynchronize();

    delete[] blurredData;

    // Use the derivatives in the x and y direction to compute the magnitudes of the gradients
    float* magnitudes = new float[imWidth * imHeight];
    findMagnitudes(Gx, Gy, magnitudes, imSize);
    hipDeviceSynchronize();
    delete[] Gx;
    delete[] Gy;

    CImg <float>  output3(magnitudes, imWidth, imHeight);
    output3.save("../../images/output/magnitudes-image.bmp");

    float* thetas = new float[imWidth * imHeight];
    findThetas(Gx, Gy, thetas, imSize);
    hipDeviceSynchronize();

    // Compare each pixel to it's two neighbors (these are decided by the direction previously calculated)
    // If the pixel is the largest out of the three, then keep it
    // If it's not the largest, then set it to 0
    float* newMagnitudes = new float[imWidth * imHeight];
    compare(newMagnitudes, magnitudes, thetas, imHeight, imWidth);
    hipDeviceSynchronize();
    delete[] thetas;
    delete[] magnitudes;

    CImg <float>  output4(newMagnitudes, imWidth, imHeight);
    output4.save("../../images/output/after-neighbourcompare-image.bmp");

    // Set high and low threshold values.
    // Values in between low and high are "weak"
    // Values below low are set to 0
    // Values above high are strong
    float highThresh =100.0f;
    float lowThresh = 60.0f;

    // classify each pixel as either weak (1), strong (2), or below low (0)
    // Store the result in a new array evals
    int* evals = new int[imWidth * imHeight];
    classify(evals, newMagnitudes, highThresh, lowThresh, imSize);
    hipDeviceSynchronize();

    CImg <float>  output5(newMagnitudes, imWidth, imHeight);
    output5.save("../../images/output/after-thresholding-image.bmp");

    // Decide what to do with the weak edge pixels by checking the 8 surrounding neighbours
    // If at least one of them is strong, then the weak pixel can stay
    // If none are strong, then the weak pixel is set to 0
    float* finalPixels = new float[imWidth * imHeight];
    neighbourCheck(finalPixels, newMagnitudes, evals, imHeight, imWidth);
    hipDeviceSynchronize();
    delete[] newMagnitudes;

    // Feed array of pixels back into CImg and save the new image
    CImg <float>  outputf(finalPixels, imWidth, imHeight);
    outputf.save("../../images/output/final-output-real.bmp");

    delete[] evals;
    delete[] finalPixels;

    std::cout << y << std::endl;
    std::cout << std::endl;
    }
    }

    return 0;
}