#include <iostream>
#include <cmath>

#include "CImg.h"
#include "convolution.h"
using namespace cimg_library;

#include "convolution.h"

#define pi 3.14159

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
    delete[] rawData;

    CImg <float>  output1(blurredData, imWidth, imHeight);
    output1.save("../../images/output/output-1.bmp");

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
    delete[] blurredData;

    // Use the derivatives in the x and y direction to compute the magnitudes of the gradients
    float* magnitudes = new float[imWidth * imHeight];
    for (int i = 0; i < imWidth * imHeight; i++){
        magnitudes[i] = sqrt(Gx[i]*Gx[i] + Gy[i]*Gy[i]);
    }
    delete[] Gx;
    delete[] Gy;

    CImg <float>  output2(magnitudes, imWidth, imHeight);
    output2.save("../../images/output/output-2.bmp");

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

    // Compare each pixel to it's two neighbors (these are decided by the direction previously calculated)
    // If the pixel is the largest out of the three, then keep it
    // If it's not the largest, then set it to 0
    for (int row = 1; row < imHeight - 1; row++){
        for (int col = 1; col < imWidth - 1; col++){
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
    delete[] thetas;

    CImg <float>  output3(magnitudes, imWidth, imHeight);
    output3.save("../../images/output/output-3.bmp");

    // for (int i = 0; i < imHeight * imWidth; i++){
    //     std::cout << magnitudes[i] << std::endl;
    // }


    // Set high and low threshold values.
    // Values in between low and high are "weak"
    // Values below low are set to 0
    // Values above high are strong
    float highThresh =100.0f;
    float lowThresh = 100.0f/3.0f;


    // classify each pixel as either weak (1), strong (2), or below low (0)
    // Store the result in a new array evals
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

    CImg <float>  output4(magnitudes, imWidth, imHeight);
    output4.save("../../images/output/output-4.bmp");


    // Decide what to do with the weak edge pixels by checking the 8 surrounding neighbours
    // If at least one of them is strong, then the weak pixel can stay
    // If none are strong, then the weak pixel is set to 0
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
                       magnitudes[row*imWidth + col] = 255;
                   }
                else{
                    magnitudes[row*imWidth + col] = 0;
                }
            }
            else if (evals[row*imWidth + col] == 2){
                 magnitudes[row*imWidth + col] = 255; 
            }
        }
    }

    // Feed array of pixels back into CImg and save the new image
    CImg <float>  outputf(magnitudes, imWidth, imHeight);
    outputf.save("../../images/output/final-output.bmp");

    delete[] magnitudes;
    delete[] evals;

    return 0;
}
