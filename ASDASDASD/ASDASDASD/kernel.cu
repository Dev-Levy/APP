
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <sstream>

#define BLOCKSIZE 16

__device__ char dev_asciichars[] = "@%#*+=-:. ";

__global__ void Pixels_To_ASCII_Kernel(unsigned char* d_img, char* d_ascii, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int idx = (y * width + x) * channels;
	float r = d_img[idx] / 255.0f;
	float g = d_img[idx + 1] / 255.0f;
	float b = d_img[idx + 2] / 255.0f;

	float grayscale = 0.299f * r + 0.587f * g + 0.114f * b;

	int asciiIndex = (int)(grayscale * 9);
	if (asciiIndex > 9)
		asciiIndex = 9;

	int ascii_width = width * 2;
	int out_idx = y * ascii_width + x * 2;

	d_ascii[out_idx] = dev_asciichars[asciiIndex];
	d_ascii[out_idx + 1] = dev_asciichars[asciiIndex];
}


void loadPNG(const char* imgpath, unsigned char*& img, int& width, int& height, int& channels) {
	img = stbi_load(imgpath, &width, &height, &channels, 0);
	if (!img) {
		std::cerr << "Failed to load image!" << std::endl;
		exit(1);
	}
}

void saveTXT(const char* filepath, char* asciiArt, int width, int height) {
	std::ofstream outputFile(filepath);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			std::cout << asciiArt[y * width + x];
			outputFile << asciiArt[y * width + x];
		}
		outputFile << '\n';
		std::cout << '\n';
	}
	outputFile.close();
}

int main() {
	const char* inputImage = "kep_lil.png";
	const char* outputAscii = "output.txt";

	unsigned char* h_img;
	int width, height, channels;

	loadPNG(inputImage, h_img, width, height, channels);

	unsigned char* d_img;
	char* d_ascii;

	int asciiWidth = width * 2;
	size_t imgSize = width * height * channels;
	size_t asciiSize = asciiWidth * height;

	cudaMalloc(&d_img, imgSize);
	cudaMalloc(&d_ascii, asciiSize);

	cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	Pixels_To_ASCII_Kernel << <gridSize, blockSize >> > (d_img, d_ascii, width, height, channels);
	cudaDeviceSynchronize();

	char* h_ascii = new char[asciiSize];
	cudaMemcpy(h_ascii, d_ascii, asciiSize, cudaMemcpyDeviceToHost);

	saveTXT(outputAscii, h_ascii, asciiWidth, height);

	cudaFree(d_img);
	cudaFree(d_ascii);
	stbi_image_free(h_img);
	delete[] h_ascii;

	std::cout << "ASCII art saved to " << outputAscii << std::endl;
	return 0;
}
