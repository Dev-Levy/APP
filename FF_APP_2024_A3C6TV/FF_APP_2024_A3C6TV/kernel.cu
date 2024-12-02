
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "fstream"
#include "sstream"
#include "iostream"

#define BLOCKSIZE 213

int width, height, channels;
unsigned char* img;
unsigned char* dev_img;
__device__ char dev_asciichars[] = "@%#*+=-:. ";


void loadPNG(const char* imgpath) {
	img = stbi_load(imgpath, &width, &height, &channels, 0);

	if (img == NULL) {
		printf("Failed to load image\n");
	}
}
void saveTXT() {
	size_t pixelCount = width * height;
	std::ofstream outputFile("csudakep.txt");

	for (size_t i = 0; i < pixelCount; i++)
	{
		if (i % width == 0 && i != 0) {
			outputFile << std::endl;
			std::cout << std::endl;
		}
		outputFile << img[i];
		std::cout << img[i];
	}
}

//__global__ void Pixels_To_ASCII_Kernel(unsigned char* d_img, int width, int height, int channels) {
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (x > width || y > height) return;
//
//	int idx = (y * width + x) * channels;
//
//	unsigned char r = d_img[idx + 0];
//	unsigned char g = d_img[idx + 1];
//	unsigned char b = d_img[idx + 2];
//
//	unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
//
//	if (gray <= 51)
//		d_img[idx] = 35; //hashtag #
//	else if (gray > 51 && gray <= 102)
//		d_img[idx] = 176; //light 
//	else if (gray > 102 && gray <= 153)
//		d_img[idx] = 177; //medium
//	else if (gray > 153 && gray <= 204)
//		d_img[idx] = 178; //dark
//	else
//		d_img[idx] = 219; //full
//
//}
__global__ void imageToASCII(unsigned char* dev_img, int width, int height, int channels) {
	int x = blockIdx.x * BLOCKSIZE + threadIdx.x;

	__shared__ char shr_asciiChars[10];

	if (x == 0)
	{
		for (size_t i = 0; i < 10; i++)
		{
			shr_asciiChars[i] = dev_asciichars[i];
		}
	}
	__syncthreads();

	if (x > (width * height)) return;

	int idx = x * 3;
	float r = dev_img[idx] / 255.0f; //0-1 között
	float g = dev_img[idx + 1] / 255.0f;
	float b = dev_img[idx + 2] / 255.0f;

	float grayscale = 0.299f * r + 0.587f * g + 0.114f * b;

	int index = (int)(grayscale * 9);
	dev_img[x] = shr_asciiChars[index];
}

void Pixels_To_ASCII(unsigned char* img, int width, int height, int channels) {

	//malloc and copy
	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc(&dev_img, imgSize);
	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);

	//kernel call
	int pixels = width * height;
	int blocks = pixels / BLOCKSIZE + 1;
	imageToASCII << <blocks, BLOCKSIZE >> > (dev_img, width, height, channels);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();

	//copy back and free
	cudaMemcpy(img, dev_img, imgSize, cudaMemcpyDeviceToHost);
	cudaFree(dev_img);
}

int main() {

	loadPNG("C:\\Users\\horga\\Downloads\\lil_test.png");

	Pixels_To_ASCII(img, width, height, channels);

	saveTXT();
	return 0;
}