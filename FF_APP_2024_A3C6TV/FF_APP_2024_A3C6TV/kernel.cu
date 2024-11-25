
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "fstream"
#include "sstream"
#include "iostream"

int width, height, channels;
unsigned char* img;
unsigned char* dev_img;


void loadPNG(const char* imgpath) {
	img = stbi_load(imgpath, &width, &height, &channels, 0);

	if (img == NULL) {
		printf("Failed to load image\n");
	}
}
void saveTXT() {
	size_t pixelCount = width * height;
	std::ofstream outputFile("csudakep.txt", std::ios::binary);

	for (size_t i = 0; i < pixelCount; i++)
	{
		if (i % width == 0) {
			outputFile << std::endl;
			std::cout << std::endl;
		}

		switch (img[i])
		{
		case 35:
			outputFile << "#";
			std::cout << static_cast<char>(35);
			break;
		case 176:
			outputFile << "░";
			std::cout << static_cast<char>(35);
			break;
		case 177:
			outputFile << "▒";
			std::cout << static_cast<char>(35);
			break;
		case 178:
			outputFile << "▓";
			std::cout << static_cast<char>(35);
			break;
		case 219:
			outputFile << "█";
			std::cout << static_cast<char>(35);
			break;
		default:
			break;
		}
	}
}

__global__ void Pixels_To_ASCII_Kernel(unsigned char* d_img, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > width || y > height) return;

	int idx = (y * width + x) * channels;

	unsigned char r = d_img[idx + 0];
	unsigned char g = d_img[idx + 1];
	unsigned char b = d_img[idx + 2];

	unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

	if (gray <= 51)
		d_img[idx] = 35; //hashtag #
	else if (gray > 51 && gray <= 102)
		d_img[idx] = 176; //light 
	else if (gray > 102 && gray <= 153)
		d_img[idx] = 177; //medium
	else if (gray > 153 && gray <= 204)
		d_img[idx] = 178; //dark
	else
		d_img[idx] = 219; //full

}

void Pixels_To_ASCII(unsigned char* img, int width, int height, int channels) {

	//malloc and copy
	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc(&dev_img, imgSize);
	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);

	//kernel call
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	Pixels_To_ASCII_Kernel << <gridSize, blockSize >> > (dev_img, width, height, channels);
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