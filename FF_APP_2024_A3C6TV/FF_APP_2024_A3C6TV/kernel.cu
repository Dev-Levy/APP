
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "fstream"
#include "sstream"



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
	std::ostringstream oss;

	for (size_t i = 0; i < pixelCount; i++)
	{
		if (i % width == 0)
			oss << std::endl;
		oss << img[i];
	}
	std::ofstream outputFile("csudakep.txt");
	std::string s = oss.str();
	outputFile << s;

}

__global__ void Pixels_To_ASCII_Kernel(unsigned char* d_img, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels;

		unsigned char r = d_img[idx + 0];
		unsigned char g = d_img[idx + 1];
		unsigned char b = d_img[idx + 2];

		unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

		if (gray <= 51)
			d_img[idx] = 35; //hashtag
		else if (gray > 51 && gray <= 102)
			d_img[idx] = 176; //light
		else if (gray > 102 && gray <= 153)
			d_img[idx] = 177; //medium
		else if (gray > 153 && gray <= 204)
			d_img[idx] = 178; //dark
		else
			d_img[idx] = 219; //full

	}

}

void Pixels_To_ASCII(unsigned char* img, int width, int height, int channels) {

	//malloc and copy
	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc(&dev_img, imgSize);
	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);

	//kernel call
	int blockSize = 192;
	int blockNum;

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