
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <sstream>

#include <stdio.h>
#include <string>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

typedef struct rgb {
	float r, g, b;
} RGB;

typedef struct hsl {
	float h, s, l;
} HSL;

int width, height, channels;
unsigned char* img;
unsigned long size, size_w_channels;

__device__ unsigned char* dev_img;

#pragma region helper_functions
void loadPNG(const char* imgpath) {
	img = stbi_load(imgpath, &width, &height, &channels, 0);

	if (img == NULL) {
		printf("Failed to load image\n");
	}
}
void savePNG(const char* filename)
{
	if (stbi_write_png(filename, width, height, channels, img, width * channels))
		printf("PNG file created successfully.");
	else
		printf("Failed to create PNG file.");
	stbi_image_free(img);
}

void setCursorPosition(int x, int y) {
	printf("\033[%d;%dH", y, x);
}
void setBackgroundColorRGB(int r, int g, int b) {
	printf("\033[48;2;%d;%d;%dm", r, g, b);
}
void reset() {
	printf("\033[2J"); //cls
	printf("\033[H"); //cursor reset
	fflush(stdout); //output clear
	printf("\033[0m"); //style reset
}

HSL rgb2hsl(float r, float g, float b) {

	HSL result;

	r /= 255;
	g /= 255;
	b /= 255;

	float max = MAX(MAX(r, g), b);
	float min = MIN(MIN(r, g), b);

	result.h = result.s = result.l = (max + min) / 2;

	if (max == min) {
		result.h = result.s = 0; // achromatic
	}
	else {
		float d = max - min;
		result.s = (result.l > 0.5) ? d / (2 - max - min) : d / (max + min);

		if (max == r) {
			result.h = (g - b) / d + (g < b ? 6 : 0);
		}
		else if (max == g) {
			result.h = (b - r) / d + 2;
		}
		else if (max == b) {
			result.h = (r - g) / d + 4;
		}

		result.h /= 6;
	}

	return result;

}

float hue2rgb(float p, float q, float t) {

	if (t < 0)
		t += 1;
	if (t > 1)
		t -= 1;
	if (t < 1. / 6)
		return p + (q - p) * 6 * t;
	if (t < 1. / 2)
		return q;
	if (t < 2. / 3)
		return p + (q - p) * (2. / 3 - t) * 6;

	return p;

}

RGB hsl2rgb(float h, float s, float l) {

	RGB result;

	if (0 == s) {
		result.r = result.g = result.b = l; // achromatic
	}
	else {
		float q = l < 0.5 ? l * (1 + s) : l + s - l * s;
		float p = 2 * l - q;
		result.r = hue2rgb(p, q, h + 1. / 3) * 255;
		result.g = hue2rgb(p, q, h) * 255;
		result.b = hue2rgb(p, q, h - 1. / 3) * 255;
	}

	return result;

}
#pragma endregion

__global__ void GrayScaling(unsigned char* d_img, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels;

		unsigned char r = d_img[idx + 0];
		unsigned char g = d_img[idx + 1];
		unsigned char b = d_img[idx + 2];

		unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

		if (gray <= 51)
			dev_img[idx] = 35; //hashtag
		else if (gray > 51 && gray <= 102)
			dev_img[idx] = 176; //light
		else if (gray > 102 && gray <= 153)
			dev_img[idx] = 177; //medium
		else if (gray > 153 && gray <= 204)
			dev_img[idx] = 178; //dark
		else
			dev_img[idx] = 219; //full

	}
}
void GrayScalingSetup(unsigned char* img, int width, int height, int channels) {

	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc(&dev_img, imgSize);

	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	GrayScaling << <gridSize, blockSize >> > (dev_img, width, height, channels);
	err = cudaGetLastError();


	cudaMemcpy(img, dev_img, imgSize, cudaMemcpyDeviceToHost);
	err = cudaGetLastError();

	cudaFree(dev_img);
}

int main() {

	loadPNG("C:\\Users\\horga\\Downloads\\lil_test.png");
	size = width * height * sizeof(unsigned char);
	size_w_channels = size * channels;

	GrayScalingSetup(img, width, height, channels);


	std::stringstream ss;

	for (size_t i = 0; i < size; i++)
	{
		ss << img[i];
	}


	//savePNG("csudakép.png");
	return 0;
}
