
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <string>

void setCursorPosition(int x, int y);
void setBackgroundColorRGB(int r, int g, int b);
void reset();

void loadPNG(const char* imgpath);
void savePNG(const char* filename);

int width, height, channels;
unsigned char* img;

__device__ int dev_width, dev_height, dev_channels;
__device__ unsigned char* dev_img;

__global__ void OverLayingRed() {
	for (int y = 0; y < dev_height; y++) {
		for (int x = 0; x < dev_width; x++) {
			int pixelIndex = (y * dev_width + x) * dev_channels;
			dev_img[pixelIndex + 0] = 0;
			dev_img[pixelIndex + 1] = 0;
			dev_img[pixelIndex + 2] = 0;
		}
	}
}

int main() {

	loadPNG("C:\\Users\\horga\\Downloads\\lil_test.png");

	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc((void**)&dev_img, imgSize);


	cudaMemcpyToSymbol(dev_width, &width, sizeof(int));
	cudaMemcpyToSymbol(dev_height, &height, sizeof(int));
	cudaMemcpyToSymbol(dev_channels, &channels, sizeof(int));
	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);

	OverLayingRed << <1, 1 >> > ();

	cudaMemcpy(img, dev_img, imgSize, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();


	//3D glitch implementation

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int pixelIndex = (y * width + x) * channels;
			unsigned char r = img[pixelIndex + 0];
			unsigned char g = img[pixelIndex + 1];
			unsigned char b = img[pixelIndex + 2];

			setBackgroundColorRGB(r, g, b);

			setCursorPosition(2 * x, y);
			printf(" ");
			setCursorPosition(2 * x + 1, y);
			printf(" ");
		}
	}

	cudaFree(dev_img);
	reset();
	savePNG("csudakép.png");
	return 0;
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
