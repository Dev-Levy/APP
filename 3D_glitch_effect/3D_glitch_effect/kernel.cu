
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <string>

int width, height, channels;
double hsl_H, hsl_S, hsl_L;
unsigned char* img;

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

void convert_RGB_To_HSL(int red, int green, int blue, double* H, double* S, double* L) {
	double red_perc = red / 255.0;
	double green_perc = green / 255.0;
	double blue_perc = blue / 255.0;

	double max, min;

	//max
	if (red_perc > green_perc && red_perc > blue_perc)
		max = red_perc;
	else if (green_perc > red_perc && green_perc > blue_perc)
		max = green_perc;
	else if (blue_perc > green_perc && blue_perc > red_perc)
		max = blue_perc;

	//min
	if (red_perc < green_perc && red_perc < blue_perc)
		min = red_perc;
	else if (green_perc < red_perc && green_perc < blue_perc)
		min = green_perc;
	else if (blue_perc < green_perc && blue_perc < red_perc)
		min = blue_perc;
	else {
		max = red_perc;
		min = red_perc;
	}

	//Luminocity
	*L = (max + min) / 2.0;

	//Saturation
	if (max == min) {
		*S = 0;
		*H = 0;
	}
	else if (*L <= 0.5) {
		*S = (max - min) / (max + min);
	}
	else if (*L > 0.5) {
		*S = (max - min) / (2.0 - max - min);
	}

	//Hue
	double H_double;
	if (max == red_perc) {
		H_double = (green_perc - blue_perc) / (max - min);
	}
	else if (max == green_perc) {
		H_double = 2.0 + (blue_perc - red_perc) / (max - min);
	}
	else if (max == blue_perc) {
		H_double = 4.0 + (red_perc - green_perc) / (max - min);
	}

	*H = H_double * 60.0;
	if (*H < 0)
		*H += 360;
}
void convert_HSL_To_RGB(double H, double S, double L, unsigned char* red, unsigned char* green, unsigned char* blue) {
	if (S == 0.0)
	{
		*red = L * 255.0;
		*green = L * 255.0;
		*blue = L * 255.0;
	}

	double C = (1 - std::abs(2 * L - 1)) * S;

	auto qwe = static_cast<int>(H / 60) % 2 - 1;
	auto asd = 1 - std::abs(qwe);

	double X = C * static_cast<double>(asd);

	double M = L - (C / 2);

	double temp_R;
	double temp_G;
	double temp_B;

	if (H >= 0 && H < 60) {
		temp_R = C; temp_G = X;	temp_B = 0;
	}
	else if (H >= 60 && H < 120) {
		temp_R = X; temp_G = C;	temp_B = 0;
	}
	else if (H >= 120 && H < 180) {
		temp_R = 0; temp_G = C;	temp_B = X;
	}
	else if (H >= 180 && H < 240) {
		temp_R = 0; temp_G = X;	temp_B = C;
	}
	else if (H >= 240 && H < 300) {
		temp_R = X; temp_G = 0;	temp_B = C;
	}
	else if (H >= 300 && H < 360) {
		temp_R = C; temp_G = 0;	temp_B = X;
	}

	*red = (temp_R + M) * 255.0;
	*green = (temp_G + M) * 255.0;
	*blue = (temp_B + M) * 255.0;


	/*double temp1;
	if (L <= 0.5)
		temp1 = L * (1.0 + S);
	else
		temp1 = L + S - (L * S);

	double temp2 = 2.0 * L - temp1;

	double H_perc = H / 360.0;
	double temp_R = H_perc + 1.0 / 3.0;
	double temp_G = H_perc;
	double temp_B = H_perc - 1.0 / 3.0;

	clamp_zero_to_one(&temp_R);
	clamp_zero_to_one(&temp_G);
	clamp_zero_to_one(&temp_B);

	calculateRGB(&temp_R, temp1, temp2);
	calculateRGB(&temp_G, temp1, temp2);
	calculateRGB(&temp_B, temp1, temp2);

	*red = temp_R * 255.0;
	*green = temp_G * 255.0;
	*blue = temp_B * 255.0;*/
}

void clamp_zero_to_one(double* num) {
	if (*num < 0)
		*num += 1.0;
	else if (*num > 1)
		*num -= 1.0;
}
void calculateRGB(double* pRGB, double temp1, double temp2) {
	if (6 * (*pRGB) < 1)
		*pRGB = (temp2 + (temp1 - temp2)) * 6 * (*pRGB);
	else if (2 * (*pRGB) < 1)
		*pRGB = temp1;
	else if (3 * (*pRGB) < 2)
		*pRGB = (temp2 + (temp1 - temp2)) * 6 * (2.0 / 3.0 - *pRGB);
	else
		*pRGB = temp2;
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

		d_img[idx + 0] = gray;
		d_img[idx + 1] = gray;
		d_img[idx + 2] = gray;
	}
}

void GrayScalingSetup(unsigned char* img, int width, int height, int channels) {

	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc(&dev_img, imgSize);

	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	GrayScaling << <gridSize, blockSize >> > (dev_img, width, height, channels);

	cudaMemcpy(img, dev_img, imgSize, cudaMemcpyDeviceToHost);

	cudaFree(dev_img);
}

__global__ void OverLayingRed(unsigned char* d_img, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels;
		d_img[idx + 0] = 0;
		//d_img[idx + 1] = 0;
		//d_img[idx + 2] = 0;
	}
}

void OverlaySetup(unsigned char* d_img, int width, int height, int channels) {
	size_t imgSize = width * height * channels * sizeof(unsigned char);
	cudaMalloc(&dev_img, imgSize);

	cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	OverLayingRed << <gridSize, blockSize >> > (dev_img, width, height, channels);

	cudaMemcpy(img, dev_img, imgSize, cudaMemcpyDeviceToHost);

	cudaFree(dev_img);
}

int main() {

	loadPNG("C:\\Users\\horga\\Downloads\\lil_test.png");

	OverlaySetup(img, width, height, channels);

	savePNG("csudakép.png");
	return 0;
}
