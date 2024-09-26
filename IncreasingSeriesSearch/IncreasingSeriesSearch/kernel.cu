#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <time.h>

#define N 1000
#define K 6


int numArray[N];
int resIndex = -1;
__device__ int dev_numArray[N];
__device__ int dev_resIndex = -1;

__global__ void FindIncSeries(int k) {

	int counter = 0;
	while (counter < k - 1 && dev_numArray[threadIdx.x + counter] < dev_numArray[threadIdx.x + counter + 1]) {
		counter++;
	}
	if (counter == k - 1)
		dev_resIndex = threadIdx.x;
}

int main() {

	srand(time(NULL));

	for (size_t i = 0; i < N; i++)
	{
		numArray[i] = rand() % 10000;
	}


	cudaMemcpyToSymbol(dev_numArray, numArray, N * sizeof(int));
	FindIncSeries << < 1, N - K >> > (K);
	cudaError_t err = cudaMemcpyFromSymbol(&resIndex, dev_resIndex, sizeof(int));
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}


	/*int index = -1;
	for (size_t i = 0; i < N - K; i++)
	{
		int counter = 0;
		while (counter < K - 1 && numArray[i + counter] < numArray[i + counter + 1]) {
			counter++;
		}
		if (counter == K - 1) {
			index = i;
		}
		else
			counter = 0;
	}*/

	/*for (size_t i = 0; i < N; i++)
	{
		printf("%d:\t%d\n", i, numArray[i]);
	}*/
	printf("Index: %d.", resIndex);
}