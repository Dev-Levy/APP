#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <time.h>

#define BLOCK_SIZE 500
#define N 10410
#define K 6


int numArray[N];
int resIndex = -1;
bool res[N - K];
__device__ int dev_numArray[N];
__device__ int dev_resIndex = -1;
__device__ bool dev_res[N - K];

__global__ void FindIncSeries_N(int k) {

	int x = threadIdx.x;
	int counter = 0;

	while (counter < k - 1 && dev_numArray[x + counter] < dev_numArray[x + counter + 1]) {
		counter++;
	}
	if (counter == k - 1)
		dev_resIndex = x;
}

__global__ void FindIncSeries_N_BLOCKS(int k) {

	__shared__ int shr_numArray[N];
	int x = threadIdx.x;
	int bl_x = blockIdx.x;
	int counter = 0;

	int i = x + bl_x * blockDim.x;
	if (i < N)
		shr_numArray[i] = dev_numArray[i];

	//túlindex védelem
	if (bl_x == N / blockDim.x + 1 && x > N % blockDim.x - k)
		return;

	while (counter < k - 1 && shr_numArray[bl_x * blockDim.x + x + counter] < shr_numArray[bl_x * blockDim.x + x + counter + 1]) {
		counter++;
	}
	if (counter == k - 1)
		dev_resIndex = bl_x * BLOCK_SIZE + x;
}

__global__ void FindIncSeries_OneCicle() {
	int x = threadIdx.x;
	int y = threadIdx.y;
	if (y == 0)
		dev_res[x] = true;

	if (!(dev_numArray[x + y] < dev_numArray[x + y + 1])) //nem növekvő
		dev_res[x] = false;
}

int main() {

	//random generate
	srand(time(NULL));
	for (size_t i = 0; i < N; i++)
	{
		numArray[i] = rand() % 10000;
	}
	numArray[10400] = 1;
	numArray[10401] = 2;
	numArray[10402] = 3;
	numArray[10403] = 4;
	numArray[10404] = 5;
	numArray[10405] = 6;
	numArray[10406] = 7;
	numArray[10407] = 8;
	numArray[10408] = 9;
	numArray[10409] = 10;


	cudaMemcpyToSymbol(dev_numArray, numArray, N * sizeof(int));
	//FindIncSeries_N << < 1, N - K >> > (K);
	FindIncSeries_N_BLOCKS << < N / BLOCK_SIZE + 1, BLOCK_SIZE >> > (K);
	cudaMemcpyFromSymbol(&resIndex, dev_resIndex, sizeof(int));

	//FindIncSeries_OneCicle << < 1, dim3(N - K, K - 1) >> > ();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
		return -1;
	}
	//cudaMemcpyFromSymbol(res, dev_res, (N - K) * sizeof(bool));

	//CPU
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

	//printing the nums
	/*for (size_t i = 0; i < N; i++)
	{
		printf("%d:\t%d\n", i, numArray[i]);
	}*/
	/*for (size_t i = 0; i < N - K; i++)
	{
		if (res[i] == true)
		{
			printf("Ez egy jo index: %d\n", i);
		}
	}*/
	printf("Index, ahol %d szam novekvo sorrendben van egymas utan: %d.", K, resIndex);
}
