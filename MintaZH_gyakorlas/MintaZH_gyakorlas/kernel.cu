#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <vector>
#define N 100

__device__ int dev_minIdx;

__global__ void MinimumMulFind_N(int* dev_v1) {
	int x = threadIdx.x;
	__shared__ size_t min;


	if (x == 0)
		min = SIZE_MAX;
	__syncthreads();

	size_t mul = 1;
	for (size_t i = 0; i < 10; i++)
	{
		mul *= dev_v1[x + i];
	}

	size_t akt = atomicMin(&min, mul);

	if (akt == mul)
		dev_minIdx = x;
}


int main() {
	//random generátor setup
	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_int_distribution<>dist(1, 10);

	//feltöltés
	std::vector<int> v1 = { 1,4,3,1,8,9,7,6,1,5,5,1,1,5,2,6,6,1,4,1,9,8,6,7,3,6,5,6,8,8,1,1,3,2,1,7,6,1,3,6 };
	/*for (size_t i = 0; i < N; i++)
		v1.push_back(dist(gen));*/


		//CPU
	size_t min = SIZE_MAX;
	int minIdx = -1;
	for (size_t i = 0; i < v1.size() - 10; i++)
	{
		size_t current = 1;
		for (size_t j = 0; j < 10; j++)
			current *= v1[i + j];

		if (min > current) {
			min = current;
			minIdx = i;
		}
	}

	//GPU

	//memory allocation
	int* dev_v1;
	int GPU_minIdx;
	cudaMalloc((void**)&dev_v1, N * sizeof(int));
	cudaError_t err = cudaGetLastError();

	//memory copying

	cudaMemcpy(dev_v1, &v1[0], N * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaGetLastError();

	MinimumMulFind_N << <1, v1.size() - 10 >> > (dev_v1);
	cudaDeviceSynchronize();
	err = cudaGetLastError();

	cudaMemcpyFromSymbol(&GPU_minIdx, dev_minIdx, sizeof(int));
	err = cudaGetLastError();

	cudaFree(dev_v1);

	for (size_t i = 0; i < v1.size(); i++)
	{
		printf("%llu: %d\n", i, v1[i]);
	}
	printf("\n%d", minIdx);
	printf("\n%d", GPU_minIdx);
}