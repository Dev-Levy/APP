#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <vector>
#define N 10000
#define BLOCK_SIZE 512


__device__ int dev_minIdx;

__global__ void MinimumMulFind_N(int* dev_v1) {
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	__shared__ size_t min;


	if (x == 0)
		min = SIZE_MAX;
	else if (x > N)
		return;
	__syncthreads();

	size_t mul = 1;
	for (size_t i = 0; i < 10; i++)
	{
		mul *= dev_v1[x + i];
	}

	if (atomicMin(&min, mul) > mul)
		atomicExch(&dev_minIdx, x);
}


int main() {
	//random generátor setup
	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_int_distribution<>dist(1, 10);

	//feltöltés
	std::vector<int> v1;
	for (size_t i = 0; i < N; i++)
		v1.push_back(dist(gen));


	//CPU
	size_t min = SIZE_MAX;
	int minIdx = -1;
	for (size_t i = 0; i < N - 10; i++)
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

	cudaEvent_t start_event, end_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&end_event);


	cudaMalloc((void**)&dev_v1, N * sizeof(int));
	cudaError_t err = cudaGetLastError();

	//memory copying

	cudaMemcpy(dev_v1, &v1[0], N * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaGetLastError();

	cudaEventRecord(start_event, 0);
	MinimumMulFind_N << <N / BLOCK_SIZE + 1, BLOCK_SIZE >> > (dev_v1);
	cudaEventRecord(end_event, 0);

	cudaEventSynchronize(start_event);
	cudaEventSynchronize(end_event);
	cudaDeviceSynchronize();
	err = cudaGetLastError();

	cudaMemcpyFromSymbol(&GPU_minIdx, dev_minIdx, sizeof(int));
	err = cudaGetLastError();

	cudaFree(dev_v1);


	float elapsed_ms;
	cudaEventElapsedTime(&elapsed_ms, start_event, end_event);


	/*for (size_t i = 0; i < N; i++)
	{
		printf("%llu: %d\n", i, v1[i]);
	}*/
	printf("\n%d", minIdx);
	printf("\n%d", GPU_minIdx);


	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int maxThreads = deviceProp.reg;

	printf("\n%d", maxThreads);
}