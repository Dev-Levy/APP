
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int tomb[] = { 2,4,8,16,32 };

__device__ int dev_tomb[5];

__global__ void Szorzas(int num)
{
	int i = threadIdx.x;

	dev_tomb[i] *= num;
}

int main()
{
	cudaMemcpyToSymbol(dev_tomb, tomb, 5 * sizeof(int));
	Szorzas <<< 1, 5 >>> (3);
	cudaMemcpyFromSymbol(tomb, dev_tomb, 5 * sizeof(int));
    return 0;
}
