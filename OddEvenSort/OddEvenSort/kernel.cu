
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <curand_kernel.h>

#define NumOfRandNumbers 10

int randNums[NumOfRandNumbers];
__device__ int dev_randNums[NumOfRandNumbers];

__global__ void generateRandomNumbers(unsigned long seed) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < NumOfRandNumbers)
	{
		curandState state;
		curand_init(seed, i, 0, &state);

		dev_randNums[i] = curand(&state) % 500;
	}
}

__global__ void evenOddSort() {

	for (size_t i = 0; i < NumOfRandNumbers; i++)
	{
		if (i % 2 == 0) //páros
		{
			if (dev_randNums[threadIdx.x * 2] > dev_randNums[(threadIdx.x * 2) + 1]) {
				int temp = dev_randNums[threadIdx.x * 2];
				dev_randNums[threadIdx.x * 2] = dev_randNums[(threadIdx.x * 2) + 1];
				dev_randNums[(threadIdx.x * 2) + 1] = temp;
			}
		}
		if (i % 2 == 1) //páratlan
		{
			if (threadIdx.x != NumOfRandNumbers / 2 - 1)
			{
				if (dev_randNums[(threadIdx.x * 2) + 1] > dev_randNums[(threadIdx.x * 2) + 2]) {
					int temp = dev_randNums[(threadIdx.x * 2) + 1];
					dev_randNums[(threadIdx.x * 2) + 1] = dev_randNums[(threadIdx.x * 2) + 2];
					dev_randNums[(threadIdx.x * 2) + 2] = temp;
				}
			}
		}
		__syncthreads();
	}
}

int main() {

	generateRandomNumbers << <1, NumOfRandNumbers >> > (time(NULL));
	cudaMemcpyFromSymbol(randNums, dev_randNums, NumOfRandNumbers * sizeof(int));


	/*for (int i = 0; i < NumOfRandNumbers; i++) {
		randNums[i] = NumOfRandNumbers - i;
	}*/
	printf("Random:\n");
	for (int i = 0; i < NumOfRandNumbers; i++) {
		printf("%d ", randNums[i]);
	}

	cudaMemcpyToSymbol(dev_randNums, randNums, NumOfRandNumbers * sizeof(int));
	evenOddSort << <1, NumOfRandNumbers / 2 >> > ();
	cudaMemcpyFromSymbol(randNums, dev_randNums, NumOfRandNumbers * sizeof(int));
	//CPU
	/*for (size_t i = 0; i < NumOfRandNumbers; i++)
	{
		if (i % 2 == 0)
		{
			for (size_t j = 0; j < NumOfRandNumbers; j = j + 2)
			{
				if (randNums[j] > randNums[j + 1])
				{
					int temp = randNums[j];
					randNums[j] = randNums[j + 1];
					randNums[j + 1] = temp;
				}
			}
		}
		else
		{
			for (size_t j = 1; j < NumOfRandNumbers - 1; j = j + 2)
			{
				if (randNums[j] > randNums[j + 1])
				{
					int temp = randNums[j];
					randNums[j] = randNums[j + 1];
					randNums[j + 1] = temp;
				}
			}
		}
	}*/

	printf("\n\nSorted:\n");
	// Print the sorted nums
	for (int i = 0; i < NumOfRandNumbers; i++) {
		printf("%d ", randNums[i]);
	}

	return 0;
}
