
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <curand_kernel.h>

#define NumOfRandNumbers 10

void swap(int i, int j);

int randNums[NumOfRandNumbers];
__device__ int dev_randNums[NumOfRandNumbers];

__global__ void generateRandomNumbers(unsigned long seed) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < NumOfRandNumbers)
	{
		// Initialize the cuRAND state
		curandState state;
		curand_init(seed, i, 0, &state);

		// Generate a random integer
		dev_randNums[i] = curand(&state) % 500; // Example: random number between 0 and 99
	}
}

__global__ void evenOddSort() {

}

int main() {

	//generateRandomNumbers << <1, NumOfRandNumbers >> > (time(NULL));
	//cudaMemcpyFromSymbol(randNums, dev_randNums, sizeof(dev_randNums));
	// Print the randNums

	for (int i = 0; i < NumOfRandNumbers; i++) {
		randNums[i] = NumOfRandNumbers - i;
	}
	printf("Random:\n");
	for (int i = 0; i < NumOfRandNumbers; i++) {
		printf("%d ", randNums[i]);
	}




	for (size_t i = 0; i < NumOfRandNumbers / 2; i++)
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
	}

	printf("\n\nSorted:\n");
	// Print the sorted nums
	for (int i = 0; i < NumOfRandNumbers; i++) {
		printf("%d ", randNums[i]);
	}

	return 0;
}
