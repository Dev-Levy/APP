
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

char word[] = "A";
char sentence[] = "CBAL";
int w_len = 1;
int s_len = 4;
int res = -2;

__device__ char* dev_word;
__device__ char* dev_sentence;
__device__ int dev_w_len;
__device__ int dev_s_len;
__device__ int dev_res;

__global__ void FindWord_1_GPU_CORE()
{
	dev_res = -1;

	for (int i = 0; i <= dev_s_len - dev_w_len; i++)
	{
		int j = 0;

		while (dev_sentence[i + j] == dev_word[j] && j < dev_w_len)
			j++;

		if (j == dev_w_len)
			dev_res = i;
	}
}
__global__ void FindWord_N_GPU_CORE()
{
	dev_res = -1;
	int i = threadIdx.x;
	int j = 0;

	while (dev_sentence[i + j] == dev_word[j] && j < dev_w_len)
		j++;

	if (j == dev_w_len)
		dev_res = i;
}

int main()
{
	cudaMemcpyToSymbol(dev_word, word, w_len * sizeof(char));
	cudaMemcpyToSymbol(dev_sentence, sentence, s_len * sizeof(char));
	cudaMemcpyToSymbol(dev_w_len, &w_len, sizeof(int));
	cudaMemcpyToSymbol(dev_s_len, &s_len, sizeof(int));


	//FindWord_1_GPU_CORE << <1, 1 >> > ();
	FindWord_N_GPU_CORE << <1, s_len - w_len >> > ();


	cudaMemcpyFromSymbol(&res, dev_res, sizeof(int));


	printf("%d", res);

	return 0;
}
