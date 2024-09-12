
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

char* word = "ASD";
char* sentence = "FGHASDJKLASDGHJKASD";
int w_len = 3;
int s_len = 19;
int res = -1;

__device__ char* dev_word;
__device__ char* dev_sentence;
__device__ int dev_w_len;
__device__ int dev_s_len;
__device__ int dev_res;

__global__ void FindWord() 
{
	int found = -1;

	for (size_t i = 0; i < dev_s_len; i++)
	{
		int j = 0;
		while (dev_sentence[i + j] == dev_word[j] && j < dev_w_len)
			j++;

		if (j == dev_w_len)
			found = i;
	}
}

int main()
{
	cudaMemcpyToSymbol(dev_word, word, sizeof(word));
	cudaMemcpyToSymbol(dev_sentence, sentence, sizeof(word));
	cudaMemcpyToSymbol(dev_w_len, &w_len, sizeof(int));
	cudaMemcpyToSymbol(dev_s_len, &s_len, sizeof(int));

	FindWord << <1, 1 >> > ();

	cudaMemcpyFromSymbol(&res, dev_res, sizeof(int));

	printf("%d", res);
	
	return 0;
}
