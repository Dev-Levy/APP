
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctype.h>

char word[] = "z";
char sentence[] = "Az Óbudai Egyetem 2013 végén elhatározta, hogy olyan kurzusokat indít, amelyeknek az a célja, hogy tantermi óra nélkül, csupán a kurzus anyagának otthoni feldolgozásával, határidőket betartva, de ezen belül saját időbeosztás szerint el lehessen sajátítani a tananyagot. Ez a tananyag is ilyen céllal készült.";
const int w_len = 1;
const int s_len = 308;
int res = -2;

bool results[s_len];


__device__ char dev_word[w_len];
__device__ char dev_sentence[s_len];
__device__ int dev_w_len;
__device__ int dev_s_len;
__device__ int dev_res;

__device__ bool dev_results[s_len];


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
__global__ void FindWord_NxM_GPU_CORE() 
{
	dev_results[threadIdx.x] = true;

	if (dev_sentence[threadIdx.x + threadIdx.y] != dev_word[threadIdx.y])
		dev_results[threadIdx.x] = false;
}

int main()
{
	cudaMemcpyToSymbol(dev_word, word, sizeof(dev_word));
	cudaMemcpyToSymbol(dev_sentence, sentence, sizeof(dev_sentence));
	cudaMemcpyToSymbol(dev_w_len, &w_len, sizeof(int));
	cudaMemcpyToSymbol(dev_s_len, &s_len, sizeof(int));


	//FindWord_1_GPU_CORE << <1, 1 >> > ();
	//FindWord_N_GPU_CORE << <1, s_len - w_len + 1 >> > ();
	FindWord_NxM_GPU_CORE << <1, dim3(s_len, w_len) >> > ();


	cudaMemcpyFromSymbol(results, dev_results, sizeof(dev_results));


	for (size_t i = 0; i < sizeof(results); i++)
	{
		if (results[i] == 1)
			printf("Word found at this character: %d\n", i+1);
	}

	return 0;
}
