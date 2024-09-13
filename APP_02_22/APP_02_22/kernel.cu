
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <windows.h>


int main()
{
	const int W = 600;
	const int H = 400;
	unsigned short img[W][H];


	//making the image, this could be a jpg import later
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			img[i][j] = rand() % 100 + 100;
			printf("\033[48;5;%dm  \033[0m", img[i][j]);
		}
		printf("\n");
	}
	getchar();
	////inside pixels
	//for (size_t i = 0; i < 11; i++)
	//{
	//	for (int i = 1; i < W - 1; i++)
	//	{
	//		for (int j = 1; j < H - 1; j++)
	//		{
	//			int sum = img[i + 1][j] + img[i - 1][j] + img[i][j + 1] + img[i][j - 1] + img[i][j];
	//			img[i][j] = sum / 5;
	//		}
	//	}
	//}
	for (size_t i = 0; i < 7; i++)
	{
		for (int i = 0; i < W; i++)
		{
			//inside
			for (int j = 0; j < H; j++)
			{
				int sum = 0;
				if (i == 0 && j == 0) // bal felso
				{
					sum = img[i + 1][j] + img[i][j + 1] + img[i][j];
					img[i][j] = sum / 3;
				}
				else if (i == 0) // felso sor
				{
					sum = img[i + 1][j] + img[i][j + 1] + img[i][j - 1] + img[i][j];
					img[i][j] = sum / 4;
				}
				else if (j == 0) // bal oldal
				{
					sum = img[i + 1][j] + img[i][j + 1] + img[i][j] + img[i - 1][j];
					img[i][j] = sum / 4;
				}
				else if (j == H) // jobb oldal
				{
					sum = img[i + 1][j] + img[i][j - 1] + img[i][j] + img[i - 1][j];
					img[i][j] = sum / 4;
				}
				else if (i == W) // also oldal
				{
					sum = img[i - 1][j] + img[i][j + 1] + img[i][j] + img[i][j - 1];
					img[i][j] = sum / 4;
				}
				else if (i == 0 && j == H)  // jobb felso
				{
					sum = img[i + 1][j] + img[i][j - 1] + img[i][j];
					img[i][j] = sum / 3;
				}
				else if (i == W && j == H) // jobb also
				{
					sum = img[i - 1][j] + img[i][j - 1] + img[i][j];
					img[i][j] = sum / 3;
				}
				else if (i == W && j == 0) //bal also
				{
					sum = img[i - 1][j] + img[i][j + 1] + img[i][j];
					img[i][j] = sum / 3;
				}
				else
				{
					sum = img[i + 1][j] + img[i - 1][j] + img[i][j + 1] + img[i][j] + img[i][j - 1];
					img[i][j] = sum / 5;
				}
			}


		}
	}

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			printf("\033[48;5;%dm  \033[0m", img[i][j]);
		}
		printf("\n");
	}
}