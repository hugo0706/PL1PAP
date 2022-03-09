#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#define WIDTH 16
#define TILE 4

__global__ void convolucion(int* c_d, int* a_d, int* b_d)
{
    __shared__ int a_s[TILE + 2][TILE + 2];
    __shared__ int b_s[3][3];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int col = (bx * TILE - 1 + tx);
    int row = (by * TILE - 1 + ty);

    int val = 0;

    if (tx < 3 && ty < 3) {
        b_s[tx][ty] = b_d[tx + ty * 3];
       
    }

    __syncthreads();

    if (col >= 0 && col <= WIDTH - 1 && row >= 0 && row <= WIDTH - 1)
        a_s[tx][ty] = a_d[col + row * WIDTH];

    __syncthreads();

    if ((col > 0 && col < WIDTH - 1 && row > 0 && row < WIDTH - 1) &&
        (tx < TILE + 1 && tx > 0 && ty < TILE + 1 && ty > 0))
    {
        for (int dcol = -1; dcol <= 1; dcol++) {
            for (int drow = -1; drow <= 1; drow++) {
                val += a_s[tx + dcol][ty + drow] * b_s[dcol + 1][(drow + 1)];
            }
        }
        c_d[(col - 1) + (row - 1) * (WIDTH - 2)] = val;
    }
}

void printMatrix(const int* m, const int rows, const int cols)
{
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            printf("%d ", m[i + j * cols]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    //declaramos 
    int* a_h, * b_h, * c_h, * a_d, * b_d, * c_d;

    //reservar mem CPU 
    a_h = (int*)malloc((WIDTH * WIDTH) * sizeof(int));
    b_h = (int*)malloc(9 * sizeof(int));
    c_h = (int*)malloc(((WIDTH - 2) * (WIDTH - 2)) * sizeof(int));

    //reservar mem GPU 
    cudaMalloc(&a_d, (WIDTH * WIDTH) * sizeof(int));
    cudaMalloc(&b_d, 9 * sizeof(int));
    cudaMalloc(&c_d, ((WIDTH - 2) * (WIDTH - 2)) * sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < (WIDTH * WIDTH); i++) {
        a_h[i] = rand()%256;
    }

    for (int i = 0; i < 9; i++) {
        b_h[i] = rand()%10;
    }

    cudaMemcpy(a_d, a_h, (WIDTH * WIDTH) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, 9 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE + 2, TILE + 2);
    dim3 gridDim(WIDTH / TILE, WIDTH / TILE);

    convolucion << < gridDim, blockDim >> > (c_d, a_d, b_d);
    cudaMemcpy(c_h, c_d, ((WIDTH - 2) * (WIDTH - 2)) * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Matriz A:\n");
    printMatrix(a_h, WIDTH, WIDTH);
    printf("Matriz B:\n");
    printMatrix(b_h, 3, 3);
    printf("Matriz C:\n");
    printMatrix(c_h, (WIDTH - 2), (WIDTH - 2));

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}
