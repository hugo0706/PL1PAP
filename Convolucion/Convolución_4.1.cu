#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#define WIDTH 16

__global__ void convolucion(int* c_d, int* a_d, int* b_d)
{
    int col = threadIdx.x + 1;
    int row = threadIdx.y + 1;

    for (int dcol = -1; dcol <= 1; dcol++) {
        for (int drow = -1; drow <= 1; drow++) {
            c_d[(col - 1) + ((row - 1) * (WIDTH-2))] += a_d[(col + dcol) + (row + drow) * WIDTH] * b_d[(dcol + 1) + (drow + 1) * 3];
        }
    }
}

void printMatrix(const int* m, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        printf("%s{", (i == 0 ? "{" : " "));
        for (int j = 0; j < cols; j++)
            printf("%s%5d", (j == 0 ? "" : ","), m[j + i * cols]);
        printf("}%s\n", (i == rows - 1 ? "}" : ""));
    }
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

    dim3 blockDim(WIDTH-2,WIDTH-2);

    convolucion << < 1, blockDim >> > (c_d, a_d, b_d);
    cudaMemcpy(c_h, c_d, ((WIDTH - 2) * (WIDTH - 2)) * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Matriz A:\n");
    printMatrix(a_h, 16, 16);
    printf("Matriz B:\n"); 
    printMatrix(b_h, 3, 3);
    printf("Matriz C:\n");
    printMatrix(c_h, 14, 14);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}
