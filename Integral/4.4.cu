#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#define numThreads 1024

__device__ float funcion_dev(float x) {
	return x / ((x * x) + 4) * sin(1 / x);
}
float funcion_h(float x) {
	return x / ((x * x) + 4) * sin(1 / x);
}
__global__ void kernelIntegral(float a, float h, float* resultados_dev, int N) {
	//El calculo se resume en : (h)*(f(a)/2+f(a+h)+f(a+2h)+...+f(b)/2)
	//Guardaremos cada uno de los elementos de f(a+h)+f(a+2h)+... en un array compartido
	//Sumaremos estos elementos con un algoritmo de reduccion y le sumaremos (f(a)+f(b))/2 multiplicaremos el resultado por  h/2
	__shared__ float elementos[numThreads];
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	float iteracion = (float)threadId + (float)blockId * (float)blockDim.x;
	float suma = 0.0;
	while (iteracion < N) {
		if (iteracion != 0) {
			suma += funcion_dev(a + h * iteracion);
		}
		iteracion += (float)blockDim.x * (float)gridDim.x;
	}
	elementos[threadId] = suma;
	__syncthreads();

	int numElem = numThreads / 2;
	while (numElem != 0) {
		if (threadId < numElem) {
			elementos[threadId] = elementos[threadId] + elementos[threadId + numElem];
		}
		numElem = numElem / 2;
		__syncthreads();
	}
	if (threadId == 0) {
		resultados_dev[blockId] = elementos[0];
	}
}

int main()
{

	int gridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, kernelIntegral, numThreads * 4, 0);

	printf("%d %d\n", gridSize, blockSize);
	int N = gridSize * blockSize;
	printf("%d\n", N);
	float a = 1;
	float b = 3;
	float h = (b - a) / N;
	float* resultado_host = (float*)malloc(sizeof(float) * gridSize);
	float resultado = 0.0;
	float* resultados_dev;
	cudaMalloc((void**)&resultados_dev, sizeof(float) * gridSize);

	kernelIntegral << <gridSize, blockSize >> > (a, h, resultados_dev, N);

	cudaMemcpy(resultado_host, resultados_dev, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);
	for (int i = 0; i < gridSize; i++) {
		resultado += resultado_host[i];
	}
	resultado += (funcion_h(a) + funcion_h(b)) / 2;
	resultado = resultado * h;
	printf("%f\n", resultado);

	free(resultado_host);
	cudaFree(resultados_dev);
}