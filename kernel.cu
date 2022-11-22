#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <cuda_fp16.h>

#include "main.cuh"
#include "file_handler.cuh"
#include "math_operation.cuh"
#include "user_input_handler.cuh"



__global__ void kernel(matrix_t matrix_data[], double* array1, double* array2, double* array3)
{
	
	long long row = (blockIdx.y * blockDim.y) + threadIdx.y;
	long long column = (blockIdx.x * blockDim.x) + threadIdx.x;
	double  temp = 0.0;

	if ((row< matrix_data[FIRST_MATRIX].rows) && (column < matrix_data[SECOND_MATRIX].columns))
	{
		for (int i = 0; i < matrix_data[FIRST_MATRIX].columns; i++)
		{
			//temp += matrix_data[FIRST_MATRIX].ptrArray[(row * matrix_data[FIRST_MATRIX].columns) + i] * matrix_data[SECOND_MATRIX].ptrArray[(i * matrix_data[SECOND_MATRIX].columns) + column];
			temp += array1[(row * matrix_data[FIRST_MATRIX].columns) + i] * array2[(i * matrix_data[SECOND_MATRIX].columns) + column];
		}
	}
	array3[(row * matrix_data[RESULT_MATRIX].columns) + column] = temp;
}

int main(void)
{
	cudaError_t cudaStatus;
	//matrix initialization
	matrix_t matrix_data[3] = { 0,0,0,0, NULL };

	if (loadTxts(matrix_data) != Success)
	{
		return Error;
	}

	//arrays of the host
	double* h_a1 = matrix_data[FIRST_MATRIX].ptrArray;
	double* h_a2 = matrix_data[SECOND_MATRIX].ptrArray;
	double* h_a3 = matrix_data[RESULT_MATRIX].ptrArray;
	//arrays of the device
	matrix_t* d_matrix_data;
	double* d_a1;
	double* d_a2;
	double* d_a3;

	//Allocate memory on the device of the three arrays
	cudaStatus = cudaMalloc((void**)&d_matrix_data, 3 * sizeof(matrix_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1st array failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&d_a1, matrix_data[FIRST_MATRIX].numElements * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1st array failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_a2, matrix_data[SECOND_MATRIX].numElements * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2nd array failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_a3, matrix_data[RESULT_MATRIX].numElements * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc result array failed!");
		goto Error;
	}

	//copy memory from host to Device

	cudaStatus = cudaMemcpy(d_matrix_data, matrix_data, 3 * sizeof(matrix_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy matrix data failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_a1, h_a1, matrix_data[FIRST_MATRIX].numElements * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 1st array failed!");
		goto Error;
	}
	//copy memory from host to Device
	cudaStatus = cudaMemcpy(d_a2, h_a2, matrix_data[SECOND_MATRIX].numElements * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2nd array failed!");
		goto Error;
	}
	//copy memory from host to Device
	cudaStatus = cudaMemcpy(d_a3, h_a3, matrix_data[RESULT_MATRIX].numElements * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy result array failed!");
		goto Error;
	}


	// Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceProp prop;

	//Declare variables
	cudaGetDeviceProperties(&prop, 0);
	//int num_threads_supported = prop.maxThreadsPerBlock;
	//int num_blocks_supported = prop.maxThreadsDim[0];


	//Grid Size
	//int NUM_BLOCKS = num_blocks_supported;

	//Threadblock size
	//int NUM_THREADS = num_threads_supported;
	//long long total_threads = NUM_BLOCKS * NUM_THREADS;
	long long numThreadsNeeded = matrix_data[RESULT_MATRIX].numElements;

	size_t bytes = numThreadsNeeded * numThreadsNeeded * sizeof(double);
	//printf("%lld \n\n", numThreadsNeeded);
	int maxBLOCK_SIZE_X = prop.maxThreadsDim[0];
	int maxBLOCK_SIZE_Y = prop.maxThreadsDim[1];
	int maxGRID_SIZE_X = prop.maxGridSize[0];
	int maxGRID_SIZE_Y = prop.maxGridSize[1];
	int maxThreadsAvailable_x = maxBLOCK_SIZE_X * maxGRID_SIZE_X;
	int maxThreadsAvailable_y = maxBLOCK_SIZE_Y * maxGRID_SIZE_Y;

	/*printf(" max block size%d,%d \n\n", maxBLOCK_SIZE_X, maxBLOCK_SIZE_Y);
	printf("max grid size %lld,%lld\n\n", maxGRID_SIZE_X, maxGRID_SIZE_Y);*/
	/*int n = 2;

	dim3 blocksPerGrid(1,1);
	dim3 threadsPerBlock(n,n);*/
	int blockdim = 32;
	int m = matrix_data[FIRST_MATRIX].rows;
	int p = matrix_data[SECOND_MATRIX].columns;
	dim3 threadsPerBlock(blockdim, blockdim);
	dim3 blocksPerGrid(1,1);

	blocksPerGrid.x = ceil(double(p)/threadsPerBlock.x);
	blocksPerGrid.y = ceil(double(m)/threadsPerBlock.y);


	// start recording
	cudaEventRecord(start);
	//matrix_mult_serial(matrix_data);
	//Launch the kernel



	kernel << <  blocksPerGrid, threadsPerBlock >> > (d_matrix_data, d_a1, d_a2, d_a3);

	cudaEventRecord(stop);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		goto Error;
	}
	cudaEventSynchronize(stop);

	// Copy data back to host
	cudaStatus = cudaMemcpy(h_a3, d_a3, matrix_data[RESULT_MATRIX].numElements * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	matrix_data[RESULT_MATRIX].ptrArray = h_a3;

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	
	//print_matrix(RESULT_MATRIX);
	printf(" (Total time: %lf)\n", milliseconds);
	writeElements(matrix_data);
Error:
	//De-allocate memory
	cudaFree(d_matrix_data);
	cudaFree(d_a1);
	cudaFree(d_a2);
	cudaFree(d_a3);

	free(array1);
	free(array2);
	free(array3);

	

	ending_program
	return 0;
}
