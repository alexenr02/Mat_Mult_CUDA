/**
  * This Program reads two text files, then upload the data of them to dynamic arrays. Then
  * make matrix multiplication using three methods: Serial, CUDA, OpenMP.
  *
  * authors: Alejandro Enriquez
  *			 Otoniel Perez
  *
  */

#include "main.cuh"
#include "file_handler.cuh"
#include "math_operation.cuh"
#include "user_input_handler.cuh"
#include "fort.cuh"

__global__ void kernel(matrix_t matrix_data[], double* array1, double* array2, double* array3)
{
	
	long long row = (blockIdx.y * blockDim.y) + threadIdx.y;
	long long column = (blockIdx.x * blockDim.x) + threadIdx.x;
	double  temp = 0.0;

	if ((row< matrix_data[FIRST_MATRIX].rows) && (column < matrix_data[SECOND_MATRIX].columns))
	{
		for (int i = 0; i < matrix_data[FIRST_MATRIX].columns; i++)
		{
			temp += array1[(row * matrix_data[FIRST_MATRIX].columns) + i] * array2[(i * matrix_data[SECOND_MATRIX].columns) + column];
		}
	}
	array3[(row * matrix_data[RESULT_MATRIX].columns) + column] = temp;
}

int main(void)
{
	double time_serial[5], time_OMP[5];
	float time_CUDA[5];
	cudaError_t cudaStatus;
	//matrix initialization
	matrix_t matrix_data[4] = { 0,0,0,0, NULL };

	if (loadTxts(matrix_data) != Success)
		{
			return Error;
		}
	for (uint8_t time_counter = 0; time_counter < 5; time_counter++)
	{

		/*------------------------------Serial algorithm time------------------------------------*/
																								//
		clock_t start_t, end_t, total_t;														//
		start_t = clock();																		//
		matrix_mult_serial(matrix_data);														//
		end_t = clock();																		//
		total_t = end_t - start_t;																//
		//printf("\nSerial Algorithm: %f ms\n", ((((float)total_t)*1000) / CLOCKS_PER_SEC));	    //
		time_serial[time_counter] = total_t;
		writeElements(matrix_data, SERIAL);														//
																								//
		/*--------------------------------------------------------------------------------------*/

	

		/*---------------------------------OMP algorithm time---------------------------------------------------------------------------*/
																																		//
		int i_omp, j_omp, k_omp;																										//
		omp_set_num_threads(omp_get_num_procs());																						//
		start_t = clock();																												//
		#pragma omp parallel for private(i_omp,j_omp,k_omp) shared(array1,array2,array3)												//
		for (i_omp = 0; i_omp < matrix_data[RESULT_MATRIX_OMP].rows; ++i_omp)															//
		{																																//
			for (j_omp = 0; j_omp < matrix_data[RESULT_MATRIX_OMP].columns; ++j_omp)													//
			{																															//
				for (k_omp = 0; k_omp < matrix_data[FIRST_MATRIX].columns; ++k_omp)														//
				{																														//
					MAT_AND_COORD(RESULT_MATRIX_OMP, i_omp, j_omp) += (MAT_AND_COORD(FIRST_MATRIX, i_omp, k_omp) * MAT_AND_COORD(SECOND_MATRIX, k_omp, j_omp));
				}																														//
			}																															//
		}																																//
		end_t = clock();																												//
		total_t = end_t - start_t;																										//
		//printf("\nOMP Algorithm: %f ms\n", ((((float)total_t) * 1000) / CLOCKS_PER_SEC));												//
		time_OMP[time_counter] = total_t; 
		writeElements(matrix_data, OMP);																								//
																																		//
		/*------------------------------------------------------------------------------------------------------------------------------*/


		//arrays of the host
		double* h_a1 = matrix_data[FIRST_MATRIX].ptrArray;
		double* h_a2 = matrix_data[SECOND_MATRIX].ptrArray;
		double* h_a3 = matrix_data[RESULT_MATRIX].ptrArray;
		//arrays of the device
		matrix_t* d_matrix_data;
		double* d_a1;
		double* d_a2;
		double* d_a3;

		//------------------------------ Allocate memory on the device of the three arrays and validations

		cudaStatus = cudaMalloc((void**)&d_matrix_data, TOTAL_MATRIX * sizeof(matrix_t));
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

		//------------------------------ copy memory from host to Device and validations

		cudaStatus = cudaMemcpy(d_matrix_data, matrix_data, TOTAL_MATRIX * sizeof(matrix_t), cudaMemcpyHostToDevice);
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

		//cudaDeviceProp prop;
		////Declare variables
		//cudaGetDeviceProperties(&prop, 0);
		//int maxBLOCK_SIZE_X = prop.maxThreadsDim[0];
		//int maxBLOCK_SIZE_Y = prop.maxThreadsDim[1];
		//int maxGRID_SIZE_X = prop.maxGridSize[0];
		//int maxGRID_SIZE_Y = prop.maxGridSize[1];
		//int maxThreadsAvailable_x = maxBLOCK_SIZE_X * maxGRID_SIZE_X;
		//int maxThreadsAvailable_y = maxBLOCK_SIZE_Y * maxGRID_SIZE_Y;

		int blockdim = WARP; //Num of threads = WARP size
		dim3 threadsPerBlock(blockdim, blockdim);
		dim3 blocksPerGrid(1,1); // declaration with default value

		blocksPerGrid.x = ceil(double(matrix_data[SECOND_MATRIX].columns)/threadsPerBlock.x);
		blocksPerGrid.y = ceil(double(matrix_data[FIRST_MATRIX].rows)/threadsPerBlock.y);
	
		/*------------------------------CUDA algorithm time-------------------------------------------------------------------------*/
																																	//
		cudaEventRecord(start);																										//
		//Launch the kernel																											//
		kernel << <  blocksPerGrid, threadsPerBlock >> > (d_matrix_data, d_a1, d_a2, d_a3);											//
		cudaEventRecord(stop);																										//
		writeElements(matrix_data, CUDA);																							//
																																	//
		
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

		//printf("\n(Total CUDA time: %lf ms)\n", milliseconds);
		time_CUDA[time_counter] = milliseconds;

		/*---------------------------------------------------------------------------------------------------------------------------*/



		


	Error:
		ending_program

	}
	
	/*-------------------------------COMPARISON OF FILES---------------------------------------------*/


		// opening both file in read only mode
	FILE* res_serial = fopen("matrizC.txt", "r");
	FILE* res_cuda = fopen("matrizC_CUDA.txt", "r");
	FILE* res_omp = fopen("matrizC_OMP.txt", "r");

	if (res_serial == NULL || res_cuda == NULL || res_omp == NULL)
	{
		printf("Error : Files not open");
		exit(0);
	}

	compareFiles(res_serial, res_cuda, res_omp);

	// closing both file
	fclose(res_serial);
	fclose(res_cuda);
	fclose(res_omp);

	//-------------------------------------------------------------------------------------------------|

	/*-------------------------------PRINTING TABLE---------------------------------------------------*/

	double avg[3] = { 0, 0, 0 };

	ft_table_t* table = ft_create_table();
	/* Change border style */
	ft_set_border_style(table, FT_BASIC_STYLE);
	ft_set_cell_prop(table, FT_ANY_ROW, 0, FT_CPROP_TEXT_ALIGN, FT_ALIGNED_CENTER);
	ft_set_cell_prop(table, FT_ANY_ROW, 1, FT_CPROP_TEXT_ALIGN, FT_ALIGNED_LEFT);
	ft_set_cell_prop(table, 0, FT_ANY_COLUMN, FT_CPROP_ROW_TYPE, FT_ROW_HEADER);
	ft_write_ln(table, "Corrida", "Serial (ms)", "CUDA (ms)", "OMP (ms)");
	//ft_write_ln(table, "1", "Ricciardo", "1:25.945", "222.128");
	ft_printf_ln(table, "1|%lf|%f|%lf", time_serial[0], time_CUDA[0], time_OMP[0]);
	ft_printf_ln(table, "2|%lf|%f|%lf", time_serial[1], time_CUDA[1], time_OMP[1]);
	ft_printf_ln(table, "3|%lf|%f|%lf", time_serial[2], time_CUDA[2], time_OMP[2]);
	ft_printf_ln(table, "4|%lf|%f|%lf", time_serial[3], time_CUDA[3], time_OMP[3]);
	ft_printf_ln(table, "5|%lf|%f|%lf", time_serial[4], time_CUDA[4], time_OMP[4]);

	for (int w = 0 ; w < 3; w++)
	{
		switch (w)
		{
			case 0:
				for (int z = 0; z < 5; z++)
				{
					avg[w] += time_serial[z];
				}
				break;
			case 1:
				for (int z = 0; z < 5; z++)
				{
					avg[w] += time_CUDA[z];
				}
				break;
			case 2:
				for (int z = 0; z < 5; z++)
				{
					avg[w] += time_OMP[z];
				}
				break;
			default:
				break;
		}
		
		avg[w] = avg[w] / 5;
	}
	

	ft_printf_ln(table, "Promedio|%lf|%f|%lf", avg[0], avg[1], avg[2]);
	ft_printf_ln(table, "Perc vs Serial|-|%f|%lf",  100*(avg[1]/avg[0]), 100*(avg[2] / avg[0]));

	printf("%s\n", ft_to_string(table));
	ft_destroy_table(table);


	uint8_t winner = 0;
	double smallest = avg[0];
	for (int a = 0; a < 3; a++)
	{
		if (avg[a] < smallest)
		{
			winner = a;
			smallest = avg[a];
		}
	}

	switch (winner)
	{
	case 0:
		printf("\n\nSerial algorithm is Choosen ONE!\n\n");
		break;
	case 1:
		printf("\n\CUDA algorithm is the Choosen ONE!\n\n");
		break;
	case 2:
		printf("\n\nOMP algorithm is Choosen ONE!\n\n");
		break;
	default:
		break;
	}


	//-------------------------------------------------------------------------------------------------|
	free(array1);            
		free(array2);            
		free(array3);            
		free(array4);
	return 0;
}
