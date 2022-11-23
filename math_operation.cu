#include "main.cuh"
#include "file_handler.cuh"
#include "math_operation.cuh"


__global__ processStatus_t printArray(uint8_t i, matrix_t matrix_data[])
{
	printf("\n");
	for (long long x = 0; x < MAX_ROWS; x++)
	{
		for (long long y = 0; y < MAX_COLUMNS; y++)
		{
			printf("  %.10lf  \t", MAT_AND_COORD(i, x, y));
		}
		printf("\n");
	}
	return Success;
}

processStatus_t validation_of_matMult(matrix_t matrix_data[])
{
	if (matrix_data[FIRST_MATRIX].columns != matrix_data[SECOND_MATRIX].rows)
	{
		printf("\n\nERROR: These arrays cannot be multiplied.\n");
		printf("Number of Columns of first matrix: %lld\n", matrix_data[FIRST_MATRIX].columns);
		printf("Number of rows of second matrix: %lld\n", matrix_data[SECOND_MATRIX].columns);
		printf("Rows of first matrix must be equal to the columns of the second matrix\n\n");

		return Error;
	}
	return Success;
}


void matrix_transpose(matrix_t matrix_data[], uint8_t which_matrix)
{
	//double aux = 0;

	//matrix_data[TRANSPOSE_MATRIX].ptrArray = ALIGNED_MALLOC(matrix_data[which_matrix].rows * matrix_data[which_matrix].columns, ALIGNMENT_8, double); //      returns an 8 BYTE aligned block of memory of (total elements * 8 bytes)
	//matrix_data[TRANSPOSE_MATRIX].columns = matrix_data[which_matrix].rows;
	//matrix_data[TRANSPOSE_MATRIX].rows = matrix_data[which_matrix].columns;


	//if (matrix_data[TRANSPOSE_MATRIX].ptrArray != NULL)
	//{
	//	for (long long i = 0; i < matrix_data[which_matrix].rows; i++)
	//	{
	//		for (long long j = 0; j < matrix_data[which_matrix].columns; j++)
	//		{
	//			MAT_AND_COORD(TRANSPOSE_MATRIX, j, i) = MAT_AND_COORD(which_matrix, i, j);
	//		}
	//	}
	//}

	//matrix_data[which_matrix].rows = matrix_data[TRANSPOSE_MATRIX].rows;
	//matrix_data[which_matrix].columns = matrix_data[TRANSPOSE_MATRIX].columns;

	//for (long long i = 0; i < matrix_data[which_matrix].rows; i++)
	//{
	//	for (long long j = 0; j < matrix_data[which_matrix].columns; j++)
	//	{
	//		MAT_AND_COORD(which_matrix, i, j) = MAT_AND_COORD(TRANSPOSE_MATRIX, i, j);
	//	}
	//}
	//_aligned_free(matrix_data[TRANSPOSE_MATRIX].ptrArray);
}

processStatus_t matrix_mult_serial(matrix_t matrix_data[])
{
	long long auxK = matrix_data[FIRST_MATRIX].columns;
	for (long long i = 0; i < matrix_data[RESULT_MATRIX].rows; i++)
	{
		for (long long j = 0; j < matrix_data[RESULT_MATRIX].columns; j++)
		{
			for (long long k = 0; k < auxK; k++)
			{
				MAT_AND_COORD(RESULT_MATRIX, i, j) = MAT_AND_COORD(RESULT_MATRIX, i, j) + (MAT_AND_COORD(FIRST_MATRIX,i,k) * MAT_AND_COORD(SECOND_MATRIX, k, j));
			}
		}
	}
	return Success;
}
