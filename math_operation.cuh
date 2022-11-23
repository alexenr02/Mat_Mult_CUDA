
#ifndef __math_operation_cuh
#define __math_operation_cuh

#include "main.cuh"


/*
* Function that prints the array dynamically allocated
*
* params ->			i:				which matrix
*					matrix_data[]:  typedef data that contains all the information about the matrix and all the pointers to the arrays
*
* return ->         Sucess
*					Error
*
*
*/

processStatus_t printArray(uint8_t i, matrix_t matrix_data[]);

/*
* Function that validates the user input. No characters or negative numbers allowed
*
* params ->			matrix_data[]:  typedef data that contains all the information about the matrix and all the pointers to the arrays
*
* return ->         Sucess
*					Error
*
*
*/
processStatus_t validation_of_matMult(matrix_t matrix_data[]);


/*
* Function undefined
*
* params ->
*
* return ->         void
*
*
*/
void matrix_transpose(matrix_t matrix_data[]);

/*
* Function that implements the serial matrix multiplication algorithm
*
* params ->			matrix_data[]:  typedef data that contains all the information about the matrix and all the pointers to the arrays
*
* return ->         Sucess
*					Error
*
*
*/
processStatus_t matrix_mult_serial(matrix_t matrix_data[]);



#endif	// __math_operation_cuh
