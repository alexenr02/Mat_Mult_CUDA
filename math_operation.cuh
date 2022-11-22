
#ifndef __math_operation_cuh
#define __math_operation_cuh

#include "main.cuh"


/*
* Function
*
* params ->
*
* return ->         void
*
*
*/

processStatus_t printArray(uint8_t i, matrix_t matrix_data[]);

/*
* Function that validates the user input. No characters or negative numbers allowed
*
* params ->
*
* return ->         void
*
*
*/
processStatus_t validation_of_matMult(matrix_t matrix_data[]);


/*
* Function that validates the user input. No characters or negative numbers allowed
*
* params ->
*
* return ->         void
*
*
*/
void matrix_transpose(matrix_t matrix_data[]);

/*
* Function that validates the user input. No characters or negative numbers allowed
*
* params ->
*
* return ->         void
*
*
*/
processStatus_t matrix_mult_serial(matrix_t matrix_data[]);





processStatus_t matrix_mult_omp(matrix_t matrix_data[]);






#endif	// __math_operation_cuh
