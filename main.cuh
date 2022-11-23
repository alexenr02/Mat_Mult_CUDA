/**
  * This Program reads two text files, then upload the data of them to dynamic arrays. Then
  * make matrix multiplication using three methods: Serial, CUDA, OpenMP.
  *
  * authors: Alejandro Enriquez
  *			 Otoniel Perez
  *
  */



#ifndef __main_cuh
#define __main_cuh

#include <stdarg.h>
#include <stdio.h>
#include <conio.h>  /* for clrscr */
#include <dos.h>  /* for delay */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <malloc.h>
#include <Windows.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define ALIGNMENT_8                         8
#define WARP                                32

// total amount of elements of the array
#define TOTAL_ELEM_ARRAY                    (matrix_data[i].rows*matrix_data[i].columns)
#define TOTAL_ELEM_RESULT_MATRIX            (matrix_data[FIRST_MATRIX].rows*matrix_data[SECOND_MATRIX].columns)
#define MATRIX(i)                           matrix_data[i].ptrArray
#define POSITION(x,y)                       (x * matrix_data[i].columns) + y

//Useful to operate over a matrix. M = matrix you want to operate, X and Y are the coordinates
#define MAT_AND_COORD(M,X,Y)                matrix_data[M].ptrArray[(X * matrix_data[M].columns) + Y]
#define MAX_ROWS                            matrix_data[i].rows
#define MAX_COLUMNS                         matrix_data[i].columns


#define print_matrix(which_matrix)          printArray(which_matrix, matrix_data);

//De-allocate memory
#define ending_program                      cudaFree(d_matrix_data); \
                                            cudaFree(d_a1);          \
                                            cudaFree(d_a2);          \
                                            cudaFree(d_a3);          \
                                                       
                                            


#define MALLOC_ARRAY(number, type) \
    ((type *)malloc((number) * sizeof(type)))

#define ALIGNED_MALLOC(SIZE, ALIGNMENT, TYPE) \
    (TYPE*)_aligned_malloc(SIZE*sizeof(TYPE*),ALIGNMENT)

//          __mm__malloc   preferencia intrinsicas

//          _aligned_malloc windows

#define CALLOC_ARRAY(elements, type) \
    ((type *)calloc(elements, (elements) * sizeof(type)))


#if 0
#define ENABLE_DEBUG_MESSAGES
#endif


#ifdef  ENABLE_DEBUG_MESSAGES
#define PRINT_PARAMS(value,...)             printf( #value "\n",## __VA_ARGS__);
#define PRINT(value)                        printf( #value "\n");
#define DEBUG(value)                        value
#else
#define PRINT_PARAMS(value,...) 
#define PRINT(value)
#define DEBUG(value)
#endif


typedef enum processStatus
{
    Success,
    Error
}processStatus_t;


//static long long numElements[NUM_FILES];
static double* array1;
static double* array2;
static double* array3;
static double* array4;


//Struct that contains the all the important information about a matrix defined. 
//The data the struct contains is the quantity of rows and columns the matrix has, how many elements it has and how many is the 
//number of elements in its txt file
//The ptrArray points to the beggining of the allocated memory, so It can access the array from this struct.
typedef struct matrix
{
    long long rows;
    long long columns;
    long long numElements;
    long long numElementsInTxt;
    double*   ptrArray;
    
}matrix_t;

enum NUM_MATRIX
{
    FIRST_MATRIX,
    SECOND_MATRIX,
    RESULT_MATRIX,
    RESULT_MATRIX_OMP,
    TOTAL_MATRIX
};

enum MAT_RESULT
{
    SERIAL,
    CUDA,
    OMP
};

typedef struct matrix_wP
{
    long long rows;
    long long columns;
    long long numElements;
    long long numElementsInTxt;
}matrix_t_wP;


#endif // __main_h