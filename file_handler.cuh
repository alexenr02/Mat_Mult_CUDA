
#ifndef __file_handler_cuh
#define __file_handler_cuh

#define READING_MODE                        "r"
#define WRITING_MODE                        "w"
#define W_R_MODE							"w+"
#if 0
#define MATRIXA_FILENAME                    "matrizA.txt"
#define MATRIXB_FILENAME                    "matrizB.txt"
#else
#define MATRIXA_FILENAME                    "matrixA1048576.txt"
#define MATRIXB_FILENAME                    "matrixB1048576.txt"
#endif


#define NUM_FILES                           2
#define MAX_FILES                           2

/*
* Function that executes the reading process of the two txt files that contains the arrays of array
* and saves them in dynamic memory arrays of arrays.
*
* params ->		    matrix_data[] Struct that contains data of both arrays, like the size of rows and columns, num of elements in the actual array and in the txt, and pointer to allocated memory
*
* return ->         void
*
*
*/
processStatus_t loadTxts(matrix_t matrix_data[]);


/*
* Function that open the txt file, read it and counts the num of elements that the array can have. This quantity is saved on numElements[]
*
* params ->         filePointer:      pointer to the txt file
*                   i:                counter to identify which of the two txt files we are handling
* return ->         void
*
*
*/
void numElementsTxt(FILE* filePointer, uint8_t i, matrix_t matrix_data[]);

/*
* Function
*
* params ->
*
* return ->         void
*
*
*/

processStatus_t verifySize(matrix_t matrix_data[], uint8_t i, const char* stringArray[NUM_FILES]);

/*
* Function
*
* params ->
*
* return ->         void
*
*
*/

processStatus_t convertToDynamic(FILE* filePointer, uint8_t i, matrix_t matrix_data[]);


/*
* Function 
*
* params ->         
*                   
* return ->         void
*
*
*/

void readElements(FILE* filePointer, uint8_t i, matrix_t matrix_data[]);







/*
* Function
*
* params ->
*
* return ->         void
*
*
*/
processStatus_t writeElements(matrix_t matrix_data[], uint8_t which_matrix);





#endif // __file_handler_h