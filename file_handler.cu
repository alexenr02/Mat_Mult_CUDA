
#include "main.cuh"
#include "file_handler.cuh"
#include "user_input_handler.cuh"
#include "math_operation.cuh"

/*
* Function that executes the reading process of the two txt files that contains the arrays of array
* and saves them in dynamic memory arrays of arrays.
*/
processStatus_t loadTxts(matrix_t matrix_data[])
{

    _Post_ _Notnull_ FILE* filePointer; /* declare the file pointer */
    FILE* filePointer_aux = NULL;
    uint8_t readStatus = 0;


    const char* stringArray[NUM_FILES] =        //Setting an array with the names of the two txt files
    {
        MATRIXA_FILENAME,
        MATRIXB_FILENAME
    };


    //loop to read two files that contains many double (8 byte) variables
    for (uint8_t i = 0; i < NUM_FILES; i++)
    {
        printf("\n---Matrix %d---\n", i + 1);
        printf("Give me the rows: ");
        matrix_data[i].rows = userinput(matrix_data[i].rows);
        printf("Give me the columns: ");
        matrix_data[i].columns = userinput(matrix_data[i].columns);
        matrix_data[i].numElements = TOTAL_ELEM_ARRAY;

        if (i == SECOND_MATRIX)
        {
            if ((validation_of_matMult(matrix_data) == Error))
            {
                return Error;
            }

            matrix_data[RESULT_MATRIX].rows = matrix_data[FIRST_MATRIX].rows;
            matrix_data[RESULT_MATRIX].columns = matrix_data[SECOND_MATRIX].columns;
            matrix_data[RESULT_MATRIX].numElements = TOTAL_ELEM_RESULT_MATRIX;
            matrix_data[RESULT_MATRIX].numElementsInTxt = TOTAL_ELEM_RESULT_MATRIX;

            matrix_data[RESULT_MATRIX_OMP].rows = matrix_data[FIRST_MATRIX].rows;
            matrix_data[RESULT_MATRIX_OMP].columns = matrix_data[SECOND_MATRIX].columns;
            matrix_data[RESULT_MATRIX_OMP].numElements = TOTAL_ELEM_RESULT_MATRIX;
            matrix_data[RESULT_MATRIX_OMP].numElementsInTxt = TOTAL_ELEM_RESULT_MATRIX;

        }
        
        errno_t reading = (fopen_s(&filePointer, stringArray[i], READING_MODE)); //Opening file in read mode
        filePointer_aux = filePointer; // save the file pointer read

        //Validation of the file reading process
        if (((reading != Success) || (filePointer == NULL || filePointer == 0)) && NUM_FILES > MAX_FILES)
        {
            printf("Error! Failed to open file!");
            return Error;
        }
        else
        {
            //counts the elements in the two files and saves it in matrix_data.numElementsInTxt for each file. matrix_data.numElementsInTxt contains the num of elements available in the first file
            numElementsTxt(filePointer, i, matrix_data);

            //Verify if the file contains the required num of elements to create the matrix of the size that the user requires
            if (verifySize(matrix_data, i, stringArray) != Success)
            {
                return Error;
            }

            //Proceed to create the dynamic arrays
            convertToDynamic(filePointer, i, matrix_data);

            PRINT_PARAMS(File % s opened and %lld data loaded into dynamic matrix, stringArray[i], TOTAL_ELEM_ARRAY);

            if (fclose(filePointer) != Success || filePointer == NULL || filePointer != filePointer_aux)  /* close the file prior to exiting the routine */
            {
                printf("problemas al cerrar el archivo");
                return Error;
            }

        }


    }
    return Success;
}

static void numElementsTxt(FILE* filePointer, uint8_t i, matrix_t matrix_data[])
{
    double fp;
    //PRINT(Contenido del archivo : );   //debug message
    while (feof(filePointer) == 0)
    {
        if (fscanf_s(filePointer, "%lf", &fp) == Error)
        {
            //PRINT_PARAMS(%lf,fp);         //debug message
            matrix_data[i].numElementsInTxt++;                //counting num of elements that the matrix will can have to allocate that amount of  memory
        }

    }
}


processStatus_t verifySize(matrix_t matrix_data[], uint8_t i, const char* stringArray[NUM_FILES])
{
    if (TOTAL_ELEM_ARRAY > matrix_data[i].numElementsInTxt)
    {
        printf("\n\n ------------------- Error --------------------------------");
        printf("\n\nElements of the text file %s available ----> %lld \n\n", stringArray[i], matrix_data[i].numElementsInTxt);

        printf("Elements asked ----> %lld\n\n", TOTAL_ELEM_ARRAY);
        printf("Error: Elements in the text file are not enough");
        return Error;
    }
    return Success;
}


processStatus_t convertToDynamic(FILE* filePointer, uint8_t i, matrix_t matrix_data[])
{
    /************************ DEBUG MESSAGES ***************************/
                                                                       //
    DEBUG(long long x = 0;)                                            //
        PRINT_PARAMS(array1 tam : % lld, TOTAL_ELEM_ARRAY);                //
    DEBUG(x = (long long)sizeof(double) * TOTAL_ELEM_ARRAY;)           //
        PRINT_PARAMS(tamaño en bytes : % lld, x);                          //
    //
/*******************************************************************/


    switch (i)
    {
        //Fill the first matrix
    case FIRST_MATRIX:
        //dynamic allocation
        array1 = ALIGNED_MALLOC(TOTAL_ELEM_ARRAY, ALIGNMENT_8, double); //      returns a pointer to a block of (TOTAL_ELEM_ARRAY * sizeof(*double)) memory alligned to 8 bytes
        if (array1 != NULL)
        {
            matrix_data[i].ptrArray = array1;
            readElements(filePointer, i, matrix_data);
        }
        else
        {
            printf("Memory not allocated, error ");
            perror("Malloc");

            return Error;
        }
        break;
        //Fill the second matrix
    case SECOND_MATRIX:
        array2 = ALIGNED_MALLOC(TOTAL_ELEM_ARRAY, ALIGNMENT_8, double); //      returns an 8 BYTE aligned block of memory of (total elements * 8 bytes)
        //cudaMallocManaged(&array2, TOTAL_ELEM_ARRAY * sizeof(double));
        array3 = ALIGNED_MALLOC(matrix_data[RESULT_MATRIX].rows*matrix_data[RESULT_MATRIX].columns, ALIGNMENT_8, double); //      returns an 8 BYTE aligned block of memory of (total elements * 8 bytes)
        //cudaMallocManaged(&array3, matrix_data[RESULT_MATRIX].rows * matrix_data[RESULT_MATRIX].columns * sizeof(double));
        array4 = ALIGNED_MALLOC(matrix_data[RESULT_MATRIX].rows * matrix_data[RESULT_MATRIX].columns, ALIGNMENT_8, double);
        DEBUG(printf("Memoria almacenada correctamente: %lld bytes\n", (long long)_aligned_msize(array3, ALIGNMENT_8, 0));)
        if ((array2 != NULL) || (array3 != NULL) || array4 != NULL)
        {
            matrix_data[i].ptrArray = array2;
            matrix_data[RESULT_MATRIX].ptrArray = array3;
            matrix_data[RESULT_MATRIX_OMP].ptrArray = array4;

            //initializing with 0's the third matrix that will allocate the result
            for (long long i = 0; i < matrix_data[RESULT_MATRIX].rows; i++)
            {
                for (long long j = 0; j < matrix_data[RESULT_MATRIX].columns; j++)
                {
                    MAT_AND_COORD(RESULT_MATRIX, i, j) = 0;
                }
            }

            for (long long i = 0; i < matrix_data[RESULT_MATRIX_OMP].rows; i++)
            {
                for (long long j = 0; j < matrix_data[RESULT_MATRIX_OMP].columns; j++)
                {
                    MAT_AND_COORD(RESULT_MATRIX_OMP, i, j) = 0;
                }
            }

            readElements(filePointer, i, matrix_data);

            DEBUG(printf("Memoria almacenada correctamente: %lld bytes\n", (long long)_aligned_msize(array1, ALIGNMENT_8, 0));)
        }
        else
        {
            printf("Memory not allocated, error ");
            perror("Malloc");

            return Error;
        }
        break;

    default:
        printf("cannot be more than 2 matrix");
        return Error;
        break;
    }

    return Success;
}





static void readElements(FILE* filePointer, uint8_t i, matrix_t matrix_data[])
{
    double fp;
    long long max_elem_allowed = TOTAL_ELEM_ARRAY;

    long long r = 0, c = 0;

    rewind(filePointer);
    while ((feof(filePointer) == 0) && max_elem_allowed != 0)
    {
        if (fscanf_s(filePointer, "%lf", &fp) == Error)
        {

            if (r < matrix_data[i].rows)
            {
                if (c < matrix_data[i].columns)
                {
                    MAT_AND_COORD(i, r, c) = fp;
                    //(MATRIX(i)[POSITION(r, c)]) = fp;
                    //(matrix_data[i].ptrArray[(r * matrix_data[i].columns) + c]) = fp;
                    c++;
                    if (c == matrix_data[i].columns)
                    {
                        c = 0;
                        r++;
                    }

                }
            }
        }
        max_elem_allowed--;
    }

    

}





processStatus_t writeElements(matrix_t matrix_data[], uint8_t which_matrix)
{
    FILE* filePointer_result; /* declare the file pointer */
    errno_t reading = Error;
    switch (which_matrix)
    {
        case SERIAL:
            reading = (processStatus_t)(fopen_s(&filePointer_result, "matrizC.txt", W_R_MODE)); //Opening file in write mode
            break;

        case CUDA:
            reading = (processStatus_t)(fopen_s(&filePointer_result, "matrizC_CUDA.txt", W_R_MODE)); //Opening file in write mode
            break;
        
        case OMP:
            reading = (processStatus_t)(fopen_s(&filePointer_result, "matrizC_OMP.txt", W_R_MODE)); //Opening file in write mode
            break;
        
        default:
            reading = Error;
            break;
    }

    
    //Validation of the file reading process
    if (((reading != Success) || (filePointer_result == NULL || filePointer_result == 0)))
    {
        printf("Error! Failed to create and open file!");
        return Error;
    }
    
    for (long long i = 0; i < matrix_data[RESULT_MATRIX].rows; i++)
    {
        for (long long j = 0; j < matrix_data[RESULT_MATRIX].columns; j++)
        {
            fprintf(filePointer_result, "%.10lf\n", MAT_AND_COORD(RESULT_MATRIX, i, j));
        }
    }

    if (fclose(filePointer_result) != Success || filePointer_result == NULL)  /* close the file prior to exiting the routine */
    {
        printf("problemas al cerrar el archivo");
        return Error;
    }
}

void compareFiles(FILE* serialFile, FILE* cudaFile, FILE* ompFile)
{
    // catching characters of the files
    char ch1 = getc(serialFile);
    char ch2 = getc(cudaFile);
    char ch3 = getc(ompFile);

    int error = 0, pos = 0, line = 1;

    // iterate loop till End Of File
    while (ch1 != EOF && ch2 != EOF && ch3 != EOF)
    {
        pos++;

        // if both variable encounters new
        // line then line variable is incremented
        // and pos variable is set to 0
        if (ch1 == '\n' && ch2 == '\n' && ch3 == '\n')
        {
            line++;
            pos = 0;
        }

        // if fetched data is not equal then
        // error is incremented
        if ((ch1 != ch2) || (ch1 != ch3))
        {
            error++;
        }

        // fetching character until end of file
        ch1 = getc(serialFile);
        ch2 = getc(cudaFile);
        ch3 = getc(ompFile);
    }

    if (error != 0)
    {
        printf("\n\n NOT PASS: Results are not equal \n\n");
    }
    else
    {
        printf("\n\n PASS: Results are equal \n\n");
    }

}