
#include "main.cuh"
#include "file_handler.cuh"
#include "user_input_handler.cuh"


/*
* Function that validates the user input. No characters or negative numbers allowed
*/
long long userinput(long long tempInput)
{
    int check;
    int cha;
    int charr;
    check = scanf_s("%lld", &tempInput);
    charr = scanf_s("%c", &cha);
    while ((check != 1) || (tempInput <= 0)) {
        printf("Your input is incorrect\n");
        printf("Please enter an postive integer: ");
        check = scanf_s("%lld", &tempInput);
        charr = scanf_s("%c", &cha);
    };
    return (tempInput);
};
