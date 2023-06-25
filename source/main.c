#define __NFRAMEWORK_IMPL__

#include "framework/framework.h"

int main(void) {
    nf_init();

    nf_matrix mat1 = nf_mat_alloc(1, 2);
    nf_mat_rand(mat1, -1.0f, 1.0f);

    nf_matrix mat2 = nf_mat_alloc(2, 2);
    nf_mat_rand(mat2, -1.0f, 1.0f);

    nf_matrix mat_dest = nf_mat_alloc(1, 2);

    nf_mat_dot(mat_dest, mat1, mat2);
    nf_mat_print(mat1);
    puts("*");
    nf_mat_print(mat2);
    puts("---------------------");
    nf_mat_print(mat_dest);


    return 0;
}