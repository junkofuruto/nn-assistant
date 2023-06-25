#define __NFMATRIX_IMPL__

#include "nnf/nf_matrix.h"


int main(void) {
    nf_init();

    nf_matrix l1_w = nf_mat_alloc(2, 2);
    nf_matrix l1_b = nf_mat_alloc(1, 2);
    nf_matrix l2_w = nf_mat_alloc(2, 1);
    nf_matrix l2_b = nf_mat_alloc(1, 1);

    nf_mat_rand(l1_w, -1.0f, 1.0f);
    nf_mat_rand(l1_b, -5.0f, 5.0f);
    nf_mat_rand(l2_w, -1.0f, 1.0f);
    nf_mat_rand(l2_b, -5.0f, 5.0f);

    nf_mat_print(l1_w);
    nf_mat_print(l1_b);
    nf_mat_print(l2_w);
    nf_mat_print(l2_b);



    return 0;
}