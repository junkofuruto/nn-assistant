#define __NFMATRIX_IMPL__

#include "nnf/nf_matrix.h"

typedef struct {
    nf_matrix l0_a;
    nf_matrix l1_w, l1_b, l1_a;
    nf_matrix l2_w, l2_b, l2_a;
} nf_model;

float forward(nf_model m) {
    nf_mat_dot(m.l1_a, m.l0_a, m.l1_w);
    nf_mat_sum(m.l1_a, m.l1_b);
    nf_mat_apply_sigmoid(m.l1_a);
    
    nf_mat_dot(m.l2_a, m.l1_a, m.l2_w);
    nf_mat_sum(m.l2_a, m.l2_b);
    nf_mat_apply_sigmoid(m.l2_a); 
    return m.l2_a.data[0];    
}

float cost(nf_model m, nf_matrix ti, nf_matrix to) {
    assert(ti.rows == to.rows);
    assert(to.cols == m.l2_a.cols);
    size_t n = ti.rows;
    float c = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        nf_matrix x = nf_mat_row(ti, i);
        nf_matrix y = nf_mat_row(to, i);
        nf_mat_copy(m.l0_a, x);
        forward(m);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = NF_MAT_AT(m.l2_a, 0, j) - NF_MAT_AT(y, 0, j);
            c += d * d;
        }
    }
    return c;
}

int main(void) {
    nf_init();

    nf_model m;

    m.l0_a = nf_mat_alloc(1, 2);

    m.l1_w = nf_mat_alloc(2, 2);
    m.l1_b = nf_mat_alloc(1, 2);
    m.l1_a = nf_mat_alloc(1, 2);

    m.l2_w = nf_mat_alloc(2, 1);
    m.l2_b = nf_mat_alloc(1, 1);
    m.l2_a = nf_mat_alloc(1, 1);

    nf_mat_rand(m.l1_w, -1.0f, 1.0f);
    nf_mat_rand(m.l1_b, -5.0f, 5.0f);
    nf_mat_rand(m.l2_w, -1.0f, 1.0f);
    nf_mat_rand(m.l2_b, -5.0f, 5.0f);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            NF_MAT_AT(m.l0_a, 0, 0) = i;
            NF_MAT_AT(m.l0_a, 0, 1) = j;
            printf("%zu ^ %zu = %f\n", i, j, forward(m));
        }
    }
    

    return 0;
}