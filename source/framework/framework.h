#ifndef __NFRAMEWORK_H__
#define __NFRAMEWORK_H__

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#define NF_MAT_AT(m, i, j) (m).data[(i) * (m).cols + (j)]

typedef struct {
    size_t cols;
    size_t rows;
    float *data;
} nf_matrix;

nf_matrix nf_mat_alloc(size_t rows, size_t cols);
void nf_mat_dot(nf_matrix dest, nf_matrix m1, nf_matrix m2);
void nf_mat_sum(nf_matrix dest, nf_matrix m1, nf_matrix m2);
void nf_mat_print(nf_matrix m);

#endif

#ifdef __NFRAMEWORK_IMPL__

nf_matrix nf_mat_alloc(size_t rows, size_t cols) {
    nf_matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = malloc(sizeof(*mat.data) * rows * cols);
    assert(mat.data != NULL);
    memset(mat.data, 0.0f, sizeof(float) * rows * cols);
    return mat;
}
void nf_mat_rand(nf_matrix m) {
    
}
void nf_mat_dot(nf_matrix dest, nf_matrix m1, nf_matrix m2) {
    (void) dest;
    (void) m1;
    (void) m2;
}
void nf_mat_sum(nf_matrix dest, nf_matrix m1, nf_matrix m2) {
    (void) dest;
    (void) m1;
    (void) m2;
}
void nf_mat_print(nf_matrix m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", NF_MAT_AT(m, i, j));
        }
        printf("\n");
    }
}

#endif