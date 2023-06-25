#ifndef __NFRAMEWORK_H__
#define __NFRAMEWORK_H__

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define NF_MAT_AT(m, i, j) (m).data[(i) * (m).cols + (j)]

typedef struct {
    size_t cols;
    size_t rows;
    float *data;
} nf_matrix;

nf_matrix nf_mat_alloc(size_t rows, size_t cols);
void nf_mat_dot(nf_matrix dest, nf_matrix m1, nf_matrix m2);
void nf_mat_rand(nf_matrix m, float min, float max);
void nf_mat_fill(nf_matrix dest, float value);
void nf_mat_sum(nf_matrix dest, nf_matrix m);
void nf_mat_print(nf_matrix m);
float nf_rand_float();
void nf_init();

#endif

#ifdef __NFRAMEWORK_IMPL__

void nf_init() {
    srand(time(0));
}
nf_matrix nf_mat_alloc(size_t rows, size_t cols) {
    nf_matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = malloc(sizeof(*mat.data) * rows * cols);
    assert(mat.data != NULL);
    memset(mat.data, 0.0f, sizeof(float) * rows * cols);
    return mat;
}
float nf_rand_float() {
    return (float) rand() / (float) RAND_MAX;
}
void nf_mat_rand(nf_matrix m, float min, float max) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            NF_MAT_AT(m, i, j) = nf_rand_float() * (max - min) + min;
        }
    }
}
void nf_mat_fill(nf_matrix dest, float value) {
    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            NF_MAT_AT(dest, i, j) = value;
        }
    }
}
void nf_mat_dot(nf_matrix dest, nf_matrix m1, nf_matrix m2) {
    assert(m1.cols == m2.rows);
    assert(dest.rows == m1.rows);
    assert(dest.cols == m2.cols);

    size_t n = m1.cols;

    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            for (size_t k = 0; k < n; ++k) {
                NF_MAT_AT(dest, i, j) += NF_MAT_AT(m1, i, k) * NF_MAT_AT(m2, k, j);
            }
        }
    }

}
void nf_mat_sum(nf_matrix dest, nf_matrix m) {
    assert(dest.rows == m.rows);
    assert(dest.cols == m.cols);
    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            NF_MAT_AT(dest, i, j) += NF_MAT_AT(m, i, j);
        }
    }
}
void nf_mat_print(nf_matrix m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%9.6f  ", NF_MAT_AT(m, i, j));
        }
        puts("");
    }
}

#endif