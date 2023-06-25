#ifndef __NFMATRIX_H__
#define __NFMATRIX_H__

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define NF_MAT_AT(m, row, col) (m).data[(row) * (m).cols + (col)]

typedef struct {
    size_t cols;
    size_t rows;
    float *data;
} nf_matrix;

nf_matrix nf_mat_alloc(size_t rows, size_t cols);
nf_matrix nf_mat_row(nf_matrix m, size_t row);
void nf_mat_copy(nf_matrix dest, nf_matrix src);
void nf_mat_dot(nf_matrix dest, nf_matrix m1, nf_matrix m2);
void nf_mat_rand(nf_matrix m, float min, float max);
void nf_mat_fill(nf_matrix dest, float value);
void nf_mat_sum(nf_matrix dest, nf_matrix m);
void nf_mat_print(nf_matrix m);
void nf_mat_apply_sigmoid(nf_matrix m);
float nf_rand_float();
void nf_init();

#endif

#ifdef __NFMATRIX_IMPL__

void nf_init() {
    struct timeval tv;
    mingw_gettimeofday(&tv, NULL);
    unsigned long long milliseconds_since_epoch =
        (unsigned long long)(tv.tv_sec) * 1000 +
        (unsigned long long)(tv.tv_usec) / 1000;
    srand((unsigned int)milliseconds_since_epoch);
}
nf_matrix nf_mat_row(nf_matrix m, size_t row) {
    return (nf_matrix) {
        .rows = 1,
        .cols = m.cols,
        .data = &NF_MAT_AT(m, row, 0),
    };
}
void nf_mat_copy(nf_matrix dest, nf_matrix src) {
    assert(dest.rows == src.rows);
    assert(dest.cols == src.cols);

    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            NF_MAT_AT(dest, i, j) = NF_MAT_AT(src, i, j);
        }
    }
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
void nf_mat_apply_sigmoid(nf_matrix m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            NF_MAT_AT(m, i, j) = 1.0f / (1.0f + expf(-NF_MAT_AT(m, i, j)));
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
    printf("{\n");
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %9.6f", NF_MAT_AT(m, i, j));
        }
        puts("");
    }
    puts("}\n");
}

#endif