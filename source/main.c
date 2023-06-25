#define __NFMATRIX_IMPL__

#include "nnf/nf_matrix.h"

typedef struct {
    nf_matrix l0_a;
    nf_matrix l1_w, l1_b, l1_a;
    nf_matrix l2_w, l2_b, l2_a;
} nf_model;

nf_model nf_model_alloc(void) {
    nf_model m;
    m.l0_a = nf_mat_alloc(1, 2);
    m.l1_w = nf_mat_alloc(2, 2);
    m.l1_b = nf_mat_alloc(1, 2);
    m.l1_a = nf_mat_alloc(1, 2);
    m.l2_w = nf_mat_alloc(2, 1);
    m.l2_b = nf_mat_alloc(1, 1);
    m.l2_a = nf_mat_alloc(1, 1);
    return m;
}

void nf_forward(nf_model m) {
    nf_mat_dot(m.l1_a, m.l0_a, m.l1_w);
    nf_mat_sum(m.l1_a, m.l1_b);
    nf_mat_apply_sigmoid(m.l1_a);
    
    nf_mat_dot(m.l2_a, m.l1_a, m.l2_w);
    nf_mat_sum(m.l2_a, m.l2_b);
    nf_mat_apply_sigmoid(m.l2_a);
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
        nf_forward(m);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = NF_MAT_AT(m.l2_a, 0, j) - NF_MAT_AT(y, 0, j);
            c += d * d;
        }
    }
    return c / n;
}

float data[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

void nf_finite_diff(nf_model m, nf_model g, float eps, nf_matrix ti, nf_matrix to) {
    float saved;
    float c = cost(m, ti, to);

    for (size_t i = 0; i < m.l1_w.rows; ++i) {
        for (size_t j = 0; j < m.l1_w.cols; ++j) {
            saved = NF_MAT_AT(m.l1_w, i, j);
            NF_MAT_AT(m.l1_w, i, j) += eps;
            NF_MAT_AT(g.l1_w, i, j) = (cost(m, ti, to) - c) / eps;
            NF_MAT_AT(m.l1_w, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.l1_b.rows; ++i) {
        for (size_t j = 0; j < m.l1_b.cols; ++j) {
            saved = NF_MAT_AT(m.l1_b, i, j);
            NF_MAT_AT(m.l1_b, i, j) += eps;
            NF_MAT_AT(g.l1_b, i, j) = (cost(m, ti, to) - c) / eps;
            NF_MAT_AT(m.l1_b, i, j) = saved;
        }
    }
    
    for (size_t i = 0; i < m.l2_w.rows; ++i) {
        for (size_t j = 0; j < m.l2_w.cols; ++j) {
            saved = NF_MAT_AT(m.l2_w, i, j);
            NF_MAT_AT(m.l2_w, i, j) += eps;
            NF_MAT_AT(g.l2_w, i, j) = (cost(m, ti, to) - c) / eps;
            NF_MAT_AT(m.l2_w, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.l2_b.rows; ++i) {
        for (size_t j = 0; j < m.l2_b.cols; ++j) {
            saved = NF_MAT_AT(m.l2_b, i, j);
            NF_MAT_AT(m.l2_b, i, j) += eps;
            NF_MAT_AT(g.l2_b, i, j) = (cost(m, ti, to) - c) / eps;
            NF_MAT_AT(m.l2_b, i, j) = saved;
        }
    }
}

void nf_learn(nf_model m, nf_model g, float rate) {
    for (size_t i; i < m.l1_w.rows; ++i) {
        for (size_t j; j < m.l1_w.cols; ++j) {
            NF_MAT_AT(m.l1_w, i, j) -= rate * NF_MAT_AT(g.l1_w, i, j);
        }
    }

    for (size_t i; i < m.l1_b.rows; ++i) {
        for (size_t j; j < m.l1_b.cols; ++j) {
            NF_MAT_AT(m.l1_b, i, j) -= rate * NF_MAT_AT(g.l1_b, i, j);
        }
    }
    
    for (size_t i; i < m.l2_w.rows; ++i) {
        for (size_t j; j < m.l2_w.cols; ++j) {
            NF_MAT_AT(m.l2_w, i, j) -= rate * NF_MAT_AT(g.l2_w, i, j);
        }
    }

    for (size_t i; i < m.l2_b.rows; ++i) {
        for (size_t j; j < m.l2_b.cols; ++j) {
            NF_MAT_AT(m.l2_b, i, j) -= rate * NF_MAT_AT(g.l2_b, i, j);
        }
    }
}

int main(void) {
    nf_init();

    size_t n = sizeof(data) / sizeof(data[0]) / 3;
    size_t stride = 3;
    nf_matrix ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .data = data
    };
    
    nf_matrix to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .data = data + 2
    };

    nf_model m = nf_model_alloc();
    nf_model g = nf_model_alloc();
    nf_mat_rand(m.l1_w, 0, 1);
    nf_mat_rand(m.l1_b, 0, 1);
    nf_mat_rand(m.l2_w, 0, 1);
    nf_mat_rand(m.l2_b, 0, 1);
    float c_b = cost(m, ti, to);
    printf("| COST_B: %f\n", c_b);
    nf_finite_diff(m, g, -1, ti, to);
    nf_learn(m, g, 0.1);
    float c_a = cost(m, ti, to);
    printf("| COST_A: %f\n", c_a);
    printf("| COST_DIFF: %f\n", c_a - c_b);
    return 0;
}